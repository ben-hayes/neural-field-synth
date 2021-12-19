from collections import namedtuple
from typing import Callable, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchtyping import TensorType
import wandb

from .neural_fields import Siren, SirenFiLM
from .signal import FIRNoiseSynth
from .utils import (
    NSYNTH_MAX_PITCH,
    NSYNTH_MAX_VEL,
    NSYNTH_NUM_INSTRUMENTS,
    exp_sigmoid,
    make_fir_sample_signal,
    make_wavetable_spiral,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_layers: int,
        nonlinearity: Callable = nn.ReLU,
    ):
        super().__init__()

        net = [
            nn.Linear(in_features, hidden_features),
            nonlinearity(),
        ]
        for _ in range(hidden_layers):
            net += [
                nn.Linear(hidden_features, hidden_features),
                nonlinearity(),
            ]

        net += [nn.Linear(hidden_features, out_features)]
        self.net = nn.Sequential(*net)

    def forward(
        self, x: TensorType[..., "batch", "in_features"]
    ) -> TensorType[..., "batch", "out_features"]:
        return self.net(x)


NeuralFieldSynthOutput = namedtuple(
    "NeuralFieldSynthOutput",
    (
        "output",
        "wavetable_signal",
        "noise_signal",
        "instrument_embedding",
        "wavetable_spiral",
        "fir_sample_signal",
        "impulse_response",
    ),
)


class NeuralFieldSynth(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        instrument_embedding_size: int = 2,
        field_hidden_size: int = 256,
        field_hidden_layers: int = 3,
        wave_field_first_omega_0: float = 100,
        wave_field_hidden_omega_0: float = 30,
        noise_field_first_omega_0: float = 100,
        noise_field_hidden_omega_0: float = 30,
        noise_ir_length: int = 128,
        noise_window_length: int = 256,
        noise_hop_length: int = 128,
        noise_window_fn: Callable = torch.hann_window,
        freeze_siren: bool = False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.field_hidden_layers = field_hidden_layers
        self.field_hidden_size = field_hidden_size

        self.wave_field = Siren(
            3 + instrument_embedding_size + 2,
            field_hidden_size,
            field_hidden_layers,
            1,
            True,
            wave_field_first_omega_0,
            wave_field_hidden_omega_0,
            freeze=freeze_siren,
        )
        self.noise_field = Siren(
            2 + instrument_embedding_size + 2,
            field_hidden_size,
            field_hidden_layers,
            noise_ir_length // 2 + 1,
            True,
            noise_field_first_omega_0,
            noise_field_hidden_omega_0,
            freeze=freeze_siren,
        )

        self.instrument_embedding = nn.Linear(
            NSYNTH_NUM_INSTRUMENTS, instrument_embedding_size, bias=False
        )

        self.noise_synth = FIRNoiseSynth(
            noise_ir_length, noise_window_length, noise_hop_length, noise_window_fn
        )
        self.noise_hop_length = noise_hop_length
        self.noise_ir_length = noise_ir_length

    def _make_instrument_embed(self, instrument):
        instrument_embed = self.instrument_embedding(instrument)
        return instrument_embed

    def forward(
        self,
        time: TensorType["time_in_samples", "batch"],
        pitch: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        velocity: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        instrument: TensorType["batch", "instruments"],
        return_params: bool = False,
    ) -> Union[torch.tensor, NeuralFieldSynthOutput]:
        instrument_embed = self._make_instrument_embed(instrument)

        return self.forward_with_instrument_embed(
            time, pitch, velocity, instrument_embed, return_params
        )

    def forward_with_instrument_embed(
        self,
        time: TensorType["time_in_samples", "batch"],
        pitch: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        velocity: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        instrument_embed: TensorType["batch", "instrument_embed"],
        return_params: bool = False,
    ) -> Union[torch.tensor, NeuralFieldSynthOutput]:
        wavetable_spiral = make_wavetable_spiral(
            pitch,
            self.sample_rate,
            time,
            torch.rand_like(pitch) * 2 * np.pi
            if self.training
            else torch.zeros_like(pitch),
        )
        fir_sample_signal = make_fir_sample_signal(
            self.noise_hop_length, self.noise_ir_length, time
        )

        instrument_embed_with_pitch_and_velocity = torch.cat(
            (
                instrument_embed,
                pitch[..., None] / NSYNTH_MAX_PITCH,
                velocity[..., None] / NSYNTH_MAX_VEL,
            ),
            dim=-1,
        )
        if instrument_embed_with_pitch_and_velocity.ndim == 2:
            instrument_embed_with_pitch_and_velocity = (
                instrument_embed_with_pitch_and_velocity[None].expand(
                    time.shape[0], -1, -1
                )
            )

        wavetable_input = torch.cat(
            (wavetable_spiral, instrument_embed_with_pitch_and_velocity), dim=-1
        )
        noise_input = torch.cat(
            (
                fir_sample_signal,
                instrument_embed_with_pitch_and_velocity[:: self.noise_hop_length][
                    None
                ].expand(fir_sample_signal.shape[0], -1, -1, -1),
            ),
            dim=-1,
        )

        wavetable_signal = self.wave_field(wavetable_input)
        noise_ir = self.noise_field(noise_input)
        noise_ir = exp_sigmoid(noise_ir)
        noise_signal = self.noise_synth(
            noise_ir[..., 0].permute(1, 2, 0), length=time.shape[0]
        )

        noise_signal = noise_signal[: time.shape[0]]

        output = wavetable_signal[..., 0] + noise_signal

        if not return_params:
            return output
        else:
            return NeuralFieldSynthOutput(
                output,
                wavetable_signal,
                noise_signal,
                instrument_embed,
                wavetable_spiral,
                fir_sample_signal,
                noise_ir,
            )


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, loss_fn, learning_rate=1e-3, log_audio: bool = True):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn

        self.learning_rate = learning_rate

        self.log_audio = log_audio

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        target = batch["audio"].float()
        instrument = batch["instrument"].float()
        pitch = batch["pitch"].float()
        velocity = batch["velocity"].float()

        time = torch.linspace(-1, 1, target.shape[-1], device=target.device)[
            ..., None
        ].expand(-1, target.shape[0])

        recon = self(time, pitch, velocity, instrument)

        loss = self.loss_fn(recon.t(), target[:, 0])
        self.log("loss", loss, on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0 and self.log_audio:
            self._log_audio(target[:, 0].t(), "target")
            self._log_audio(recon, "recon")

        return loss

    def _log_audio(self, audio_batch, caption):
        for i in range(audio_batch.shape[-1]):
            self.logger.experiment.log(
                {
                    "%s_%d"
                    % (caption, i): wandb.Audio(
                        audio_batch[:, i].detach().cpu().numpy(),
                        self.model.sample_rate,
                        "%s_%d" % (caption, i),
                    )
                },
            )
