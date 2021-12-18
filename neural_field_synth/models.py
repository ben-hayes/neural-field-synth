from collections import namedtuple
from typing import Callable, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchtyping import TensorType

from .neural_fields import SirenFiLM
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
        "wave_film_params",
        "noise_film_params",
    ),
)


class NeuralFieldSynth(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        mlp_hidden_size: int = 1024,
        mlp_hidden_layers: int = 3,
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
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.field_hidden_layers = field_hidden_layers
        self.field_hidden_size = field_hidden_size

        self.wave_field = SirenFiLM(
            3,
            field_hidden_size,
            field_hidden_layers,
            1,
            True,
            wave_field_first_omega_0,
            wave_field_hidden_omega_0,
        )
        self.noise_field = SirenFiLM(
            2,
            field_hidden_size,
            field_hidden_layers,
            noise_ir_length // 2 + 1,
            True,
            noise_field_first_omega_0,
            noise_field_hidden_omega_0,
        )

        self.wave_mlp = MLP(
            4,
            mlp_hidden_size,
            (field_hidden_layers + 1) * field_hidden_size * 2,
            mlp_hidden_layers,
        )
        self.noise_mlp = MLP(
            4,
            mlp_hidden_size,
            (field_hidden_layers + 1) * field_hidden_size * 2,
            mlp_hidden_layers,
        )

        self.instrument_embedding = nn.Linear(NSYNTH_NUM_INSTRUMENTS, 2, bias=False)

        self.noise_synth = FIRNoiseSynth(
            noise_ir_length, noise_window_length, noise_hop_length, noise_window_fn
        )
        self.noise_hop_length = noise_hop_length
        self.noise_ir_length = noise_ir_length

    def _make_instrument_embed(self, instrument):
        instrument_embed = self.instrument_embedding(instrument)
        return instrument_embed

    def _make_film_inputs(self, instrument_embed, pitch, velocity):
        film_mlp_input = torch.cat(
            (
                instrument_embed,
                pitch[..., None] / NSYNTH_MAX_PITCH,
                velocity[..., None] / NSYNTH_MAX_VEL,
            ),
            dim=-1,
        )
        if film_mlp_input.ndim == 2:
            film_mlp_input = film_mlp_input[None]
        return film_mlp_input

    def _get_film_params(self, film_mlp_input):
        wave_film = self.wave_mlp(film_mlp_input).view(
            film_mlp_input.shape[0],
            film_mlp_input.shape[1],
            self.field_hidden_layers + 1,
            self.field_hidden_size,
            2,
        )
        noise_film = self.noise_mlp(film_mlp_input).view(
            film_mlp_input.shape[0],
            film_mlp_input.shape[1],
            self.field_hidden_layers + 1,
            self.field_hidden_size,
            2,
        )

        wave_scale, wave_shift = (
            wave_film[..., 0],
            wave_film[..., 1],
        )
        noise_scale, noise_shift = (
            noise_film[..., 0],
            noise_film[..., 1],
        )

        return wave_scale, wave_shift, noise_scale, noise_shift

    def forward(
        self,
        time: TensorType["time_in_samples", "batch"],
        pitch: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        velocity: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        instrument: TensorType["batch", "instruments"],
        return_params: bool = False,
    ) -> Union[torch.tensor, NeuralFieldSynthOutput]:
        instrument_embed = self._make_instrument_embed(instrument)
        film_mlp_input = self._make_film_inputs(instrument_embed, pitch, velocity)
        wave_scale, wave_shift, noise_scale, noise_shift = self._get_film_params(
            film_mlp_input
        )

        wavetable_spiral = make_wavetable_spiral(
            pitch, self.sample_rate, time, torch.rand_like(pitch) * 2 * np.pi
        )
        fir_sample_signal = make_fir_sample_signal(
            self.noise_hop_length, self.noise_ir_length, time
        )

        wavetable_signal = self.wave_field(wavetable_spiral, wave_scale, wave_shift)
        noise_ir = self.noise_field(fir_sample_signal, noise_scale, noise_shift)
        noise_ir = exp_sigmoid(noise_ir)
        noise_signal = self.noise_synth(noise_ir[..., 0].permute(1, 2, 0))

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
                (wave_scale, wave_shift),
                (noise_scale, noise_shift),
            )


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, loss_fn, learning_rate=1e-3):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn

        self.learning_rate = learning_rate

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

        loss = self.loss_fn(recon.t(), target)
        return loss
