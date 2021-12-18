from typing import Callable, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from neural_field_synth.signal import FIRNoiseSynth
from torchtyping import TensorType

from .neural_fields import SirenFiLM
from .utils import NSYNTH_NUM_INSTRUMENTS, make_fir_sample_signal, make_wavetable_spiral


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

        net = [nn.Linear(in_features, hidden_features), nonlinearity()]
        for _ in range(hidden_layers):
            net += [nn.Linear(hidden_features, hidden_features), nonlinearity()]

        net += [nn.Linear(hidden_features, out_features)]
        self.net = nn.Sequential(*net)

    def forward(
        self, x: TensorType[..., "batch", "in_features"]
    ) -> TensorType[..., "batch", "out_features"]:
        return self.net(x)


class NeuralFieldSynth(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        mlp_hidden_size: int = 128,
        mlp_hidden_layers: int = 3,
        field_hidden_size: int = 128,
        field_hidden_layers: int = 3,
        wave_field_first_omega_0: int = 30,
        wave_field_hidden_omega_0: int = 30,
        noise_field_first_omega_0: int = 30,
        noise_field_hidden_omega_0: int = 30,
        noise_ir_length: int = 64,
        noise_window_length: int = 256,
        noise_hop_length: int = 128,
        noise_window_fn: Callable = torch.hann_window,
    ):
        super().__init__()

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
            1,
            True,
            noise_field_first_omega_0,
            noise_field_hidden_omega_0,
        )

        self.wave_mlp = MLP(
            4,
            mlp_hidden_size,
            field_hidden_layers * field_hidden_size * 2,
            mlp_hidden_layers,
        )
        self.noise_mlp = MLP(
            4,
            mlp_hidden_size,
            field_hidden_layers * field_hidden_size * 2,
            mlp_hidden_layers,
        )

        self.instrument_embedding = nn.Linear(NSYNTH_NUM_INSTRUMENTS, 2, bias=False)

        self.noise_synth = FIRNoiseSynth(
            noise_ir_length, noise_window_length, noise_hop_length, noise_window_fn
        )

    def forward(
        self,
        time: TensorType["time_in_samples", "batch"],
        pitch: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        velocity: Union[TensorType["batch"], TensorType["time_in_samples", "batch"]],
        instrument: TensorType["batch", "instruments"],
    ) -> torch.tensor:
        instrument_embed = self.instrument_embedding(instrument)
        film_mlp_input = torch.cat((instrument_embed, pitch, velocity), dim=-1)
        wave_film = self.wave_mlp(film_mlp_input)
        noise_film = self.noise_mlp(film_mlp_input)

        wave_scale, wave_shift = (
            wave_film[..., : wave_film.shape[-1] // 2],
            wave_film[..., wave_film.shape[-1] // 2 :],
        )
        noise_scale, noise_shift = (
            noise_film[..., : noise_film.shape[-1] // 2],
            noise_film[..., noise_film.shape[-1] // 2 :],
        )

        wavetable_spiral = make_wavetable_spiral(
            pitch, self.sample_rate, time, torch.rand_like(pitch) * 2 * np.pi
        )
        fir_sample_signal = make_fir_sample_signal(
            self.fir_hop_length, self.fir_ir_length, time
        )

        wavetable_signal = self.wave_field(wavetable_spiral, wave_scale, wave_shift)
        noise_ir = self.noise_field(fir_sample_signal, noise_scale, noise_shift)
        noise_signal = self.noise_synth(noise_ir[..., 0].permute(1, 2, 0))

        noise_signal = noise_signal[: wavetable_signal.shape[0]]

        return noise_signal + wavetable_signal
