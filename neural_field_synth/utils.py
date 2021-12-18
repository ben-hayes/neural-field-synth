from typing import Union

import numpy as np
import torch
from torchtyping import TensorType

NSYNTH_NUM_INSTRUMENTS = 1006
NSYNTH_MAX_PITCH = 127
NSYNTH_MAX_VEL = 127


def midi_to_hz(note):
    return 2.0 ** ((note - 69.0) / 12.0) * 440.0


def make_wavetable_spiral(
    pitch: Union[
        TensorType["batch"],
        TensorType["time_in_samples", "batch"],
    ],
    sample_rate: float,
    t: TensorType["time_in_samples", "batch"],
    initial_phase: TensorType["batch"],
):
    """Convert pitch and time envelopes to a phase spiral to index the wavespace.
    Calculations performed in 64-bit to avoid cumsum rounding artefacts."""
    freq = midi_to_hz(pitch).double()
    if pitch.ndim == 1:
        pitch = pitch[None, ...]

    if pitch.ndim == 2 and pitch.shape[-1] == 1:
        freq = freq.expand_as(t)

    phase = freq.cumsum(dim=0) * 2 * np.pi / sample_rate + initial_phase[None]

    x, y = torch.cos(phase), torch.sin(phase)

    return torch.stack((x, y, t), dim=-1).float()


def make_fir_sample_signal(
    hop_length: int, ir_length: int, t: TensorType["time_in_samples", "batch"]
) -> TensorType["freq_bins", "time_in_hops", "batch", "features"]:
    time_steps = t[::hop_length][None, ..., None]
    ir_axis = torch.linspace(0, 1, ir_length // 2 + 1, device=t.device)[
        :, None, None, None
    ]
    ir_axis = ir_axis.expand([ir_axis.shape[0]] + list(time_steps.shape[1:]))

    return torch.cat((time_steps, ir_axis), dim=-1)
