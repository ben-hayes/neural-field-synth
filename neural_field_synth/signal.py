from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class FIRNoiseSynth(nn.Module):
    def __init__(
        self,
        ir_length: int,
        window_length: int,
        hop_length: int,
        window_fn: Callable = torch.hann_window,
    ):
        super().__init__()
        self.ir_length = ir_length
        self.window_length = window_length
        self.hop_length = hop_length
        self.register_buffer("window", window_fn(ir_length))
        self.register_buffer("inv_window", window_fn(window_length))

    def forward(self, H_re: torch.tensor, length=None) -> torch.tensor:
        # H_re: [time, batch, feat]
        H_im = torch.zeros_like(H_re)
        H_z = torch.complex(H_re, H_im)

        h = torch.fft.irfft(H_z)
        h = h.roll(self.ir_length // 2, -1)
        h = h * self.window.view(1, 1, -1)
        h = F.pad(h, (0, self.window_length - self.ir_length))
        H = torch.fft.rfft(h)

        noise = torch.rand(self.hop_length * H_re.shape[0] - 1, device=H_re.device)
        X = torch.stft(noise, self.window_length, self.hop_length, return_complex=True)
        X = X.unsqueeze(0)
        Y = X * H.permute(1, 2, 0)
        y = torch.istft(
            Y,
            self.window_length,
            self.hop_length,
            length=length,
            window=self.inv_window,
        )
        return y.t()[: H_re.shape[0] * self.hop_length]
