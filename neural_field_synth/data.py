import json
import os

import numpy as np
import torch
import torchaudio

from .utils import NSYNTH_NUM_INSTRUMENTS


class NSynthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, overfit: bool = False):
        self.data_path = data_path
        with open(os.path.join(data_path, "examples.json")) as f:
            self.metadata = json.load(f)
        self.keys = list(self.metadata.keys())

        self.overfit = overfit

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        if self.overfit:
            idx = 101

        key = self.keys[idx]
        meta = self.metadata[key]

        instrument_id = meta["instrument"]
        instrument_str = meta["instrument_str"]
        pitch = meta["pitch"]
        velocity = meta["velocity"]

        instrument = np.zeros(NSYNTH_NUM_INSTRUMENTS)
        instrument[instrument_id] = 1

        audio, sr = torchaudio.load(
            os.path.join(self.data_path, "audio", "%s.wav" % key)
        )

        return {
            "audio": audio,
            "sample_rate": sr,
            "instrument": instrument,
            "instrument_str": instrument_str,
            "pitch": pitch,
            "velocity": velocity,
        }
