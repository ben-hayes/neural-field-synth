import json
import os

import numpy as np
import torch
import torchaudio

from .utils import NSYNTH_NUM_INSTRUMENTS


class NSynthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        with open(os.path.join(data_path, "examples.json")) as f:
            self.metadata = json.load(f)
        self.keys = list(self.metadata.keys())

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        meta = self.metadata[key]

        instrument_id = meta["instrument"]
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
            "pitch": pitch,
            "velocity": velocity,
        }
