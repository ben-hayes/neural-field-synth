import json
import os

import numpy as np
import torch
import torchaudio


class NSynthDataset(torch.utils.data.Dataset):
    num_instruments = 1006

    def __init__(self, data_path):
        self.data_path = data_path
        with open(os.path.join(data_path, "examples.json")) as f:
            self.metadata = json.load(f)
        self.keys = list(self.metadata.keys())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        key = self.keys[idx]
        meta = self.metadata[key]

        instrument_id = meta["instrument"]
        pitch = meta["pitch"]
        velocity = meta["velocity"]

        instrument = np.zeros(self.num_instruments)
        instrument[instrument_id] = 1

        audio, sr = torchaudio.load(os.path.join(self.data_path, "%s.wav" % key))

        return {
            "audio": audio,
            "sample_rate": sr,
            "instrument": instrument,
            "pitch": pitch,
            "velocity": velocity,
        }
