import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedAudioCapsDataset(Dataset):
    """
    Loads cached NumPy embeddings for fast CPU training without heavy encoder passes:
      - audio_embeddings.npy: shape [N, L_a, 768]
      - text_embeddings.npy:  shape [N, L_t, 768]
    """

    def __init__(self, split_dir: str, max_samples: int = None):
        audio_path = os.path.join(split_dir, "audio_embeddings.npy")
        text_path = os.path.join(split_dir, "text_embeddings.npy")

        if not (os.path.exists(audio_path) and os.path.exists(text_path)):
            raise FileNotFoundError(
                f"Cached embeddings not found in '{split_dir}'. Expected "
                f"'audio_embeddings.npy' and 'text_embeddings.npy'."
            )

        self.audio = np.load(audio_path, mmap_mode="r")
        self.text = np.load(text_path, mmap_mode="r")

        if max_samples is not None:
            self.audio = self.audio[:max_samples]
            self.text = self.text[:max_samples]

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.audio[idx], dtype=torch.float32),
            torch.tensor(self.text[idx], dtype=torch.float32),
        )


class SyntheticATRDataset(Dataset):
    """Generates correlated synthetic audio/text sequence pairs for smoke testing."""
    def __init__(self, num_samples: int, l_a: int, l_t: int, d_in: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        concepts = rng.standard_normal((num_samples, 1, d_in)).astype(np.float32)
        self.audio = concepts + 0.35 * rng.standard_normal((num_samples, l_a, d_in)).astype(np.float32)
        self.text = concepts + 0.35 * rng.standard_normal((num_samples, l_t, d_in)).astype(np.float32)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return torch.from_numpy(self.audio[idx]), torch.from_numpy(self.text[idx])