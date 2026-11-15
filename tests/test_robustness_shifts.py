import numpy as np
import pytest
import torch

from src.dataset import CachedAudioCapsDataset
from src.integrity import IntegrityViolationError
from src.losses import CMSCLoss
from src.models import BidirectionalLGMM


def test_missing_cached_dataset_raises_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cached embeddings not found"):
        CachedAudioCapsDataset(str(tmp_path / "missing_split"))


def test_corrupted_nan_embedding_in_cache_is_caught(tmp_path, checker):
    split_dir = tmp_path / "corrupted_split"
    split_dir.mkdir()
    audio_arr = np.random.randn(4, 16, 64).astype(np.float32)
    text_arr = np.random.randn(4, 12, 64).astype(np.float32)
    audio_arr[2, 5, 10] = np.nan

    np.save(split_dir / "audio_embeddings.npy", audio_arr)
    np.save(split_dir / "text_embeddings.npy", text_arr)

    ds = CachedAudioCapsDataset(str(split_dir))
    a_batch = torch.stack([ds[i][0] for i in range(4)])
    t_batch = torch.stack([ds[i][1] for i in range(4)])

    model = BidirectionalLGMM(audio_in_dim=64, text_in_dim=64, shared_dim=32, checker=checker)
    with pytest.raises(IntegrityViolationError, match="FiniteValueGate"):
        model(a_batch, t_batch)