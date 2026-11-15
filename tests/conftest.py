import json
import pytest
import torch

from src.integrity import ComputationalIntegrityChecker
from src.models import safe_l2_normalize
from src.telemetry import ExperimentTracker


@pytest.fixture
def checker():
    return ComputationalIntegrityChecker(norm_tol=1e-5, strict=True)


@pytest.fixture
def normalized_batch():
    audio = safe_l2_normalize(torch.randn(4, 10, 512), dim=-1)
    text = safe_l2_normalize(torch.randn(4, 8, 512), dim=-1)
    return audio, text


@pytest.fixture
def valid_receipt_factory(tmp_path):
    def _build(model_type: str = "bi_lgmm", beta: float = 0.3) -> dict:
        cfg = {
            "experiment_name": "pytest_suite",
            "device": "cpu",
            "data": {"cache_dir": "synthetic://test"},
            "loss": {"tau": 0.07, "beta": beta},
            "integrity": {"norm_tol": 1e-5},
        }
        tracker = ExperimentTracker("pytest_suite", model_type, cfg, output_dir=str(tmp_path))
        tracker.start()
        metrics = {
            "A2T_R@1": 40.0, "A2T_R@5": 70.0, "A2T_R@10": 85.0,
            "T2A_R@1": 35.0, "T2A_R@5": 65.0, "T2A_R@10": 80.0,
        }
        receipt_path = tracker.stop_and_save(test_metrics=metrics, num_parameters=1024)
        with open(receipt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _build