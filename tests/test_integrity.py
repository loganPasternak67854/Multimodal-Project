import pytest
import torch
from src.integrity import ComputationalIntegrityChecker, IntegrityViolationError
from src.models import DirectionalLGMMStream
from src.losses import CMSCLoss


def test_unnormalized_vector_triggers_abort():
    checker = ComputationalIntegrityChecker(norm_tol=1e-5)
    stream = DirectionalLGMMStream(checker=checker)
    bad_audio = torch.ones(4, 10, 512) * 3.0
    good_text = torch.nn.functional.normalize(torch.randn(4, 8, 512), dim=-1)

    with pytest.raises(IntegrityViolationError, match="L2NormalizationGate"):
        stream(bad_audio, good_text)


def test_zero_vector_triggers_abort():
    checker = ComputationalIntegrityChecker(norm_tol=1e-5)
    stream = DirectionalLGMMStream(checker=checker)
    zero_audio = torch.zeros(4, 10, 512)
    good_text = torch.nn.functional.normalize(torch.randn(4, 8, 512), dim=-1)

    with pytest.raises(IntegrityViolationError, match="ZeroVectorGate"):
        stream(zero_audio, good_text)


def test_nan_injection_triggers_abort():
    checker = ComputationalIntegrityChecker()
    loss_fn = CMSCLoss(tau=0.07, beta=0.3, checker=checker)
    bad_sim = torch.eye(4)
    bad_sim[0, 1] = float("nan")

    with pytest.raises(IntegrityViolationError, match="FiniteValueGate"):
        loss_fn({"S_AT": bad_sim, "S_AA": torch.eye(4), "S_TT": torch.eye(4)})


def test_invalid_beta_triggers_abort():
    checker = ComputationalIntegrityChecker()
    with pytest.raises(IntegrityViolationError, match="SoftTargetScaleGate"):
        CMSCLoss(tau=0.07, beta=1.25, checker=checker)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])