import pytest
import torch

from src.integrity import IntegrityViolationError
from src.losses import CMSCLoss
from src.models import DirectionalLGMMStream


def test_l2_normalization_gate_aborts_on_unnormalized_vectors(checker, normalized_batch):
    _, good_t = normalized_batch
    stream = DirectionalLGMMStream(checker=checker)
    bad_a = torch.ones(4, 10, 512) * 2.5

    with pytest.raises(IntegrityViolationError, match="L2NormalizationGate"):
        stream(bad_a, good_t)


def test_zero_vector_gate_aborts_on_zero_embeddings(checker, normalized_batch):
    _, good_t = normalized_batch
    stream = DirectionalLGMMStream(checker=checker)
    zero_a = torch.zeros(4, 10, 512)

    with pytest.raises(IntegrityViolationError, match="ZeroVectorGate"):
        stream(zero_a, good_t)


def test_cosine_bounds_gate_aborts_on_out_of_range_similarity(checker):
    out_of_bounds_sim = torch.ones(4, 4, 10, 8) * 1.5
    with pytest.raises(IntegrityViolationError, match="CosineBoundsGate"):
        checker.check_cosine_bounds(out_of_bounds_sim, "injected_cosine")


def test_attention_distribution_gate_aborts_on_non_unit_softmax(checker):
    bad_weights = torch.full((4, 4, 10, 8), 0.5)
    with pytest.raises(IntegrityViolationError, match="AttentionDistributionGate"):
        checker.check_attention_distribution(bad_weights, "injected_weights")


def test_finite_value_gate_aborts_on_nan_and_inf(checker):
    loss_fn = CMSCLoss(tau=0.07, beta=0.3, checker=checker)
    nan_sim = torch.eye(4)
    nan_sim[1, 2] = float("nan")

    with pytest.raises(IntegrityViolationError, match="FiniteValueGate"):
        loss_fn({"S_AT": nan_sim, "S_AA": torch.eye(4), "S_TT": torch.eye(4)})

    inf_sim = torch.eye(4)
    inf_sim[0, 3] = float("inf")
    with pytest.raises(IntegrityViolationError, match="FiniteValueGate"):
        loss_fn({"S_AT": inf_sim, "S_AA": torch.eye(4), "S_TT": torch.eye(4)})


def test_soft_target_scale_gate_aborts_on_invalid_beta(checker):
    with pytest.raises(IntegrityViolationError, match="SoftTargetScaleGate"):
        CMSCLoss(tau=0.07, beta=1.25, checker=checker)

    with pytest.raises(IntegrityViolationError, match="SoftTargetScaleGate"):
        CMSCLoss(tau=0.07, beta=-0.05, checker=checker)