import logging
import torch

logger = logging.getLogger("IntegrityGate")


class IntegrityViolationError(RuntimeError):
    """Raised when a computational integrity action-gate triggers."""
    pass

class ComputationalIntegrityChecker:
    """
    Enforces numerical stability and geometric invariants across the matching pipeline.
    Aborts training immediately with a detailed exception report upon violation.
    """

    def __init__(self, norm_tol: float = 1e-5, strict: bool = True):
        self.norm_tol = norm_tol
        self.strict = strict

    def _abort(self, gate_name: str, message: str):
        report = (
            f"\n{'='*70}\n"
            f"[ACTION-GATE ABORT] Gate: {gate_name}\n"
            f"Details: {message}\n"
            f"{'='*70}"
        )
        logger.error(report)
        raise IntegrityViolationError(report)

    def check_finite(self, tensor: torch.Tensor, name: str):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_cnt = torch.isnan(tensor).sum().item()
            inf_cnt = torch.isinf(tensor).sum().item()
            self._abort(
                "FiniteValueGate",
                f"Tensor '{name}' contains {nan_cnt} NaN(s) and {inf_cnt} Inf(s). "
                f"Shape={tuple(tensor.shape)}"
            )

    def check_l2_normalized(self, tensor: torch.Tensor, name: str):
        """
        Verifies ||z||_2 = 1.0 +/- norm_tol along the last dimension
        and guards against zero-vector collapse.
        """
        self.check_finite(tensor, name)
        norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1)

        if (norms == 0).any():
            self._abort(
                "ZeroVectorGate",
                f"Zero vector detected in '{name}'. Cannot compute safe cosine similarity."
            )

        max_dev = torch.max(torch.abs(norms - 1.0)).item()
        if max_dev > self.norm_tol:
            self._abort(
                "L2NormalizationGate",
                f"Tensor '{name}' violated L2-normalization bounds (||z||_2 = 1.0 +/- {self.norm_tol}). "
                f"Observed max deviation: {max_dev:.3e} (min={norms.min().item():.6f}, max={norms.max().item():.6f})."
            )

    def check_cosine_bounds(self, sim_tensor: torch.Tensor, name: str, eps: float = 1e-5):
        """Verifies cosine similarities lie strictly within [-1 - eps, 1 + eps]."""
        self.check_finite(sim_tensor, name)
        min_val = sim_tensor.min().item()
        max_val = sim_tensor.max().item()
        if min_val < -1.0 - eps or max_val > 1.0 + eps:
            self._abort(
                "CosineBoundsGate",
                f"Cosine similarity '{name}' out of [-1, 1] bounds: min={min_val:.6f}, max={max_val:.6f}."
            )

    def check_attention_distribution(self, weights: torch.Tensor, name: str, dim: int = -1):
        """Verifies attention weights sum to 1.0 along the target context dimension."""
        self.check_finite(weights, name)
        row_sums = weights.sum(dim=dim)
        max_dev = torch.max(torch.abs(row_sums - 1.0)).item()
        if max_dev > self.norm_tol:
            self._abort(
                "AttentionDistributionGate",
                f"Softmax weights '{name}' failed unit-sum check along dim={dim}. "
                f"Max deviation from 1.0: {max_dev:.3e}."
            )

    def check_beta_scale(self, beta: float):
        if not (0.0 <= beta <= 1.0) or torch.isnan(torch.tensor(beta)):
            self._abort(
                "SoftTargetScaleGate",
                f"Invalid soft-target interpolation weight beta={beta}. Must satisfy beta in [0, 1]."
            )