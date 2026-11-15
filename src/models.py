import torch
import torch.nn as nn
import torch.nn.functional as F
from src.integrity import ComputationalIntegrityChecker


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    if (norm == 0).any():
        return x
    return x / torch.clamp(norm, min=eps)


class SharedProjectionHead(nn.Module):
    """Maps 768-d HT-SAT and BERT local sequences into a 512-d shared semantic space."""

    def __init__(self, audio_in_dim: int = 768, text_in_dim: int = 768, shared_dim: int = 512):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_in_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shared_dim, shared_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_in_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shared_dim, shared_dim),
        )

    def forward(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        f_a = safe_l2_normalize(self.audio_proj(audio_feats), dim=-1)
        f_t = safe_l2_normalize(self.text_proj(text_feats), dim=-1)
        return f_a, f_t


class DirectionalLGMMStream(nn.Module):
    """
    Computes unidirectional Local-to-Global Multiscale Matching (Query -> Context)
    for all pairs (M queries x N contexts) in a batch.
    """

    def __init__(
        self,
        tau_w: float = 0.25,
        lse_lambda: float = 10.0,
        checker: ComputationalIntegrityChecker = None,
    ):
        super().__init__()
        self.tau_w = tau_w
        self.lse_lambda = lse_lambda
        self.checker = checker or ComputationalIntegrityChecker()

    def forward(self, query_seq: torch.Tensor, context_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_seq:   [M, L_q, D] L2-normalized local query embeddings
            context_seq: [N, L_c, D] L2-normalized local context embeddings
        Returns:
            S_g:         [M, N] global directional similarity matrix
        """
        self.checker.check_l2_normalized(query_seq, "query_seq")
        self.checker.check_l2_normalized(context_seq, "context_seq")

        M, L_q, D = query_seq.shape
        N, L_c, _ = context_seq.shape

        # 1. Local-Local Interaction: s_{ij} = f_{q,i}^T f_{c,j} -> [M, N, L_q, L_c]
        sim_matrix = torch.einsum("mld,nkd->mnlk", query_seq, context_seq)
        self.checker.check_cosine_bounds(sim_matrix, "local_local_sim")

        # Normalize s_{ij} across the query length dimension (Eq. 2 in Wang et al.)
        s_bar = sim_matrix / torch.sqrt(torch.sum(sim_matrix ** 2, dim=2, keepdim=True) + 1e-8)

        # Softmax along context length dimension (L_c)
        weights = F.softmax(s_bar / self.tau_w, dim=-1)
        self.checker.check_attention_distribution(weights, "local_attention_weights", dim=-1)

        # 2. Local-Global Interaction: v_{q,i}^c = sum_j w_{ij} f_{c,j} -> [M, N, L_q, D]
        context_aware_query = torch.einsum("mnlk,nkd->mnld", weights, context_seq)
        context_aware_query_norm = safe_l2_normalize(context_aware_query, dim=-1)
        self.checker.check_l2_normalized(context_aware_query_norm, "context_aware_query")

        # Cosine similarity S(f_{q,i}, v_{q,i}^c) -> [M, N, L_q]
        query_expanded = query_seq.unsqueeze(1).expand(M, N, L_q, D)
        local_global_sim = torch.sum(query_expanded * context_aware_query_norm, dim=-1)
        self.checker.check_cosine_bounds(local_global_sim, "local_global_sim")

        # 3. Global-Global Interaction via LogSumExp (LSE) pooling (Eq. 5)
        # S_g = (1 / lambda) * logsumexp(lambda * S(f_{q,i}, v_{q,i}^c), dim=-1)
        S_g = torch.logsumexp(self.lse_lambda * local_global_sim, dim=-1) / self.lse_lambda
        self.checker.check_finite(S_g, "global_directional_S_g")
        return S_g


class BidirectionalLGMM(nn.Module):
    """
    Proposed Symmetric Bidirectional LGMM:
    Computes parallel A->T and T->A multiscale streams and fuses them via
    S_g(A, T) = 0.5 * [S_g(A -> T) + S_g(T -> A)].
    Also computes symmetric intra-modal similarities S_g(A, A) and S_g(T, T) for CMSC.
    """

    def __init__(
        self,
        audio_in_dim: int = 768,
        text_in_dim: int = 768,
        shared_dim: int = 512,
        tau_w: float = 0.25,
        lse_lambda: float = 10.0,
        checker: ComputationalIntegrityChecker = None,
    ):
        super().__init__()
        self.checker = checker or ComputationalIntegrityChecker()
        self.proj = SharedProjectionHead(audio_in_dim, text_in_dim, shared_dim)
        self.a2t_stream = DirectionalLGMMStream(tau_w, lse_lambda, self.checker)
        self.t2a_stream = DirectionalLGMMStream(tau_w, lse_lambda, self.checker)

    def compute_cross_modal_sim(self, f_a: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        s_a2t = self.a2t_stream(f_a, f_t)             # [B, B]
        s_t2a = self.t2a_stream(f_t, f_a).transpose(0, 1)  # [B, B] aligned to (A_m, T_n)
        s_sym = 0.5 * (s_a2t + s_t2a)
        self.checker.check_finite(s_sym, "symmetric_S_g_AT")
        return s_sym

    def compute_intra_modal_sim(self, f_seq: torch.Tensor, stream: DirectionalLGMMStream) -> torch.Tensor:
        s_dir = stream(f_seq, f_seq)
        s_sym = 0.5 * (s_dir + s_dir.transpose(0, 1))
        self.checker.check_finite(s_sym, "symmetric_intra_S_g")
        return s_sym

    def forward(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        f_a, f_t = self.proj(audio_feats, text_feats)
        s_at = self.compute_cross_modal_sim(f_a, f_t)
        s_aa = self.compute_intra_modal_sim(f_a, self.a2t_stream)
        s_tt = self.compute_intra_modal_sim(f_t, self.t2a_stream)
        return {"S_AT": s_at, "S_AA": s_aa, "S_TT": s_tt}


class AsymmetricLGMM(nn.Module):
    """Baseline 1: Original unidirectional LGMM (Wang et al., ICASSP 2024)."""

    def __init__(
        self,
        audio_in_dim: int = 768,
        text_in_dim: int = 768,
        shared_dim: int = 512,
        tau_w: float = 0.25,
        lse_lambda: float = 10.0,
        checker: ComputationalIntegrityChecker = None,
    ):
        super().__init__()
        self.checker = checker or ComputationalIntegrityChecker()
        self.proj = SharedProjectionHead(audio_in_dim, text_in_dim, shared_dim)
        self.stream = DirectionalLGMMStream(tau_w, lse_lambda, self.checker)

    def forward(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        f_a, f_t = self.proj(audio_feats, text_feats)
        s_at = self.stream(f_a, f_t)
        s_aa = self.stream(f_a, f_a)
        s_tt = self.stream(f_t, f_t)
        return {"S_AT": s_at, "S_AA": s_aa, "S_TT": s_tt}


class MMTBaseline(nn.Module):
    """Baseline 2: Lightweight Multi-Modal Transformer global encoder baseline."""

    def __init__(self, audio_in_dim: int = 768, text_in_dim: int = 768, shared_dim: int = 512):
        super().__init__()
        self.proj = SharedProjectionHead(audio_in_dim, text_in_dim, shared_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.audio_tf = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.text_tf = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, audio_feats: torch.Tensor, text_feats: torch.Tensor):
        f_a, f_t = self.proj(audio_feats, text_feats)
        g_a = safe_l2_normalize(self.audio_tf(f_a).mean(dim=1), dim=-1)
        g_t = safe_l2_normalize(self.text_tf(f_t).mean(dim=1), dim=-1)
        return {
            "S_AT": g_a @ g_t.T,
            "S_AA": g_a @ g_a.T,
            "S_TT": g_t @ g_t.T,
        }