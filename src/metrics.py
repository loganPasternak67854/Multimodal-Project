import torch


@torch.no_grad()
def compute_retrieval_metrics(sim_matrix: torch.Tensor) -> dict:
    """
    Args:
        sim_matrix: [N, N] global similarity matrix S_g(A_i, T_j)
    Returns:
        Dictionary containing R@1, R@5, R@10 for A2T and T2A (%)
    """
    N = sim_matrix.size(0)
    gt = torch.arange(N, device=sim_matrix.device).unsqueeze(1)

    # Audio-to-Text (rows = audio queries, cols = text candidates)
    a2t_ranks = torch.argsort(sim_matrix, dim=1, descending=True)
    a2t_hits = (a2t_ranks == gt)

    # Text-to-Audio (rows = text queries, cols = audio candidates)
    t2a_ranks = torch.argsort(sim_matrix.T, dim=1, descending=True)
    t2a_hits = (t2a_ranks == gt)

    def recall_at_k(hits: torch.Tensor, k: int) -> float:
        k_eff = min(k, hits.size(1))
        return 100.0 * hits[:, :k_eff].any(dim=1).float().mean().item()

    return {
        "A2T_R@1": recall_at_k(a2t_hits, 1),
        "A2T_R@5": recall_at_k(a2t_hits, 5),
        "A2T_R@10": recall_at_k(a2t_hits, 10),
        "T2A_R@1": recall_at_k(t2a_hits, 1),
        "T2A_R@5": recall_at_k(t2a_hits, 5),
        "T2A_R@10": recall_at_k(t2a_hits, 10),
    }