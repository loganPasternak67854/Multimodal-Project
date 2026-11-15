import torch
import torch.nn as nn
import torch.nn.functional as F
from src.integrity import ComputationalIntegrityChecker


class CMSCLoss(nn.Module):
    """
    Cross-Modal Similarity Consistency (CMSC) Objective:
    L_total = L_InterC + L_Jnt + L_IntraC
    """

    def __init__(
        self,
        tau: float = 0.07,
        beta: float = 0.3,
        checker: ComputationalIntegrityChecker = None,
    ):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.checker = checker or ComputationalIntegrityChecker()
        self.checker.check_beta_scale(self.beta)

    def forward(self, sim_dict: dict) -> dict:
        self.checker.check_beta_scale(self.beta)

        S_AT = sim_dict["S_AT"]  # [B, B]
        S_AA = sim_dict["S_AA"]  # [B, B]
        S_TT = sim_dict["S_TT"]  # [B, B]

        self.checker.check_finite(S_AT, "loss_input_S_AT")
        self.checker.check_finite(S_AA, "loss_input_S_AA")
        self.checker.check_finite(S_TT, "loss_input_S_TT")

        B = S_AT.size(0)
        device = S_AT.device
        targets = torch.arange(B, device=device)

        # Inter-modal Contrastive Learning
        logits_a2t = S_AT / self.tau
        logits_t2a = S_AT.transpose(0, 1) / self.tau
        loss_inter_c = F.cross_entropy(logits_a2t, targets) + F.cross_entropy(logits_t2a, targets)

        # Joint Inter- and Intra-modal Soft Supervision
        Y = torch.eye(B, device=device)
        soft_AA = self.beta * S_AA + (1.0 - self.beta) * Y
        soft_TT = self.beta * S_TT + (1.0 - self.beta) * Y
        self.checker.check_finite(soft_AA, "soft_target_S_AA")
        self.checker.check_finite(soft_TT, "soft_target_S_TT")

        p_soft_AA = F.softmax(soft_AA / self.tau, dim=-1)
        p_soft_TT = F.softmax(soft_TT / self.tau, dim=-1)

        log_p_AT = F.log_softmax(logits_a2t, dim=-1)
        log_p_TA = F.log_softmax(logits_t2a, dim=-1)

        kl_a = F.kl_div(log_p_AT, p_soft_AA, reduction="batchmean")
        kl_t = F.kl_div(log_p_TA, p_soft_TT, reduction="batchmean")
        loss_jnt = 0.5 * (kl_a + kl_t)

        # Intra-modal Contrastive Learning
        diag_pos = torch.diag(S_AT).unsqueeze(1) / self.tau 
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)

        neg_AA = S_AA[off_diag_mask].view(B, B - 1) / self.tau
        neg_TT = S_TT[off_diag_mask].view(B, B - 1) / self.tau

        intra_logits_A = torch.cat([diag_pos, neg_AA], dim=1)
        intra_logits_T = torch.cat([diag_pos, neg_TT], dim=1)
        zero_targets = torch.zeros(B, dtype=torch.long, device=device)

        loss_intra_c = (
            F.cross_entropy(intra_logits_A, zero_targets)
            + F.cross_entropy(intra_logits_T, zero_targets)
        )

        # Total Objective
        loss_total = loss_inter_c + loss_jnt + loss_intra_c
        self.checker.check_finite(loss_total, "loss_total")

        return {
            "total": loss_total,
            "inter_c": loss_inter_c.detach(),
            "jnt": loss_jnt.detach(),
            "intra_c": loss_intra_c.detach(),
        }