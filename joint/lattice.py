from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TopKLattice:
    top_token_ids: torch.Tensor
    top_log_probs: torch.Tensor
    position_entropy: torch.Tensor
    top1_top2_margin: torch.Tensor
    topk_mass: torch.Tensor
    log_z: torch.Tensor

    @property
    def horizon(self) -> int:
        return int(self.top_token_ids.shape[0])

    @property
    def topk(self) -> int:
        return int(self.top_token_ids.shape[1])


@torch.inference_mode()
def extract_topk_lattice(draft_logits: torch.Tensor, topk: int) -> TopKLattice:
    if draft_logits.dim() != 2:
        raise ValueError(f"draft_logits must have shape [H, V], got {tuple(draft_logits.shape)}")
    if draft_logits.shape[0] == 0:
        raise ValueError("draft_logits must contain at least one future position")

    logits = draft_logits.float()
    k = min(int(topk), int(logits.shape[-1]))
    top_logits, top_token_ids = torch.topk(logits, k=k, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = top_logits - log_z

    probs = torch.softmax(logits, dim=-1)
    log_probs = logits - log_z
    position_entropy = -(probs * log_probs).sum(dim=-1)

    if k > 1:
        top1_top2_margin = top_log_probs[:, 0] - top_log_probs[:, 1]
    else:
        top1_top2_margin = torch.full((logits.shape[0],), float("inf"), device=logits.device)

    topk_mass = top_log_probs.exp().sum(dim=-1).clamp(max=1.0)

    return TopKLattice(
        top_token_ids=top_token_ids,
        top_log_probs=top_log_probs,
        position_entropy=position_entropy,
        top1_top2_margin=top1_top2_margin,
        topk_mass=topk_mass,
        log_z=log_z.squeeze(-1),
    )
