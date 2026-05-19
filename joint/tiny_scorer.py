from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .lattice import TopKLattice
from .tree import SelectedTree


TINY_SCORER_FEATURES = (
    "step_log_prob",
    "position_entropy",
    "top1_top2_margin",
    "topk_mass",
    "depth_norm",
    "rank_norm",
    "step_gap",
)


class TinyScorer(nn.Module):
    INPUT_DIM = len(TINY_SCORER_FEATURES)

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(self.INPUT_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(features.float())).squeeze(-1)

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        payload = {
            "model_type": "TinyScorer",
            "hidden_dim": self.hidden_dim,
            "input_dim": self.INPUT_DIM,
            "feature_names": list(TINY_SCORER_FEATURES),
            "state_dict": self.state_dict(),
            **extra,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["TinyScorer", dict[str, Any]]:
        payload = torch.load(path, map_location=device)
        model = cls(hidden_dim=payload.get("hidden_dim", 32))
        model.load_state_dict(payload["state_dict"])
        return model, payload


def load_tiny_scorer(
    path: str | Path, device: torch.device | str
) -> tuple[TinyScorer, dict[str, Any]]:
    model, payload = TinyScorer.from_checkpoint(path, device=device)
    model = model.to(device)
    model.eval()
    return model, payload


def build_tiny_scorer_features(lattice: TopKLattice) -> torch.Tensor:
    H, K = lattice.top_log_probs.shape
    device = lattice.top_log_probs.device

    step_lp = lattice.top_log_probs.float()
    entropy = lattice.position_entropy.float().unsqueeze(1).expand(H, K)
    margin = lattice.top1_top2_margin.float().unsqueeze(1).expand(H, K)
    mass = lattice.topk_mass.float().unsqueeze(1).expand(H, K)
    depth_norm = (
        torch.arange(1, H + 1, device=device, dtype=torch.float32) / max(H, 1)
    ).unsqueeze(1).expand(H, K)
    rank_norm = (
        torch.arange(K, device=device, dtype=torch.float32) / max(K - 1, 1)
    ).unsqueeze(0).expand(H, K)
    step_gap = step_lp - step_lp[:, 0:1]

    return torch.stack(
        [step_lp, entropy, margin, mass, depth_norm, rank_norm, step_gap], dim=-1
    )


@torch.inference_mode()
def compute_score_log_probs(scorer: TinyScorer, lattice: TopKLattice) -> torch.Tensor:
    features = build_tiny_scorer_features(lattice)
    logits = scorer(features)
    return -torch.nn.functional.softplus(-logits)


_EMPTY_PARENTS = None
_EMPTY_VIS = None


def _make_empty_tree(device: torch.device) -> SelectedTree:
    global _EMPTY_PARENTS, _EMPTY_VIS
    if _EMPTY_PARENTS is None or _EMPTY_PARENTS.device != device:
        _EMPTY_PARENTS = torch.tensor([-1], dtype=torch.long, device=device)
        _EMPTY_VIS = torch.ones((1, 1), dtype=torch.bool, device=device)
    return SelectedTree(
        token_ids=torch.empty(0, dtype=torch.long, device=device),
        depths=torch.empty(0, dtype=torch.long, device=device),
        parents=_EMPTY_PARENTS,
        visibility=_EMPTY_VIS,
        old_to_new=torch.zeros(1, dtype=torch.long, device=device),
        selected_old_node_ids=torch.empty(0, dtype=torch.long, device=device),
        child_maps=[{}],
    )


class _LiteBuffers:
    """Pre-allocated numpy buffers reused across rounds."""
    __slots__ = ("cap", "nd_parent", "nd_token", "nd_depth", "nd_score_cum", "nd_draft_cum")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.nd_parent = np.empty(cap, dtype=np.int32)
        self.nd_token = np.empty(cap, dtype=np.int64)
        self.nd_depth = np.empty(cap, dtype=np.int32)
        self.nd_score_cum = np.empty(cap, dtype=np.float64)
        self.nd_draft_cum = np.empty(cap, dtype=np.float64)


_lite_bufs: _LiteBuffers | None = None


@torch.inference_mode()
def taps_lite_select(
    scorer: TinyScorer,
    lattice: TopKLattice,
    max_pool_nodes: int = 768,
    max_pool_seqs: int = 48,
    max_tree_seqs: int = 64,
    max_tree_nodes: int = 192,
) -> SelectedTree:
    global _lite_bufs

    device = lattice.top_log_probs.device
    H, K = lattice.top_log_probs.shape
    if H == 0:
        return _make_empty_tree(device)

    # --- Phase 1: GPU scoring (~0.1ms) ---
    features = build_tiny_scorer_features(lattice)
    score_lp = -torch.nn.functional.softplus(-scorer(features))

    # --- Phase 2: single bulk GPU→CPU transfer ---
    cpu_data = torch.stack([lattice.top_log_probs.float(), score_lp, lattice.top_token_ids.float()], dim=0).cpu().numpy()

    draft_lp = cpu_data[0]
    score_arr = cpu_data[1]
    token_ids_np = cpu_data[2].astype(np.int64)

    # --- Phase 3: CPU beam search with pre-allocated buffers ---
    cap = min(max_pool_nodes, H * K) + 1
    if _lite_bufs is None or _lite_bufs.cap < cap:
        _lite_bufs = _LiteBuffers(cap)
    buf = _lite_bufs
    buf.nd_parent[0] = -1
    buf.nd_token[0] = -1
    buf.nd_depth[0] = 0
    buf.nd_score_cum[0] = 0.0
    buf.nd_draft_cum[0] = 0.0
    num_nodes = 1

    beam = np.array([0], dtype=np.int32)

    for d in range(H):
        if beam.size == 0 or num_nodes >= cap:
            break

        n_beam = beam.size
        n_children = n_beam * K

        child_score_cum = buf.nd_score_cum[beam, None] + score_arr[d]
        child_draft_cum = buf.nd_draft_cum[beam, None] + draft_lp[d]
        child_parents = np.repeat(beam, K)
        child_tokens = np.tile(token_ids_np[d], n_beam)

        flat_scores = child_score_cum.ravel()
        budget = min(max_pool_seqs, cap - num_nodes, n_children)
        if budget <= 0:
            break

        if budget < n_children:
            top_idx = np.argpartition(flat_scores, -budget)[-budget:]
        else:
            top_idx = np.arange(n_children)

        start_idx = num_nodes
        end_idx = start_idx + top_idx.size
        sl = slice(start_idx, end_idx)
        buf.nd_parent[sl] = child_parents[top_idx]
        buf.nd_token[sl] = child_tokens[top_idx]
        buf.nd_depth[sl] = d + 1
        buf.nd_score_cum[sl] = flat_scores[top_idx]
        buf.nd_draft_cum[sl] = child_draft_cum.ravel()[top_idx]
        num_nodes = end_idx
        beam = np.arange(start_idx, end_idx, dtype=np.int32)

    if num_nodes <= 1:
        return _make_empty_tree(device)

    # --- Phase 4: tree selection (prefix-closed by cum_draft_lp) ---
    non_root = np.arange(1, num_nodes)
    ranked = non_root[np.argsort(-buf.nd_draft_cum[non_root])]

    nd_par = buf.nd_parent
    selected = {0}
    n_leaves = 0
    for nid in ranked:
        if n_leaves >= max_tree_seqs or len(selected) >= max_tree_nodes:
            break
        ancestors = []
        cur = int(nid)
        while cur > 0 and cur not in selected:
            ancestors.append(cur)
            cur = int(nd_par[cur])
        if cur not in selected or len(selected) + len(ancestors) > max_tree_nodes:
            continue
        selected.update(ancestors)
        n_leaves += 1

    selected.discard(0)
    if not selected:
        return _make_empty_tree(device)

    # --- Phase 5: compact + child_maps ---
    sel_sorted = sorted(selected)
    old_to_new_map = {0: 0}
    edge_to_new: dict[tuple[int, int], int] = {}
    kept_tokens: list[int] = []
    kept_depths: list[int] = []
    kept_parents_new: list[int] = []
    child_maps: list[dict[int, int]] = [{}]

    nd_tok = buf.nd_token
    nd_dep = buf.nd_depth
    for old_id in sel_sorted:
        par_old = int(nd_par[old_id])
        if par_old not in old_to_new_map:
            continue
        par_new = old_to_new_map[par_old]
        tok = int(nd_tok[old_id])
        key = (par_new, tok)
        if key in edge_to_new:
            old_to_new_map[old_id] = edge_to_new[key]
            continue
        new_id = len(kept_tokens) + 1
        old_to_new_map[old_id] = new_id
        edge_to_new[key] = new_id
        kept_tokens.append(tok)
        kept_depths.append(int(nd_dep[old_id]))
        kept_parents_new.append(par_new)
        child_maps.append({})
        child_maps[par_new][tok] = new_id

    if not kept_tokens:
        return _make_empty_tree(device)

    # --- Phase 6: visibility on CPU ---
    total = len(kept_tokens) + 1
    vis = np.zeros((total, total), dtype=np.bool_)
    vis[0, 0] = True
    for i in range(1, total):
        vis[i, i] = True
        cur = kept_parents_new[i - 1]
        while cur >= 0:
            vis[i, cur] = True
            if cur == 0:
                break
            cur = kept_parents_new[cur - 1] if cur > 0 else -1

    # --- Phase 7: CPU→GPU ---
    parents_arr = np.array([-1] + kept_parents_new, dtype=np.int64)
    tokens_arr = np.array(kept_tokens, dtype=np.int64)
    depths_arr = np.array(kept_depths, dtype=np.int64)

    return SelectedTree(
        token_ids=torch.from_numpy(tokens_arr).to(device, non_blocking=True),
        depths=torch.from_numpy(depths_arr).to(device, non_blocking=True),
        parents=torch.from_numpy(parents_arr).to(device, non_blocking=True),
        visibility=torch.from_numpy(vis).to(device, non_blocking=True),
        old_to_new=torch.zeros(num_nodes, dtype=torch.long, device=device),
        selected_old_node_ids=torch.empty(0, dtype=torch.long, device=device),
        child_maps=child_maps,
    )
