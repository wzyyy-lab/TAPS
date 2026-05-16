from __future__ import annotations

from dataclasses import dataclass

import torch

from .pool import CandidateTrie


@dataclass
class SelectedTree:
    token_ids: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    visibility: torch.Tensor
    old_to_new: torch.Tensor
    selected_old_node_ids: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.token_ids.numel())

    @property
    def current_length(self) -> int:
        return int(self.parents.numel())


def build_visibility_from_parents(parents: torch.Tensor, max_depth: int | None = None) -> torch.Tensor:
    device = parents.device
    total_nodes = int(parents.numel())
    if total_nodes == 0:
        raise ValueError("parents must include at least the root")
    parent_for_gather = parents.clone()
    parent_for_gather[0] = 0
    if max_depth is None:
        max_depth = total_nodes
    max_depth = max(1, int(max_depth))

    ancestors = [torch.arange(total_nodes, dtype=torch.long, device=device)]
    current = ancestors[0]
    for _ in range(max_depth):
        current = parent_for_gather.gather(0, current)
        ancestors.append(current)
    ancestor_matrix = torch.stack(ancestors, dim=1)
    visibility = torch.zeros((total_nodes, total_nodes), dtype=torch.bool, device=device)
    visibility.scatter_(1, ancestor_matrix.clamp(0, total_nodes - 1), True)
    return visibility


def compact_selected_trie(trie: CandidateTrie, selected_mask: torch.Tensor) -> SelectedTree:
    selected_mask = selected_mask.to(device=trie.device, dtype=torch.bool)
    if selected_mask.numel() != trie.num_nodes:
        raise ValueError("selected_mask must have one entry per non-root candidate node")

    selected_old_node_ids = selected_mask.nonzero(as_tuple=False).flatten() + 1
    if selected_old_node_ids.numel() == 0:
        parents = torch.tensor([-1], dtype=torch.long, device=trie.device)
        return SelectedTree(
            token_ids=torch.empty(0, dtype=torch.long, device=trie.device),
            depths=torch.empty(0, dtype=torch.long, device=trie.device),
            parents=parents,
            visibility=torch.ones((1, 1), dtype=torch.bool, device=trie.device),
            old_to_new=torch.zeros((trie.num_total_nodes,), dtype=torch.long, device=trie.device),
            selected_old_node_ids=selected_old_node_ids,
        )

    # Bulk transfer to CPU (single sync instead of per-node .item() calls)
    selected_list = selected_old_node_ids.cpu().tolist()
    parents_cpu = trie.parents.cpu().tolist()
    token_ids_cpu = trie.token_ids.cpu().tolist()

    # Deduplicate on CPU (no GPU syncs in the loop)
    old_to_new_cpu = {0: 0}
    edge_to_new: dict[tuple[int, int], int] = {}
    kept_old_nodes: list[int] = []
    for old_node in selected_list:
        parent_old = parents_cpu[old_node]
        if parent_old not in old_to_new_cpu:
            continue
        parent_new = old_to_new_cpu[parent_old]
        token = token_ids_cpu[old_node - 1]
        key = (parent_new, token)
        if key in edge_to_new:
            old_to_new_cpu[old_node] = edge_to_new[key]
            continue
        old_to_new_cpu[old_node] = len(kept_old_nodes) + 1
        edge_to_new[key] = old_to_new_cpu[old_node]
        kept_old_nodes.append(old_node)

    selected_old_node_ids = torch.tensor(kept_old_nodes, dtype=torch.long, device=trie.device)
    if selected_old_node_ids.numel() == 0:
        parents = torch.tensor([-1], dtype=torch.long, device=trie.device)
        return SelectedTree(
            token_ids=torch.empty(0, dtype=torch.long, device=trie.device),
            depths=torch.empty(0, dtype=torch.long, device=trie.device),
            parents=parents,
            visibility=torch.ones((1, 1), dtype=torch.bool, device=trie.device),
            old_to_new=torch.zeros((trie.num_total_nodes,), dtype=torch.long, device=trie.device),
            selected_old_node_ids=selected_old_node_ids,
        )

    old_to_new = torch.full((trie.num_total_nodes,), -1, dtype=torch.long, device=trie.device)
    mapped_old_nodes = torch.tensor(list(old_to_new_cpu.keys()), dtype=torch.long, device=trie.device)
    mapped_new_nodes = torch.tensor(list(old_to_new_cpu.values()), dtype=torch.long, device=trie.device)
    old_to_new[mapped_old_nodes] = mapped_new_nodes

    old_nonroot = selected_old_node_ids - 1
    parent_new_ids = [old_to_new_cpu.get(parents_cpu[n], -1) for n in kept_old_nodes]
    new_parents_nonroot = torch.tensor(parent_new_ids, dtype=torch.long, device=trie.device)
    if (new_parents_nonroot < 0).any():
        raise ValueError("selected trie is not prefix-closed")

    parents = torch.cat([torch.tensor([-1], dtype=torch.long, device=trie.device), new_parents_nonroot], dim=0)
    depths = trie.depths[old_nonroot]
    visibility = build_visibility_from_parents(parents, max_depth=int(depths.max().item()) if depths.numel() else 1)
    return SelectedTree(
        token_ids=trie.token_ids[old_nonroot],
        depths=depths,
        parents=parents,
        visibility=visibility,
        old_to_new=old_to_new,
        selected_old_node_ids=selected_old_node_ids,
    )


def compile_joint_tree(
    root_token_id: torch.Tensor,
    start: int,
    selected_tree: SelectedTree,
    past_length: int,
    dtype: torch.dtype,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = selected_tree.current_length
    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if selected_tree.num_nodes > 0:
        verify_input_ids[0, 1:current_length].copy_(selected_tree.token_ids, non_blocking=True)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if selected_tree.num_nodes > 0:
        verify_position_ids[0, 1:current_length].copy_(selected_tree.depths, non_blocking=True)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(selected_tree.visibility, non_blocking=True)
    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)
    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_tree_tensorized(selected_tree: SelectedTree, posterior: torch.Tensor) -> tuple[list[int], int]:
    if posterior.dim() != 2 or posterior.shape[0] != 1:
        raise ValueError("posterior must have shape [1, tree_length]")
    if selected_tree.current_length == 1:
        return [0], int(posterior[0, 0].item())

    child_parent_ids = selected_tree.parents[1:]
    child_token_ids = selected_tree.token_ids
    child_indices = torch.arange(1, selected_tree.current_length, dtype=torch.long, device=posterior.device)

    accepted = [0]
    current = torch.tensor(0, dtype=torch.long, device=posterior.device)
    next_token = posterior[0, current]
    max_steps = selected_tree.current_length
    for _ in range(max_steps):
        matches = (child_parent_ids == current) & (child_token_ids == next_token)
        match_idx = matches.nonzero(as_tuple=False)
        if match_idx.numel() == 0:
            return accepted, int(next_token.item())
        current = child_indices[match_idx[0, 0]]
        accepted.append(int(current.item()))
        next_token = posterior[0, current]
    return accepted, int(next_token.item())
