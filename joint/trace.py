from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .pool import CandidateTrie, candidate_trie_from_dict, candidate_trie_to_cpu_dict


@dataclass
class HiddenProvenance:
    layer_ids: list[int]
    token_position: str = "runtime_target_hidden_last_available"
    timing: str = "before_current_round_draft_and_verification"
    projection_version: str = "dflash_fc_input"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HiddenProvenance":
        return cls(
            layer_ids=[int(x) for x in data.get("layer_ids", [])],
            token_position=str(data.get("token_position", "runtime_target_hidden_last_available")),
            timing=str(data.get("timing", "before_current_round_draft_and_verification")),
            projection_version=str(data.get("projection_version", "dflash_fc_input")),
        )


def assert_hidden_provenance_matches(trace_provenance: dict[str, Any], runtime_provenance: dict[str, Any]) -> None:
    if HiddenProvenance.from_dict(trace_provenance).to_dict() != HiddenProvenance.from_dict(runtime_provenance).to_dict():
        raise ValueError(
            "target_hidden_proj provenance mismatch between training trace and runtime; "
            f"trace={trace_provenance}, runtime={runtime_provenance}"
        )


@torch.inference_mode()
def extract_target_child_labels(
    target_logits: torch.Tensor,
    trie: CandidateTrie,
) -> dict[str, torch.Tensor]:
    if target_logits.dim() != 3 or target_logits.shape[0] != 1:
        raise ValueError("target_logits must have shape [1, tree_length, vocab]")
    device = target_logits.device
    parent_ids = trie.edge_parent_ids.to(device)
    child_token_ids = trie.token_ids.to(device)
    num_parents = trie.num_total_nodes
    logits_2d = target_logits[0]
    if num_parents > logits_2d.shape[0]:
        raise ValueError("trie parent count exceeds target logits tree length")
    if child_token_ids.numel():
        child_logits = logits_2d[parent_ids, child_token_ids].float()
    else:
        child_logits = torch.empty(0, dtype=torch.float32, device=device)

    log_z_chunks = []
    chunk_size = 64
    for start in range(0, num_parents, chunk_size):
        end = min(num_parents, start + chunk_size)
        log_z_chunks.append(torch.logsumexp(logits_2d[start:end].float(), dim=-1))
    log_z = torch.cat(log_z_chunks, dim=0) if log_z_chunks else torch.empty(0, dtype=torch.float32, device=device)
    child_log_probs = child_logits - log_z.gather(0, parent_ids)
    child_probs = child_log_probs.exp()
    greedy_tokens = logits_2d[:num_parents].argmax(dim=-1)

    target_edge_indices = torch.full((num_parents,), -1, dtype=torch.long, device=device)
    target_edge_probs = child_probs
    target_other_probs = torch.ones((num_parents,), dtype=torch.float32, device=device)
    target_other_probs.scatter_add_(0, parent_ids, -child_probs.float())
    target_other_probs.clamp_(min=0.0, max=1.0)

    edge_indices = torch.arange(trie.num_nodes, dtype=torch.long, device=device)
    greedy_match = child_token_ids == greedy_tokens.gather(0, parent_ids)
    if greedy_match.any():
        matched_parent_ids = parent_ids[greedy_match]
        matched_edges = edge_indices[greedy_match]
        target_edge_indices[matched_parent_ids] = matched_edges

    return {
        "target_child_logits": child_logits.detach().cpu(),
        "target_child_probs": child_probs.detach().cpu(),
        "target_other_probs": target_other_probs.detach().cpu(),
        "target_edge_indices": target_edge_indices.detach().cpu(),
        "target_next_token_per_parent": greedy_tokens.detach().cpu(),
    }


def make_trace_record(
    *,
    lattice_data: dict[str, torch.Tensor],
    trie: CandidateTrie,
    target_labels: dict[str, torch.Tensor],
    root_token_id: int,
    round_start: int,
    target_hidden_proj: torch.Tensor | None,
    hidden_provenance: HiddenProvenance,
    accepted_indices: list[int] | None = None,
    target_greedy_tokens: list[int] | None = None,
    ddtree_accept_length: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "top_token_ids": lattice_data["top_token_ids"].detach().cpu(),
        "top_log_probs": lattice_data["top_log_probs"].detach().cpu(),
        "position_entropy": lattice_data["position_entropy"].detach().cpu(),
        "top1_top2_margin": lattice_data["top1_top2_margin"].detach().cpu(),
        "topk_mass": lattice_data["topk_mass"].detach().cpu(),
        "root_token_id": int(root_token_id),
        "round_start": int(round_start),
        "candidate_trie": candidate_trie_to_cpu_dict(trie),
        "hidden_provenance": hidden_provenance.to_dict(),
        "accepted_indices": [] if accepted_indices is None else [int(x) for x in accepted_indices],
        "target_greedy_tokens": [] if target_greedy_tokens is None else [int(x) for x in target_greedy_tokens],
        "ddtree_accept_length": None if ddtree_accept_length is None else int(ddtree_accept_length),
    }
    if target_hidden_proj is not None:
        record["target_hidden_proj"] = target_hidden_proj.detach().cpu()
    record.update(target_labels)
    return record


def save_trace_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, output_path)


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        records: list[dict[str, Any]] = []
        for file_path in sorted(path.rglob("*.pt")):
            loaded = torch.load(file_path, map_location="cpu")
            if isinstance(loaded, list):
                records.extend(loaded)
            else:
                records.append(loaded)
        return records
    loaded = torch.load(path, map_location="cpu")
    return loaded if isinstance(loaded, list) else [loaded]


def trace_record_to_trie(record: dict[str, Any], device: torch.device | str = "cpu") -> CandidateTrie:
    return candidate_trie_from_dict(record["candidate_trie"], device=device)
