from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import JointDDTConfig


@dataclass
class EdgeFeatureBatch:
    child_token_ids: torch.Tensor
    parent_token_ids: torch.Tensor
    root_token_ids: torch.Tensor
    depths: torch.Tensor
    ranks: torch.Tensor
    source_ids: torch.Tensor
    scalar_features: torch.Tensor
    parent_token_ids_for_other: torch.Tensor
    root_token_ids_for_other: torch.Tensor
    parent_depths: torch.Tensor
    parent_scalar_features: torch.Tensor
    context_hidden: torch.Tensor | None = None
    parent_context_hidden: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "EdgeFeatureBatch":
        return EdgeFeatureBatch(
            child_token_ids=self.child_token_ids.to(device),
            parent_token_ids=self.parent_token_ids.to(device),
            root_token_ids=self.root_token_ids.to(device),
            depths=self.depths.to(device),
            ranks=self.ranks.to(device),
            source_ids=self.source_ids.to(device),
            scalar_features=self.scalar_features.to(device),
            parent_token_ids_for_other=self.parent_token_ids_for_other.to(device),
            root_token_ids_for_other=self.root_token_ids_for_other.to(device),
            parent_depths=self.parent_depths.to(device),
            parent_scalar_features=self.parent_scalar_features.to(device),
            context_hidden=None if self.context_hidden is None else self.context_hidden.to(device),
            parent_context_hidden=None if self.parent_context_hidden is None else self.parent_context_hidden.to(device),
        )


class NodeValueNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        scalar_feature_dim: int,
        context_hidden_dim: int = 0,
        token_embed_dim: int = 64,
        depth_embed_dim: int = 16,
        source_embed_dim: int = 8,
        hidden_dim: int = 256,
        max_depth: int = 64,
        max_rank: int = 128,
        num_source_types: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_config = {
            "vocab_size": int(vocab_size),
            "scalar_feature_dim": int(scalar_feature_dim),
            "context_hidden_dim": int(context_hidden_dim),
            "token_embed_dim": int(token_embed_dim),
            "depth_embed_dim": int(depth_embed_dim),
            "source_embed_dim": int(source_embed_dim),
            "hidden_dim": int(hidden_dim),
            "max_depth": int(max_depth),
            "max_rank": int(max_rank),
            "num_source_types": int(num_source_types),
            "dropout": float(dropout),
        }

        self.token_embed = nn.Embedding(vocab_size, token_embed_dim)
        self.depth_embed = nn.Embedding(max_depth + 1, depth_embed_dim)
        self.rank_embed = nn.Embedding(max_rank + 1, depth_embed_dim)
        self.source_embed = nn.Embedding(num_source_types, source_embed_dim)
        self.scalar_norm = nn.LayerNorm(scalar_feature_dim)
        self.parent_scalar_norm = nn.LayerNorm(scalar_feature_dim)
        self.context_proj = None
        context_out = 0
        if context_hidden_dim > 0:
            context_out = token_embed_dim
            self.context_proj = nn.Sequential(
                nn.LayerNorm(context_hidden_dim),
                nn.Linear(context_hidden_dim, context_out),
                nn.SiLU(),
            )

        edge_in = (
            3 * token_embed_dim
            + 2 * depth_embed_dim
            + source_embed_dim
            + scalar_feature_dim
            + context_out
        )
        other_in = (
            2 * token_embed_dim
            + depth_embed_dim
            + scalar_feature_dim
            + context_out
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.other_mlp = nn.Sequential(
            nn.Linear(other_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def vocab_size(self) -> int:
        return int(self.model_config["vocab_size"])

    @property
    def scalar_feature_dim(self) -> int:
        return int(self.model_config["scalar_feature_dim"])

    @property
    def context_hidden_dim(self) -> int:
        return int(self.model_config["context_hidden_dim"])

    def validate_vocab_size(self, runtime_vocab_size: int | None) -> None:
        if runtime_vocab_size is None:
            return
        if int(runtime_vocab_size) != self.vocab_size:
            raise ValueError(
                "NodeValueNet vocab_size does not match runtime target vocab. "
                f"checkpoint={self.vocab_size}, runtime={int(runtime_vocab_size)}"
            )

    def _checked_token_ids(self, token_ids: torch.Tensor, name: str) -> torch.Tensor:
        # Runtime candidates come directly from full-vocab top-k logits after
        # checkpoint/runtime vocab validation. Avoid a device sync on every
        # decode round; keep strict checks while training and in unit tests.
        if self.training and token_ids.numel():
            min_token = int(token_ids.min().item())
            max_token = int(token_ids.max().item())
            if min_token < 0 or max_token >= self.vocab_size:
                raise ValueError(
                    f"{name} contains token id outside checkpoint vocab_size. "
                    f"min={min_token}, max={max_token}, vocab_size={self.vocab_size}"
                )
        return token_ids.clamp(0, self.vocab_size - 1)

    def forward(self, batch: EdgeFeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
        param_dtype = self.token_embed.weight.dtype
        child = self.token_embed(self._checked_token_ids(batch.child_token_ids, "child_token_ids"))
        parent = self.token_embed(self._checked_token_ids(batch.parent_token_ids, "parent_token_ids"))
        root = self.token_embed(self._checked_token_ids(batch.root_token_ids, "root_token_ids"))
        depth = self.depth_embed(batch.depths.clamp(0, self.model_config["max_depth"]))
        rank = self.rank_embed(batch.ranks.clamp(0, self.model_config["max_rank"]))
        source = self.source_embed(batch.source_ids.clamp(0, self.model_config["num_source_types"] - 1))
        scalars = self.scalar_norm(batch.scalar_features.to(dtype=param_dtype))

        parts = [child, parent, root, depth, rank, source, scalars]
        other_parts = [
            self.token_embed(self._checked_token_ids(batch.parent_token_ids_for_other, "parent_token_ids_for_other")),
            self.token_embed(self._checked_token_ids(batch.root_token_ids_for_other, "root_token_ids_for_other")),
            self.depth_embed(batch.parent_depths.clamp(0, self.model_config["max_depth"])),
            self.parent_scalar_norm(batch.parent_scalar_features.to(dtype=param_dtype)),
        ]

        if self.context_proj is not None:
            if batch.context_hidden is None:
                raise ValueError("context_hidden is required by this NodeValueNet checkpoint")
            context = self.context_proj(batch.context_hidden.to(dtype=param_dtype))
            parent_source = batch.parent_context_hidden if batch.parent_context_hidden is not None else batch.context_hidden
            parent_context_projected = self.context_proj(parent_source.to(dtype=param_dtype))
            if context.dim() == 1:
                context = context.unsqueeze(0)
            if parent_context_projected.dim() == 1:
                parent_context_projected = parent_context_projected.unsqueeze(0)
            if context.shape[0] == 1:
                edge_context = context.expand(batch.child_token_ids.numel(), -1)
            elif context.shape[0] == batch.child_token_ids.numel():
                edge_context = context
            else:
                raise ValueError(
                    "context_hidden must have batch size 1 or match the number of edges; "
                    f"got {context.shape[0]} and {batch.child_token_ids.numel()}"
                )
            if parent_context_projected.shape[0] == 1:
                parent_context = parent_context_projected.expand(batch.parent_token_ids_for_other.numel(), -1)
            elif parent_context_projected.shape[0] == batch.parent_token_ids_for_other.numel():
                parent_context = parent_context_projected
            else:
                raise ValueError(
                    "parent_context_hidden must have batch size 1 or match the number of parents; "
                    f"got {parent_context_projected.shape[0]} and {batch.parent_token_ids_for_other.numel()}"
                )
            parts.append(edge_context)
            other_parts.append(parent_context)

        edge_input = torch.cat([part.to(dtype=param_dtype) for part in parts], dim=-1)
        other_input = torch.cat([part.to(dtype=param_dtype) for part in other_parts], dim=-1)
        edge_logits = self.edge_mlp(edge_input).squeeze(-1).float()
        other_logits = self.other_mlp(other_input).squeeze(-1).float()
        return edge_logits, other_logits

    def save_checkpoint(
        self,
        path: str | Path,
        joint_config: JointDDTConfig | None = None,
        calibration: dict[str, Any] | None = None,
        hidden_provenance: dict[str, Any] | None = None,
        feature_schema: dict[str, Any] | None = None,
        tokenizer_vocab_size: int | None = None,
    ) -> None:
        payload = {
            "model_config": self.model_config,
            "state_dict": self.state_dict(),
            "joint_config": None if joint_config is None else joint_config.to_dict(),
            "calibration": calibration or {},
            "hidden_provenance": hidden_provenance or {},
            "feature_schema": feature_schema or {},
            "tokenizer_vocab_size": self.vocab_size if tokenizer_vocab_size is None else int(tokenizer_vocab_size),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["NodeValueNet", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location)
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        return model, payload


def load_node_value_net(
    checkpoint: str | Path,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> tuple[NodeValueNet, dict[str, Any]]:
    model, payload = NodeValueNet.from_checkpoint(checkpoint, map_location=device)
    if dtype is not None:
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    model.eval()
    return model, payload
