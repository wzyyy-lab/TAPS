from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JointDDTConfig:
    proposal_mode: str = "joint"
    joint_topk: int = 32
    candidate_pool_nodes: int = 2048
    candidate_pool_sequences: int = 256
    candidate_pool_source: str = "union"
    min_verify_sequences: int = 4
    max_verify_sequences: int = 64
    min_verify_nodes: int = 16
    max_verify_nodes: int = 192
    min_useful_verify_nodes: int = 8
    target_accept_drop: float = 0.03
    hard_accept_drop_limit: float = 0.05
    fallback_to_ddtree: bool = True
    fallback_backend: str = "gpu_marginal"
    max_pool_build_overhead_ratio: float = 0.25
    latency_gate_warmup_rounds: int = 4
    latency_gate_small_tree_nodes: int = 32
    max_fallback_rate: float = 0.25
    high_confidence_q: float = 0.65
    low_confidence_q: float = 0.20
    low_topk_mass: float = 0.90
    min_best_reach_for_joint: float = 1e-4
    min_calibration_confidence: float = 0.0
    utility_threshold: float = 0.0
    lambda_node: float = 1.0
    lambda_quadratic: float = 0.002
    lambda_prompt: float = 0.0
    per_parent_child_cap: int = 16
    selection_top_multiplier: int = 4
    enable_value_beam_pool: bool = True
    enable_diversity_pool: bool = True
    enable_entropy_pool: bool = True
    marginal_pool_fraction: float = 0.55
    value_pool_fraction: float = 0.20
    diversity_pool_fraction: float = 0.15
    entropy_pool_fraction: float = 0.10
    diversity_first_token_cap_fraction: float = 0.50
    entropy_bonus: float = 0.35
    value_beam_logit_weight: float = 0.75
    max_depth: int | None = None
    feature_dtype: str = "float32"
    debug_force_cpu_heap: bool = False
    hidden_provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.proposal_mode not in {"joint", "ddtree", "dflash", "all"}:
            raise ValueError(f"Unsupported proposal_mode: {self.proposal_mode}")
        if self.joint_topk <= 0:
            raise ValueError("joint_topk must be positive")
        if self.candidate_pool_nodes <= 0:
            raise ValueError("candidate_pool_nodes must be positive")
        if self.candidate_pool_sequences <= 0:
            raise ValueError("candidate_pool_sequences must be positive")
        if self.candidate_pool_source not in {"union", "ddtree_heap"}:
            raise ValueError("candidate_pool_source must be union or ddtree_heap")
        if self.max_verify_nodes < 0:
            raise ValueError("max_verify_nodes must be non-negative")
        if self.min_verify_nodes < 0:
            raise ValueError("min_verify_nodes must be non-negative")
        if self.min_verify_nodes > self.max_verify_nodes:
            raise ValueError("min_verify_nodes cannot exceed max_verify_nodes")
        if self.max_verify_sequences <= 0:
            raise ValueError("max_verify_sequences must be positive")
        if self.min_verify_sequences <= 0:
            raise ValueError("min_verify_sequences must be positive")
        if self.min_verify_sequences > self.max_verify_sequences:
            raise ValueError("min_verify_sequences cannot exceed max_verify_sequences")
        if self.per_parent_child_cap <= 0:
            raise ValueError("per_parent_child_cap must be positive")
        if self.fallback_backend not in {"gpu_marginal", "cpu_ddtree", "none"}:
            raise ValueError("fallback_backend must be gpu_marginal, cpu_ddtree, or none")
        if self.selection_top_multiplier <= 0:
            raise ValueError("selection_top_multiplier must be positive")
        if self.latency_gate_small_tree_nodes <= 0:
            raise ValueError("latency_gate_small_tree_nodes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "JointDDTConfig":
        if data is None:
            config = cls()
        else:
            valid_keys = set(cls.__dataclass_fields__.keys())
            config = cls(**{key: value for key, value in data.items() if key in valid_keys})
        config.validate()
        return config

    @classmethod
    def from_args(cls, args: Any) -> "JointDDTConfig":
        data = {}
        for key in cls.__dataclass_fields__:
            if hasattr(args, key):
                data[key] = getattr(args, key)
        if hasattr(args, "joint_topk"):
            data["joint_topk"] = getattr(args, "joint_topk")
        if hasattr(args, "candidate_pool_nodes"):
            data["candidate_pool_nodes"] = getattr(args, "candidate_pool_nodes")
        if hasattr(args, "candidate_pool_sequences"):
            data["candidate_pool_sequences"] = getattr(args, "candidate_pool_sequences")
        if hasattr(args, "max_verify_nodes"):
            data["max_verify_nodes"] = getattr(args, "max_verify_nodes")
        if hasattr(args, "max_verify_sequences"):
            data["max_verify_sequences"] = getattr(args, "max_verify_sequences")
        if hasattr(args, "fallback_to_ddtree"):
            data["fallback_to_ddtree"] = getattr(args, "fallback_to_ddtree")
        return cls.from_dict(data)
