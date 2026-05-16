from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from dflash import cuda_time, empty_stage_times
from ddtree import compact_dynamic_cache
from model import DFlashDraftModel, extract_context_feature, sample

from .config import JointDDTConfig
from .lattice import extract_topk_lattice
from .model import NodeValueNet, load_node_value_net
from .pool import build_ddtree_candidate_trie, build_marginal_candidate_trie, build_union_candidate_trie, value_token_scores_from_edges
from .selector import score_candidate_trie, select_joint_tree, select_marginal_tree
from .tree import compile_joint_tree, follow_tree_tensorized


JOINT_STAGE_ORDER = (
    "draft",
    "lattice",
    "pool",
    "feature_mlp_select",
    "tree_compile",
    "verify",
    "commit",
    "fallback",
)


def _runtime_hidden_provenance(model: DFlashDraftModel) -> dict:
    return {
        "layer_ids": [int(x) for x in model.target_layer_ids],
        "token_position": "runtime_target_hidden_last_available",
        "timing": "before_current_round_draft_and_verification",
        "projection_version": "dflash_fc_input",
    }


def _load_joint_checkpoint(
    checkpoint: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[NodeValueNet, JointDDTConfig, dict]:
    if checkpoint is None:
        raise ValueError("--joint-checkpoint is required when proposal-mode=joint")
    model, payload = load_node_value_net(checkpoint, device=device, dtype=torch.float32)
    config = JointDDTConfig.from_dict(payload.get("joint_config"))
    return model, config, payload


@torch.inference_mode()
def joint_ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    joint_checkpoint: str | None = None,
    joint_model: NodeValueNet | None = None,
    joint_config: JointDDTConfig | None = None,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if temperature >= 1e-5:
        raise ValueError("Joint-DDT v1 supports greedy target verification only: temperature must be 0.0")
    if block_size <= 1:
        from dflash import dflash_generate

        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    if joint_model is None:
        joint_model, checkpoint_config, payload = _load_joint_checkpoint(joint_checkpoint, model.device, target.dtype)
        if joint_config is None:
            joint_config = checkpoint_config
        hidden_provenance = payload.get("hidden_provenance", {})
        runtime_provenance = _runtime_hidden_provenance(model)
        if hidden_provenance and hidden_provenance != runtime_provenance:
            raise ValueError(
                "Joint checkpoint hidden provenance does not match runtime. "
                f"checkpoint={hidden_provenance}, runtime={runtime_provenance}"
            )
        joint_model.validate_vocab_size(getattr(target.config, "vocab_size", None))
        calibration = payload.get("calibration", {})
    else:
        joint_model.validate_vocab_size(getattr(target.config, "vocab_size", None))
        runtime_provenance = _runtime_hidden_provenance(model)
        if joint_config is not None and joint_config.hidden_provenance and joint_config.hidden_provenance != runtime_provenance:
            raise ValueError(
                "Joint config hidden provenance does not match runtime. "
                f"config={joint_config.hidden_provenance}, runtime={runtime_provenance}"
            )
        calibration = {}
    if joint_config is None:
        joint_config = JointDDTConfig()
    joint_config.validate()
    joint_model.eval()

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    max_verify_nodes = max(0, int(joint_config.max_verify_nodes))
    max_tree_nodes = 1 + max_verify_nodes

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(JOINT_STAGE_ORDER)
    fallback_counts: dict[str, int] = {}
    round_joint_metrics = []
    ema_verify_time = None
    ema_joint_overhead = None

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -draft_horizon:, :]
        )
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        lattice_start = cuda_time()
        lattice = extract_topk_lattice(draft_logits[0], joint_config.joint_topk)
        lattice_elapsed = cuda_time() - lattice_start
        stage_times["lattice"] += lattice_elapsed

        context_hidden = None
        if joint_model.context_hidden_dim > 0:
            context_hidden = target_hidden[:, -1, :]
            if context_hidden.shape[-1] != joint_model.context_hidden_dim:
                raise ValueError(
                    "Joint checkpoint context_hidden_dim does not match runtime target_hidden. "
                    f"checkpoint={joint_model.context_hidden_dim}, runtime={context_hidden.shape[-1]}"
                )

        pool_start = cuda_time()
        value_token_scores = None
        seed_score_elapsed = 0.0
        if joint_config.candidate_pool_source == "ddtree_heap":
            candidate_trie = build_ddtree_candidate_trie(draft_logits[0], joint_config.candidate_pool_nodes)
        else:
            if joint_config.enable_value_beam_pool:
                seed_nodes = max(1, int(joint_config.candidate_pool_nodes * min(joint_config.marginal_pool_fraction, 0.25)))
                seed_trie = build_marginal_candidate_trie(lattice, joint_config, max_nodes=seed_nodes)
                seed_score_start = cuda_time()
                seed_scores = score_candidate_trie(
                    seed_trie,
                    lattice,
                    root_token_id=root_token[0, 0],
                    model=joint_model,
                    config=joint_config,
                    prompt_length=start,
                    context_hidden=context_hidden,
                    calibration=calibration,
                )
                value_token_scores = value_token_scores_from_edges(lattice, seed_trie, seed_scores.edge_logits)
                seed_score_elapsed = cuda_time() - seed_score_start
                stage_times["feature_mlp_select"] += seed_score_elapsed
            candidate_trie = build_union_candidate_trie(lattice, joint_config, value_token_scores=value_token_scores)
        pool_elapsed = max(0.0, cuda_time() - pool_start - seed_score_elapsed)
        stage_times["pool"] += pool_elapsed

        select_start = cuda_time()
        selection = select_joint_tree(
            candidate_trie,
            lattice,
            root_token_id=root_token[0, 0],
            model=joint_model,
            config=joint_config,
            prompt_length=start,
            context_hidden=context_hidden,
            calibration=calibration,
        )
        selected_tree = selection.selected_tree
        fallback_reason = selection.fallback_reason
        select_elapsed = cuda_time() - select_start
        stage_times["feature_mlp_select"] += select_elapsed

        current_overhead = lattice_elapsed + pool_elapsed + seed_score_elapsed + select_elapsed
        if ema_joint_overhead is None:
            ema_joint_overhead = current_overhead
        else:
            ema_joint_overhead = 0.9 * ema_joint_overhead + 0.1 * current_overhead
        if (
            ema_verify_time is not None
            and len(acceptance_lengths) >= int(joint_config.latency_gate_warmup_rounds)
            and ema_joint_overhead > float(joint_config.max_pool_build_overhead_ratio) * ema_verify_time
        ):
            fallback_reason = fallback_reason or "latency_gate_small_tree"

        completed_rounds = max(1, len(acceptance_lengths))
        fallback_rate_so_far = sum(fallback_counts.values()) / completed_rounds
        cpu_fallback_rate_allowed = fallback_rate_so_far <= float(joint_config.max_fallback_rate)
        use_ddtree_fallback = bool(
            fallback_reason
            and joint_config.fallback_to_ddtree
            and cpu_fallback_rate_allowed
            and (joint_config.fallback_backend == "cpu_ddtree" or joint_config.debug_force_cpu_heap)
        )
        use_gpu_marginal_fallback = bool(
            fallback_reason
            and joint_config.fallback_to_ddtree
            and not use_ddtree_fallback
            and (
                joint_config.fallback_backend == "gpu_marginal"
                or (joint_config.fallback_backend == "cpu_ddtree" and not cpu_fallback_rate_allowed)
            )
        )
        if use_ddtree_fallback:
            from ddtree import build_ddtree_tree, compile_ddtree_tree

            fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
            fallback_start = cuda_time()
            node_token_ids, node_depths, parents, child_maps, visibility_cpu, _ = build_ddtree_tree(
                draft_logits[0],
                max_verify_nodes,
            )
            stage_times["fallback"] += cuda_time() - fallback_start

            tree_compile_start = cuda_time()
            (
                verify_input_ids,
                verify_position_ids,
                verify_attention_mask,
                previous_tree_start,
                previous_tree_length,
            ) = compile_ddtree_tree(
                root_token_id=root_token[0, 0],
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility_cpu=visibility_cpu,
                past_length=start,
                dtype=target.dtype,
                device=model.device,
                verify_input_ids_buffer=verify_input_ids_buffer,
                verify_position_ids_buffer=verify_position_ids_buffer,
                attention_mask_buffer=attention_mask_buffer,
                tree_visibility_buffer=tree_visibility_buffer,
                previous_tree_start=previous_tree_start,
                previous_tree_length=previous_tree_length,
            )
            stage_times["tree_compile"] += cuda_time() - tree_compile_start
        else:
            if use_gpu_marginal_fallback:
                fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
                fallback_config = JointDDTConfig.from_dict(joint_config.to_dict())
                fallback_config.max_verify_nodes = min(
                    int(joint_config.max_verify_nodes),
                    int(joint_config.latency_gate_small_tree_nodes),
                )
                fallback_config.min_verify_nodes = min(int(fallback_config.min_verify_nodes), int(fallback_config.max_verify_nodes))
                selected_tree = select_marginal_tree(candidate_trie, fallback_config)
            if selected_tree.current_length > max_tree_nodes:
                selected_tree = select_marginal_tree(candidate_trie, joint_config)
            tree_compile_start = cuda_time()
            (
                verify_input_ids,
                verify_position_ids,
                verify_attention_mask,
                previous_tree_start,
                previous_tree_length,
            ) = compile_joint_tree(
                root_token_id=root_token[0, 0],
                start=start,
                selected_tree=selected_tree,
                past_length=start,
                dtype=target.dtype,
                verify_input_ids_buffer=verify_input_ids_buffer,
                verify_position_ids_buffer=verify_position_ids_buffer,
                attention_mask_buffer=attention_mask_buffer,
                tree_visibility_buffer=tree_visibility_buffer,
                previous_tree_start=previous_tree_start,
                previous_tree_length=previous_tree_length,
            )
            stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        verify_elapsed = cuda_time() - verify_stage_start
        stage_times["verify"] += verify_elapsed
        if ema_verify_time is None:
            ema_verify_time = verify_elapsed
        else:
            ema_verify_time = 0.9 * ema_verify_time + 0.1 * verify_elapsed

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        if use_ddtree_fallback:
            from ddtree import follow_verified_tree

            accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        else:
            accepted_indices, next_token = follow_tree_tensorized(selected_tree, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        round_joint_metrics.append({
            **selection.metrics,
            "used_ddtree_fallback": use_ddtree_fallback,
            "used_gpu_marginal_fallback": use_gpu_marginal_fallback,
            "verify_nodes": int(verify_input_ids.shape[1] - 1),
            "batch_max_verify_nodes": int(verify_input_ids.shape[1] - 1),
            "padded_verify_nodes": int(max_tree_nodes - 1),
            "effective_padded_node_count": int(verify_input_ids.shape[1] - 1),
            "attention_mask_density": float((verify_attention_mask == 0).float().mean().item()),
            "joint_overhead_time": float(current_overhead),
            "seed_score_time": float(seed_score_elapsed),
            "ema_joint_overhead": float(ema_joint_overhead or 0.0),
            "ema_verify_time": float(ema_verify_time or 0.0),
            "fallback_rate_so_far": float(fallback_rate_so_far),
        })

        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "joint_metrics": round_joint_metrics[-1],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in verify_input_ids[0, 1:].detach().cpu().tolist()],
                    "parents": (
                        [int(parent) for parent in parents]
                        if use_ddtree_fallback
                        else [int(parent) for parent in selected_tree.parents.detach().cpu().tolist()]
                    ),
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
        joint_metrics=round_joint_metrics,
        fallback_counts=fallback_counts,
    )
