from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint.config import JointDDTConfig
from joint.lattice import TopKLattice
from joint.model import EdgeFeatureBatch, NodeValueNet
from joint.pool import candidate_trie_from_dict
from joint.segments import (
    grouped_cross_entropy,
    grouped_kl_divergence,
    grouped_softmax,
    propagate_reach_from_edges,
    sibling_rank_loss,
)
from joint.selector import SCALAR_FEATURE_NAMES, build_edge_feature_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--runtime-topk", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256, help="Approximate parent groups per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--token-embed-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--min-vocab-size", type=int, default=200000)
    parser.add_argument("--use-target-hidden", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Set to 0 to disable early stopping")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--split-strategy", choices=("stratified", "contiguous"), default="stratified")
    parser.add_argument("--reach-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    return parser.parse_args()


def make_lattice(record: dict, device: torch.device) -> TopKLattice:
    top_token_ids = record["top_token_ids"].to(device=device, dtype=torch.long)
    top_log_probs = record["top_log_probs"].to(device=device, dtype=torch.float32)
    k = min(top_token_ids.shape[1], top_log_probs.shape[1])
    return TopKLattice(
        top_token_ids=top_token_ids[:, :k],
        top_log_probs=top_log_probs[:, :k],
        position_entropy=record["position_entropy"].to(device=device, dtype=torch.float32),
        top1_top2_margin=record["top1_top2_margin"].to(device=device, dtype=torch.float32),
        topk_mass=record["topk_mass"].to(device=device, dtype=torch.float32),
        log_z=torch.empty(top_token_ids.shape[0], dtype=torch.float32, device=device),
    )


def _as_record_list(loaded: object) -> list[dict]:
    return loaded if isinstance(loaded, list) else [loaded]


def _trace_dataset_for_file(root: Path, file_path: Path) -> str:
    if root.is_dir():
        relative = file_path.relative_to(root)
        return str(relative.parts[0]) if len(relative.parts) > 1 else "__flat__"
    return "__single_file__"


def load_trace_records_for_training(path: str | Path) -> list[dict]:
    trace_path = Path(path)
    if trace_path.is_dir():
        records: list[dict] = []
        for file_path in sorted(trace_path.rglob("*.pt")):
            dataset = _trace_dataset_for_file(trace_path, file_path)
            loaded = torch.load(file_path, map_location="cpu")
            for record_index, record in enumerate(_as_record_list(loaded)):
                enriched = dict(record)
                enriched["_trace_dataset"] = dataset
                enriched["_trace_file"] = str(file_path.relative_to(trace_path))
                enriched["_trace_record_index"] = int(record_index)
                records.append(enriched)
        return records

    loaded = torch.load(trace_path, map_location="cpu")
    records = []
    for record_index, record in enumerate(_as_record_list(loaded)):
        enriched = dict(record)
        enriched["_trace_dataset"] = "__single_file__"
        enriched["_trace_file"] = trace_path.name
        enriched["_trace_record_index"] = int(record_index)
        records.append(enriched)
    return records


def record_dataset(record: dict) -> str:
    return str(record.get("_trace_dataset", "__unknown__"))


def record_split_group(record: dict) -> str:
    dataset = record_dataset(record)
    for key in ("prompt_id", "sample_id", "instance_id", "question_id", "task_id"):
        if key in record:
            return f"{dataset}:{key}:{record[key]}"
    trace_file = record.get("_trace_file")
    if trace_file is not None:
        return f"{dataset}:file:{trace_file}"
    record_index = record.get("_trace_record_index", id(record))
    return f"{dataset}:record:{record_index}"


def split_train_val_records(
    records: list[dict],
    validation_fraction: float,
    shuffle_seed: int,
    strategy: str,
) -> tuple[list[dict], list[dict]]:
    if not records:
        raise ValueError("No trace records found")
    if strategy == "contiguous":
        split = max(1, int(len(records) * (1.0 - validation_fraction)))
        train_records = list(records[:split])
        val_records = list(records[split:]) if split < len(records) else list(records[: min(len(records), 32)])
        return train_records, val_records

    if strategy != "stratified":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[record_dataset(record)][record_split_group(record)].append(record)

    rng = random.Random(shuffle_seed)
    train_records: list[dict] = []
    val_records: list[dict] = []
    for dataset in sorted(groups):
        split_groups = list(groups[dataset].values())
        rng.shuffle(split_groups)
        total_records = sum(len(group) for group in split_groups)
        if validation_fraction <= 0.0 or len(split_groups) <= 1:
            train_records.extend(record for group in split_groups for record in group)
            continue
        target_val = max(1, int(total_records * validation_fraction))
        dataset_val: list[dict] = []
        dataset_train: list[dict] = []
        for group_index, group in enumerate(split_groups):
            has_train_group_left = group_index < len(split_groups) - 1
            needs_val = len(dataset_val) < target_val
            if needs_val and has_train_group_left:
                dataset_val.extend(group)
            else:
                dataset_train.extend(group)
        if not dataset_val:
            dataset_val.extend(split_groups[0])
            dataset_train = [record for group in split_groups[1:] for record in group]
        train_records.extend(dataset_train)
        val_records.extend(dataset_val)

    if not val_records:
        shuffled = list(records)
        rng.shuffle(shuffled)
        fallback_count = min(len(shuffled), max(1, int(len(shuffled) * max(validation_fraction, 0.05))))
        val_records = shuffled[:fallback_count]
        train_records = shuffled[fallback_count:] or shuffled[:1]

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def describe_record_groups(records: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(record_dataset(record) for record in records).items()))


def reach_bce_loss(reach_probs: torch.Tensor, reach_labels: torch.Tensor) -> torch.Tensor:
    probs = reach_probs.float().clamp(1e-6, 1.0 - 1e-6)
    labels = reach_labels.float()
    positive = labels * torch.log(probs)
    negative = (1.0 - labels) * torch.log1p(-probs)
    return -(positive + negative).mean()


def compute_training_losses(
    model: NodeValueNet,
    batch: EdgeFeatureBatch,
    parent_ids: torch.Tensor,
    target_edges: torch.Tensor,
    edge_probs: torch.Tensor,
    other_probs: torch.Tensor,
    reach_labels: torch.Tensor,
    child_node_ids: torch.Tensor,
    root_node_ids: torch.Tensor,
    reach_loss_weight: float,
    rank_loss_weight: float,
    rank_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    edge_logits, other_logits = model(batch)
    loss_ce = grouped_cross_entropy(edge_logits, parent_ids, other_logits, target_edges)
    loss_kl = grouped_kl_divergence(edge_logits, parent_ids, other_logits, edge_probs, other_probs)
    q_cond, _ = grouped_softmax(edge_logits, parent_ids, other_logits)
    q_reach = propagate_reach_from_edges(
        q_cond,
        parent_ids,
        child_node_ids,
        root_node_ids,
        other_logits.numel(),
        depths=batch.depths,
    )
    edge_reach = q_reach[child_node_ids]
    loss_reach = reach_bce_loss(edge_reach, reach_labels)
    loss_rank = sibling_rank_loss(q_cond.float(), parent_ids, reach_labels > 0.5, margin=rank_margin)
    loss = loss_ce + loss_kl + float(reach_loss_weight) * loss_reach + float(rank_loss_weight) * loss_rank
    return loss, {
        "edge_logits": edge_logits,
        "other_logits": other_logits,
        "q_cond": q_cond,
        "edge_reach": edge_reach,
        "loss_ce": loss_ce,
        "loss_kl": loss_kl,
        "loss_reach": loss_reach,
        "loss_rank": loss_rank,
    }


def fit_reach_temperature_grid(
    calibration_batches: list[dict[str, torch.Tensor]],
    temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
) -> tuple[float, torch.Tensor, torch.Tensor]:
    if not calibration_batches:
        return 1.0, torch.empty(0), torch.empty(0)

    best_temp = 1.0
    best_loss = float("inf")
    best_probs = torch.empty(0)
    best_labels = torch.empty(0)
    for temp in temperatures:
        probs_all = []
        labels_all = []
        total_loss = 0.0
        total_edges = 0
        for item in calibration_batches:
            edge_logits = item["edge_logits"] / max(float(temp), 1e-4)
            other_logits = item["other_logits"] / max(float(temp), 1e-4)
            parent_ids = item["parent_ids"]
            child_node_ids = item["child_node_ids"]
            root_node_ids = item["root_node_ids"]
            depths = item["depths"]
            labels = item["reach_labels"]
            q_cond, _ = grouped_softmax(edge_logits, parent_ids, other_logits)
            q_reach = propagate_reach_from_edges(
                q_cond,
                parent_ids,
                child_node_ids,
                root_node_ids,
                other_logits.numel(),
                depths=depths,
            )
            probs = q_reach[child_node_ids].float().clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(probs, labels.float(), reduction="sum")
            total_loss += float(loss.item())
            total_edges += int(labels.numel())
            probs_all.append(probs.detach().cpu())
            labels_all.append(labels.detach().cpu())
        mean_loss = total_loss / max(total_edges, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_temp = float(temp)
            best_probs = torch.cat(probs_all) if probs_all else torch.empty(0)
            best_labels = torch.cat(labels_all) if labels_all else torch.empty(0)
    return best_temp, best_probs, best_labels


def concat_batches(items: list[tuple[EdgeFeatureBatch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    edge_batches, parent_ids_list, target_edges_list, edge_probs_list, other_probs_list, reach_labels_list = zip(*items)
    parent_offsets = []
    edge_offsets = []
    parent_total = 0
    edge_total = 0
    for batch in edge_batches:
        parent_offsets.append(parent_total)
        edge_offsets.append(edge_total)
        parent_total += int(batch.parent_token_ids_for_other.numel())
        edge_total += int(batch.child_token_ids.numel())

    def cat_attr(name: str) -> torch.Tensor:
        return torch.cat([getattr(batch, name) for batch in edge_batches], dim=0)

    edge_context_parts = []
    parent_context_parts = []
    for batch in edge_batches:
        if batch.context_hidden is None:
            continue
        context = batch.context_hidden
        parent_context = batch.parent_context_hidden if batch.parent_context_hidden is not None else context
        if context.shape[0] == 1:
            context = context.expand(batch.child_token_ids.numel(), -1)
        if parent_context.shape[0] == 1:
            parent_context = parent_context.expand(batch.parent_token_ids_for_other.numel(), -1)
        edge_context_parts.append(context)
        parent_context_parts.append(parent_context)
    context_hidden = None if not edge_context_parts else torch.cat(edge_context_parts, dim=0)
    parent_context_hidden = None if not parent_context_parts else torch.cat(parent_context_parts, dim=0)
    merged = EdgeFeatureBatch(
        child_token_ids=cat_attr("child_token_ids"),
        parent_token_ids=cat_attr("parent_token_ids"),
        root_token_ids=cat_attr("root_token_ids"),
        depths=cat_attr("depths"),
        ranks=cat_attr("ranks"),
        source_ids=cat_attr("source_ids"),
        scalar_features=cat_attr("scalar_features"),
        parent_token_ids_for_other=cat_attr("parent_token_ids_for_other"),
        root_token_ids_for_other=cat_attr("root_token_ids_for_other"),
        parent_depths=cat_attr("parent_depths"),
        parent_scalar_features=cat_attr("parent_scalar_features"),
        context_hidden=context_hidden,
        parent_context_hidden=parent_context_hidden,
    )

    device = edge_batches[0].child_token_ids.device
    parent_ids = torch.cat([ids + offset for ids, offset in zip(parent_ids_list, parent_offsets)], dim=0)
    child_node_ids = torch.cat(
        [
            torch.arange(1, int(batch.child_token_ids.numel()) + 1, dtype=torch.long, device=device) + parent_offset
            for batch, parent_offset in zip(edge_batches, parent_offsets)
        ],
        dim=0,
    )
    root_node_ids = torch.tensor(parent_offsets, dtype=torch.long, device=device)
    target_edges = []
    for target_edge, edge_offset in zip(target_edges_list, edge_offsets):
        adjusted = target_edge.clone()
        mask = adjusted >= 0
        adjusted[mask] += edge_offset
        target_edges.append(adjusted)
    return (
        merged,
        parent_ids,
        torch.cat(target_edges, dim=0),
        torch.cat(edge_probs_list, dim=0),
        torch.cat(other_probs_list, dim=0),
        torch.cat(reach_labels_list, dim=0),
        child_node_ids,
        root_node_ids,
    )


def record_to_training_item(record: dict, model: NodeValueNet, device: torch.device, use_target_hidden: bool):
    trie = candidate_trie_from_dict(record["candidate_trie"], device=device)
    lattice = make_lattice(record, device)
    root_token = torch.tensor(int(record["root_token_id"]), dtype=torch.long, device=device)
    context_hidden = None
    if use_target_hidden and "target_hidden_proj" in record and model.context_hidden_dim > 0:
        hidden = record["target_hidden_proj"].to(device=device, dtype=torch.float32)
        context_hidden = hidden.reshape(-1, hidden.shape[-1])[-1:].contiguous()
    batch = build_edge_feature_batch(trie, lattice, root_token, model, context_hidden=context_hidden)
    parent_ids = trie.edge_parent_ids
    target_edges = record["target_edge_indices"].to(device=device, dtype=torch.long)
    edge_probs = record["target_child_probs"].to(device=device, dtype=torch.float32)
    other_probs = record["target_other_probs"].to(device=device, dtype=torch.float32)
    reach_labels = torch.zeros((trie.num_nodes,), dtype=torch.float32, device=device)
    for accepted in record.get("accepted_indices", []):
        if int(accepted) > 0 and int(accepted) <= trie.num_nodes:
            reach_labels[int(accepted) - 1] = 1.0
    return batch, parent_ids, target_edges, edge_probs, other_probs, reach_labels


def infer_vocab_size(records: list[dict], minimum: int) -> int:
    max_token = 0
    for record in records:
        max_token = max(max_token, int(record["top_token_ids"].max().item()))
        trie_tokens = record.get("candidate_trie", {}).get("token_ids", [])
        if trie_tokens:
            max_token = max(max_token, max(int(x) for x in trie_tokens))
        max_token = max(max_token, int(record.get("root_token_id", 0)))
    return max(minimum, max_token + 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    records = load_trace_records_for_training(args.traces)
    if args.max_records is not None:
        rng = random.Random(args.shuffle_seed)
        records = list(records)
        rng.shuffle(records)
        records = records[: args.max_records]
    if not records:
        raise ValueError("No trace records found")

    train_records, val_records = split_train_val_records(
        records,
        validation_fraction=args.validation_fraction,
        shuffle_seed=args.shuffle_seed,
        strategy=args.split_strategy,
    )
    print(f"loaded_records={len(records)} split_strategy={args.split_strategy} shuffle_seed={args.shuffle_seed}")
    print(f"train_records={len(train_records)} train_by_dataset={describe_record_groups(train_records)}")
    print(f"val_records={len(val_records)} val_by_dataset={describe_record_groups(val_records)}")

    context_hidden_dim = 0
    if args.use_target_hidden and "target_hidden_proj" in records[0]:
        context_hidden_dim = int(records[0]["target_hidden_proj"].reshape(-1, records[0]["target_hidden_proj"].shape[-1]).shape[-1])
    if args.vocab_size <= 0 and args.target_model is None:
        raise ValueError(
            "Training a runtime checkpoint requires an exact tokenizer/model vocab. "
            "Pass --target-model or --vocab-size; do not rely on trace-only inference."
        )
    if args.vocab_size > 0:
        vocab_size = args.vocab_size
    elif args.target_model is not None:
        vocab_size = int(AutoConfig.from_pretrained(args.target_model).vocab_size)
    else:
        vocab_size = infer_vocab_size(records, args.min_vocab_size)
    model = NodeValueNet(
        vocab_size=vocab_size,
        scalar_feature_dim=len(SCALAR_FEATURE_NAMES),
        context_hidden_dim=context_hidden_dim,
        token_embed_dim=args.token_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if args.init_checkpoint is not None:
        init_model, init_payload = NodeValueNet.from_checkpoint(args.init_checkpoint, map_location=device)
        if init_model.model_config != model.model_config:
            raise ValueError(
                "--init-checkpoint model_config does not match requested training config: "
                f"checkpoint={init_model.model_config}, requested={model.model_config}"
            )
        model.load_state_dict(init_payload["state_dict"])
        print(f"Loaded init checkpoint: {args.init_checkpoint}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss = float("inf")
    stale_epochs = 0
    joint_config = JointDDTConfig(joint_topk=args.runtime_topk)
    hidden_provenance = records[0].get("hidden_provenance", {})

    def evaluate(records_eval: list[dict]) -> tuple[float, dict]:
        model.eval()
        losses = []
        calibration_batches: list[dict[str, torch.Tensor]] = []

        def flush_eval_batch(pending_eval: list) -> None:
            batch, parent_ids, target_edges, edge_probs, other_probs, reach_labels, child_node_ids, root_node_ids = concat_batches(pending_eval)
            loss, loss_parts = compute_training_losses(
                model,
                batch,
                parent_ids,
                target_edges,
                edge_probs,
                other_probs,
                reach_labels,
                child_node_ids,
                root_node_ids,
                args.reach_loss_weight,
                args.rank_loss_weight,
                args.rank_margin,
            )
            losses.append(float(loss.item()))
            calibration_batches.append(
                {
                    "edge_logits": loss_parts["edge_logits"].detach().cpu(),
                    "other_logits": loss_parts["other_logits"].detach().cpu(),
                    "parent_ids": parent_ids.detach().cpu(),
                    "child_node_ids": child_node_ids.detach().cpu(),
                    "root_node_ids": root_node_ids.detach().cpu(),
                    "depths": batch.depths.detach().cpu(),
                    "reach_labels": reach_labels.detach().cpu(),
                }
            )

        with torch.inference_mode():
            pending_eval = []
            pending_parents_eval = 0
            for record_eval in records_eval:
                item_eval = record_to_training_item(record_eval, model, device, args.use_target_hidden)
                pending_eval.append(item_eval)
                pending_parents_eval += int(item_eval[0].parent_token_ids_for_other.numel())
                if pending_parents_eval < args.batch_size:
                    continue
                flush_eval_batch(pending_eval)
                pending_eval = []
                pending_parents_eval = 0
            if pending_eval:
                flush_eval_batch(pending_eval)

        if calibration_batches:
            from joint.calibration import expected_calibration_error

            temp, probs, labels = fit_reach_temperature_grid(calibration_batches)
            stats = expected_calibration_error(probs, labels).to_dict()
            calibration = {
                "global_temperature": temp,
                "calibration_target": "q_reach",
                **stats,
                "valid": True,
            }
        else:
            calibration = {"global_temperature": 1.0, "calibration_target": "q_reach", "confidence": 0.0, "valid": False}
        return sum(losses) / max(len(losses), 1), calibration

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        accum_steps = 0
        pending = []
        pending_parents = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_records = list(train_records)
        random.Random(args.shuffle_seed + epoch + 1).shuffle(epoch_records)
        progress = tqdm(epoch_records, desc=f"epoch {epoch + 1}/{args.epochs}")
        for record in progress:
            item = record_to_training_item(record, model, device, args.use_target_hidden)
            pending.append(item)
            pending_parents += int(item[0].parent_token_ids_for_other.numel())
            if pending_parents < args.batch_size:
                continue

            batch, parent_ids, target_edges, edge_probs, other_probs, reach_labels, child_node_ids, root_node_ids = concat_batches(pending)
            pending = []
            pending_parents = 0
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
                loss, _ = compute_training_losses(
                    model,
                    batch,
                    parent_ids,
                    target_edges,
                    edge_probs,
                    other_probs,
                    reach_labels,
                    child_node_ids,
                    root_node_ids,
                    args.reach_loss_weight,
                    args.rank_loss_weight,
                    args.rank_margin,
                )
                loss = loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()
            accum_steps += 1
            if accum_steps >= max(1, args.grad_accum):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
            running_loss += float(loss.detach().item()) * max(1, args.grad_accum)
            steps += 1
            progress.set_postfix(loss=running_loss / max(steps, 1))

        if pending:
            batch, parent_ids, target_edges, edge_probs, other_probs, reach_labels, child_node_ids, root_node_ids = concat_batches(pending)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
                loss, _ = compute_training_losses(
                    model,
                    batch,
                    parent_ids,
                    target_edges,
                    edge_probs,
                    other_probs,
                    reach_labels,
                    child_node_ids,
                    root_node_ids,
                    args.reach_loss_weight,
                    args.rank_loss_weight,
                    args.rank_margin,
                )
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            accum_steps += 1
            running_loss += float(loss.detach().item()) * max(1, args.grad_accum)
            steps += 1

        if accum_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss, calibration = evaluate(val_records)
        print(f"epoch {epoch + 1}: train_loss={running_loss / max(steps, 1):.6f} val_loss={epoch_loss:.6f} best_val={best_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            stale_epochs = 0
            model.eval()
            model.save_checkpoint(
                Path(args.output) / "best.pt",
                joint_config=joint_config,
                calibration=calibration,
                hidden_provenance=hidden_provenance,
                feature_schema={"scalar_features": list(SCALAR_FEATURE_NAMES)},
                tokenizer_vocab_size=vocab_size,
            )
        else:
            stale_epochs += 1
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            break

    model.eval()
    model.save_checkpoint(
        Path(args.output) / "last.pt",
        joint_config=joint_config,
        calibration=calibration if "calibration" in locals() else {"global_temperature": 1.0, "valid": False},
        hidden_provenance=hidden_provenance,
        feature_schema={"scalar_features": list(SCALAR_FEATURE_NAMES)},
        tokenizer_vocab_size=vocab_size,
    )


if __name__ == "__main__":
    main()
