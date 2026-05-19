from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint.tiny_scorer import TINY_SCORER_FEATURES, TinyScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_trace_records(path: str) -> list[dict]:
    trace_path = Path(path)
    records: list[dict] = []
    if trace_path.is_dir():
        for fp in sorted(trace_path.rglob("*.pt")):
            loaded = torch.load(fp, map_location="cpu")
            if isinstance(loaded, list):
                records.extend(loaded)
            else:
                records.append(loaded)
    else:
        loaded = torch.load(trace_path, map_location="cpu")
        if isinstance(loaded, list):
            records.extend(loaded)
        else:
            records.append(loaded)
    return records


def extract_training_sample(record: dict) -> dict | None:
    top_log_probs = record["top_log_probs"].float()
    H, K = top_log_probs.shape
    if H == 0 or K == 0:
        return None

    position_entropy = record["position_entropy"].float()
    top1_top2_margin = record["top1_top2_margin"].float()
    topk_mass = record["topk_mass"].float()

    trie = record["candidate_trie"]
    depths = torch.tensor(trie["depths"], dtype=torch.long)
    ranks = torch.tensor(trie["ranks"], dtype=torch.long)
    target_probs = record["target_child_probs"].float()

    if depths.numel() == 0:
        return None

    step_lp = top_log_probs
    entropy = position_entropy.unsqueeze(1).expand(H, K)
    margin = top1_top2_margin.unsqueeze(1).expand(H, K)
    mass = topk_mass.unsqueeze(1).expand(H, K)
    depth_norm = (torch.arange(1, H + 1, dtype=torch.float32) / max(H, 1)).unsqueeze(1).expand(H, K)
    rank_norm = (torch.arange(K, dtype=torch.float32) / max(K - 1, 1)).unsqueeze(0).expand(H, K)
    step_gap = step_lp - step_lp[:, 0:1]
    features_grid = torch.stack([step_lp, entropy, margin, mass, depth_norm, rank_norm, step_gap], dim=-1)

    target_grid = torch.zeros(H, K, dtype=torch.float32)
    count_grid = torch.zeros(H, K, dtype=torch.float32)
    mask_grid = torch.zeros(H, K, dtype=torch.bool)

    d_idx = (depths - 1).clamp(0, H - 1)
    r_idx = ranks.clamp(0, K - 1)
    target_grid.index_put_((d_idx, r_idx), target_probs, accumulate=True)
    count_grid.index_put_((d_idx, r_idx), torch.ones_like(target_probs), accumulate=True)
    has_data = count_grid > 0
    target_grid[has_data] /= count_grid[has_data]
    mask_grid[has_data] = True

    observed_features = features_grid[mask_grid]
    observed_targets = target_grid[mask_grid].clamp(1e-6, 1.0 - 1e-6)

    return {
        "features": observed_features,
        "targets": observed_targets,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading traces from {args.traces} ...")
    records = load_trace_records(args.traces)
    print(f"Loaded {len(records)} trace rounds")

    if args.max_records and len(records) > args.max_records:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[:args.max_records]

    samples = []
    for r in records:
        s = extract_training_sample(r)
        if s is not None and s["features"].numel() > 0:
            samples.append(s)
    print(f"Extracted {len(samples)} valid training samples")

    all_features = torch.cat([s["features"] for s in samples], dim=0)
    all_targets = torch.cat([s["targets"] for s in samples], dim=0)
    print(f"Total training points: {all_features.shape[0]}")
    print(f"Target stats: mean={all_targets.mean():.4f}, median={all_targets.median():.4f}, "
          f"p10={all_targets.quantile(0.1):.4f}, p90={all_targets.quantile(0.9):.4f}")

    n = all_features.shape[0]
    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_n = max(1, int(n * args.val_fraction))
    val_idx = torch.tensor(indices[:val_n], dtype=torch.long)
    train_idx = torch.tensor(indices[val_n:], dtype=torch.long)

    train_features = all_features[train_idx].to(device)
    train_targets = all_targets[train_idx].to(device)
    val_features = all_features[val_idx].to(device)
    val_targets = all_targets[val_idx].to(device)
    print(f"Train: {train_features.shape[0]}, Val: {val_features.shape[0]}")

    model = TinyScorer(hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"TinyScorer params: {param_count}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train_features.shape[0], device=device)
        train_f = train_features[perm]
        train_t = train_targets[perm]

        running_loss = 0.0
        steps = 0
        for i in range(0, train_f.shape[0], args.batch_size):
            batch_f = train_f[i:i + args.batch_size]
            batch_t = train_t[i:i + args.batch_size]

            logits = model(batch_f)
            loss = F.binary_cross_entropy_with_logits(logits, batch_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(steps, 1)

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_features)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_targets).item()
            val_probs = torch.sigmoid(val_logits)
            mae = (val_probs - val_targets).abs().mean().item()
            corr_num = ((val_probs - val_probs.mean()) * (val_targets - val_targets.mean())).sum()
            corr_den = (val_probs.std() * val_targets.std() * val_probs.numel())
            correlation = (corr_num / corr_den.clamp_min(1e-8)).item() if corr_den > 0 else 0.0

        print(f"epoch {epoch + 1}/{args.epochs}: train_bce={train_loss:.6f} val_bce={val_loss:.6f} "
              f"val_mae={mae:.4f} val_corr={correlation:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                output_dir / "best.pt",
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=mae,
                val_correlation=correlation,
                epoch=epoch + 1,
                feature_names=list(TINY_SCORER_FEATURES),
            )
            print(f"  -> saved best checkpoint (val_bce={val_loss:.6f})")

    model.save_checkpoint(
        output_dir / "last.pt",
        train_loss=train_loss,
        val_loss=val_loss,
        epoch=args.epochs,
        feature_names=list(TINY_SCORER_FEATURES),
    )
    print(f"\nDone. Best val BCE: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
