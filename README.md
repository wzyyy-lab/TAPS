# TAPS

TAPS is a learned proposal selector for DDTree-style speculative decoding. It starts from a large DDTree candidate pool, scores candidate nodes with a lightweight node value model, and sends only a compact subset of high-utility nodes/sequences to the target verifier. The goal is to keep most of DDTree's accepted length while reducing verification work.

This repository includes:

- DDTree and DFlash baselines.
- Trace collection for training the TAPS node value model.
- Node value model training and offline selector evaluation.
- TAPS-Lite: a minimal-overhead variant using a 303-parameter TinyScorer with CPU beam search (~1.5ms scoring overhead per round).
- A shared benchmark entry point for TAPS, TAPS-Lite, DDTree, and DFlash on identical prompts.

## Idea

For each decoding step, DDTree can produce many possible draft-tree nodes. Verifying all of them is expensive. TAPS treats this as a selection problem:

1. Build a DDTree candidate pool from the DFlash draft model.
2. Estimate each node's reach value with a trained node value network.
3. Select a bounded number of nodes and sequences for target-model verification.
4. Accept tokens using the same verifier semantics as DDTree, but with a smaller verification budget.

Two variants are provided:

- **TAPS64**: uses the full node value model to score candidates from a 512-node pool, verifying at most 64 nodes and 64 sequences.
- **TAPS-Lite**: uses a TinyScorer (7-feature → 32-hidden → 1 MLP, 303 parameters) that scores candidates from draft-model statistics only (log-probs, entropy, margin, topk mass). GPU scoring + single bulk transfer to CPU + numpy beam search keeps per-round overhead to ~1.5ms.

## Requirements

The code is intended for a CUDA GPU environment.

Tested setup:

- Linux
- Python 3.11
- CUDA-capable GPU
- PyTorch with CUDA
- `transformers`, `datasets`, `flash-attn`, and the packages in `requirements.txt`

Install dependencies:

```bash
git clone <your-taps-repo-url> TAPS
cd TAPS
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Prepare a target model and a DFlash draft model. The experiments below used Qwen3-4B as the target model and Qwen3-4B-DFlash-b16 as the draft model:

```bash
export TARGET_MODEL=/path/to/Qwen3-4B
export DRAFT_MODEL=/path/to/Qwen3-4B-DFlash-b16
export RUN_DIR=outputs/taps_repro
mkdir -p "$RUN_DIR"
```

## Supported Datasets

The benchmark loader currently supports `aime25`, `gsm8k`, `humaneval`, `livecodebench`, `math500`, `mbpp`, and `mt-bench`. Local dataset files can be placed under the Hugging Face assets layout expected by `model/utils.py`; otherwise the loader falls back to the corresponding Hugging Face dataset names when available.

## Trace Sampling

The TAPS checkpoint used for the reported results was trained from the following trace prompts. `--max-samples` selects dataset rows after shuffling. For single-turn datasets, one row is one prompt. For `mt-bench`, 40 selected rows expand to 80 prompts because each row has two turns.

| Dataset | Selected rows | Expanded prompts | Trace chunks passed to `--sample-offset/--max-samples` |
|---|---:|---:|---|
| `aime25` | 15 | 15 | `0/15` |
| `gsm8k` | 64 | 64 | `0/16`, `16/16`, `32/16`, `48/16` |
| `humaneval` | 64 | 64 | `0/16`, `16/16`, `32/16`, `48/16` |
| `livecodebench` | 64 | 64 | `0/16`, `16/16`, `32/16`, `48/16` |
| `math500` | 64 | 64 | `0/16`, `16/16`, `32/16`, `48/16` |
| `mbpp` | 64 | 64 | `0/16`, `16/16`, `32/16`, `48/16` |
| `mt-bench` | 40 | 80 | `0/16`, `16/16`, `32/8` |

Create the trace job list:

```bash
cat > "$RUN_DIR/trace_jobs.tsv" <<'EOF'
aime25 0 15
gsm8k 0 16
gsm8k 16 16
gsm8k 32 16
gsm8k 48 16
humaneval 0 16
humaneval 16 16
humaneval 32 16
humaneval 48 16
livecodebench 0 16
livecodebench 16 16
livecodebench 32 16
livecodebench 48 16
math500 0 16
math500 16 16
math500 32 16
math500 48 16
mbpp 0 16
mbpp 16 16
mbpp 32 16
mbpp 48 16
mt-bench 0 16
mt-bench 16 16
mt-bench 32 8
EOF
```

Collect traces chunk by chunk:

```bash
while read -r DATASET OFFSET N; do
  OUT="$RUN_DIR/traces/$DATASET/chunk_$OFFSET"
  mkdir -p "$OUT"
  python scripts/60_collect_joint_trace.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --datasets "$DATASET" \
    --tree-budget-baseline 512 \
    --topk-collect 512 \
    --candidate-pool-nodes 512 \
    --candidate-pool-sequences 512 \
    --max-samples "$N" \
    --sample-offset "$OFFSET" \
    --shuffle-seed 2026 \
    --max-new-tokens 512 \
    --output "$OUT"
done < "$RUN_DIR/trace_jobs.tsv"
```

## Train TAPS

Train the node value model from the collected traces:

```bash
python scripts/61_train_node_value_net.py \
  --traces "$RUN_DIR/traces" \
  --output "$RUN_DIR/checkpoints" \
  --target-model "$TARGET_MODEL" \
  --runtime-topk 512 \
  --epochs 30 \
  --batch-size 4096 \
  --grad-accum 1 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --validation-fraction 0.05 \
  --split-strategy stratified \
  --reach-loss-weight 0.5 \
  --rank-loss-weight 0.5 \
  --rank-margin 0.05 \
  --shuffle-seed 2026 \
  --early-stop-patience 0

export CKPT="$RUN_DIR/checkpoints/best.pt"
```

Optional offline selector evaluation:

```bash
python scripts/62_eval_joint_selector.py \
  --traces "$RUN_DIR/traces" \
  --checkpoint "$CKPT" \
  --baseline ddtree_512 \
  --target-accept-drop 0.03 \
  --max-verify-nodes 64 \
  --max-verify-sequences 64 \
  --utility-thresholds 0,0.001,0.003,0.01 \
  --output "$RUN_DIR/selector_eval.json"
```

## Train TAPS-Lite

TAPS-Lite replaces the full node value model with a TinyScorer — a 303-parameter MLP (7→32→1) that scores candidates using only draft-model statistics (log-probs, entropy, margin, topk mass, depth, rank). It uses the same trace data as TAPS.

```bash
python scripts/train_tiny_scorer.py \
  --traces "$RUN_DIR/traces" \
  --output "$RUN_DIR/tiny_scorer" \
  --epochs 30 \
  --lr 3e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 32 \
  --batch-size 512 \
  --val-fraction 0.1 \
  --seed 2026

export TINY_CKPT="$RUN_DIR/tiny_scorer/best.pt"
```

## Run Benchmarks

The examples below run on one held-out `gsm8k` chunk. Use the same `--dataset`, `--sample-offset`, `--max-samples`, and `--shuffle-seed` for all methods when comparing them.

Run TAPS64:

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 16 \
  --sample-offset 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path "$RUN_DIR/taps64_gsm8k_offset64_n16.pt" \
  --proposal-mode joint \
  --tree-budget 512 \
  --joint-checkpoint "$CKPT" \
  --joint-topk 512 \
  --candidate-pool-nodes 512 \
  --candidate-pool-sequences 512 \
  --candidate-pool-source ddtree_heap \
  --min-verify-nodes 16 \
  --max-verify-nodes 64 \
  --min-verify-sequences 4 \
  --max-verify-sequences 64 \
  --no-fallback-to-ddtree \
  --fallback-backend none \
  --utility-threshold 0
```

Run TAPS-Lite:

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 16 \
  --sample-offset 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path "$RUN_DIR/taps_lite_gsm8k_offset64_n16.pt" \
  --proposal-mode joint \
  --tree-budget 512 \
  --tiny-scorer-checkpoint "$TINY_CKPT" \
  --joint-topk 64 \
  --candidate-pool-nodes 768 \
  --candidate-pool-sequences 48 \
  --candidate-pool-source taps_lite \
  --min-verify-nodes 16 \
  --max-verify-nodes 64 \
  --min-verify-sequences 4 \
  --max-verify-sequences 64 \
  --no-fallback-to-ddtree \
  --fallback-backend none
```

Run DDTree512:

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 16 \
  --sample-offset 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path "$RUN_DIR/ddtree512_gsm8k_offset64_n16.pt" \
  --proposal-mode ddtree \
  --tree-budget 512
```

Run DFlash:

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 16 \
  --sample-offset 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path "$RUN_DIR/dflash_gsm8k_offset64_n16.pt" \
  --proposal-mode dflash \
  --flash-attn
```

For the full benchmark, run the same method commands over the held-out chunks used in the results table: `aime25:15/15`, `gsm8k:64/16,80/16,96/16,112/16`, `humaneval:64/16,80/16,96/16,112/16`, `livecodebench:64/16,80/16,96/16,112/16`, `math500:64/16,80/16,96/16,112/16`, `mbpp:64/16,80/16,96/16,112/16`, and `mt-bench:40/16,56/16,72/16`.

## Results

All numbers below use the same target model, draft model, held-out prompts, `shuffle_seed=2026`, and `max_new_tokens=512`. Throughput is generated tokens divided by generation wall time; model loading and job startup time are excluded. Accept length is response-weighted average accepted length.

| Dataset | TAPS64 Tok/s | TAPS64 Accept | DDTree512 Tok/s | DDTree512 Accept | DFlash Tok/s | DFlash Accept |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 195.27 | 8.26 | 86.75 | 9.09 | 121.50 | 6.27 |
| `aime25` | 227.63 | 8.80 | 100.69 | 9.78 | 152.06 | 6.97 |
| `gsm8k` | 216.73 | 8.51 | 96.90 | 9.31 | 139.72 | 6.46 |
| `humaneval` | 231.40 | 8.92 | 101.44 | 9.79 | 146.51 | 6.71 |
| `livecodebench` | 208.68 | 8.82 | 91.23 | 9.72 | 138.96 | 6.92 |
| `math500` | 224.43 | 9.84 | 107.19 | 10.58 | 158.51 | 7.71 |
| `mbpp` | 204.61 | 8.22 | 92.89 | 9.10 | 124.20 | 6.01 |
| `mt-bench` | 130.46 | 5.74 | 58.44 | 6.52 | 76.56 | 4.16 |

TAPS64 verifies 64 nodes on average from a 512-node candidate pool, with an average of 32.97 selected sequences and no DDTree fallback in the reported run.

### Relative comparison

| Dataset | vs DDTree512 Tok/s | vs DDTree512 Accept | vs DFlash Tok/s | vs DFlash Accept |
|---|---:|---:|---:|---:|
| Overall | **+125.1%** | -9.1% | **+60.7%** | **+31.7%** |
| `aime25` | **+126.1%** | -10.0% | **+49.7%** | **+26.3%** |
| `gsm8k` | **+123.7%** | -8.6% | **+55.1%** | **+31.7%** |
| `humaneval` | **+128.1%** | -8.9% | **+57.9%** | **+32.9%** |
| `livecodebench` | **+128.7%** | -9.3% | **+50.2%** | **+27.5%** |
| `math500` | **+109.4%** | -7.0% | **+41.6%** | **+27.6%** |
| `mbpp` | **+120.3%** | -9.7% | **+64.7%** | **+36.8%** |
| `mt-bench` | **+123.2%** | -12.0% | **+70.4%** | **+38.0%** |

TAPS64 achieves 2.25x the throughput of DDTree512 with a 9.1% acceptance length trade-off, and 1.61x the throughput of DFlash while also improving acceptance length by 31.7%.

## Acknowledgements

TAPS builds on the DDTree and DFlash speculative decoding code paths. The baseline commands above are kept in this repository so that TAPS, DDTree, and DFlash can be evaluated under the same prompt selection and timing setup.
