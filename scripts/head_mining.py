#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

# Keep CUDA GEMM deterministic defaults unless caller overrides.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "EleutherAI/pythia-1.4b"
DEFAULT_SEED = 42
SUMMARY_DECREASE_RATIO_THRESHOLD = 0.8
SUMMARY_DELTA_MEAN_THRESHOLD = -0.01
SUMMARY_DECREASE_RATIO_FALLBACKS = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_reproducibility(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass

    if device.type == "cuda":
        # Required by CUDA for deterministic GEMM kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def load_model(model_name: str, device_name: str):
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def forward_last_token(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if input_ids.is_cuda else nullcontext()
    with torch.inference_mode(), amp_ctx:
        logits = model(input_ids).logits
    last_logits = logits[0, -1]
    last_probs = torch.softmax(last_logits, dim=-1)
    return last_logits, last_probs


def _parse_head_label(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"L(\d+)\.H(\d+)", label.strip())
    if not m:
        raise ValueError(f"Invalid head label: {label}. Expected Lx.Hy")
    return int(m.group(1)), int(m.group(2))


def _head_label(layer: int, head: int) -> str:
    return f"L{layer}.H{head}"


def _replace_last_token_heads_hook(
    head_indices: list[int], n_heads: int, donor_head_vec_by_head: dict[int, torch.Tensor]
):
    def hook(module, inputs):
        hidden = inputs[0]
        _, seq_len, hidden_dim = hidden.shape
        head_dim = hidden_dim // n_heads
        patched = hidden.clone()
        for head_idx in head_indices:
            donor = donor_head_vec_by_head.get(head_idx)
            if donor is None:
                continue
            start = head_idx * head_dim
            end = start + head_dim
            patched[:, seq_len - 1, start:end] = donor.to(hidden.device)
        return (patched,)

    return hook


def _forward_with_hidden_cache(
    model, input_ids: torch.Tensor, n_layers: int
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    cached: dict[int, torch.Tensor] = {}
    handles = []

    def build_hook(layer_idx: int):
        def hook(module, inputs):
            cached[layer_idx] = inputs[0].detach().clone()
            return inputs

        return hook

    for layer_idx in range(n_layers):
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(build_hook(layer_idx))
        handles.append(h)

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if input_ids.is_cuda else nullcontext()
    with torch.inference_mode(), amp_ctx:
        logits = model(input_ids).logits
    for h in handles:
        h.remove()

    last_logits = logits[0, -1]
    last_probs = torch.softmax(last_logits, dim=-1)
    hidden_by_layer = {layer_idx: hidden[0, -1].cpu() for layer_idx, hidden in cached.items()}
    return last_logits, last_probs, hidden_by_layer


def _topk_ids(probs: torch.Tensor, k: int) -> list[int]:
    return torch.topk(probs, k=min(k, probs.shape[-1])).indices.tolist()


def _token_rank(probs: torch.Tensor, token_id: int) -> int:
    # Rank = 1 + number of tokens with strictly higher probability.
    return int((probs > probs[token_id]).sum().item()) + 1


def _snapshot(tokenizer, logits: torch.Tensor, probs: torch.Tensor, top_k: int) -> dict:
    ids = _topk_ids(probs, top_k)
    return {
        "top1_id": ids[0],
        "top1_token": tokenizer.decode([ids[0]]),
        "top1_logit": float(logits[ids[0]].item()),
        "top1_prob": float(probs[ids[0]].item()),
        "topk": [
            {
                "rank": rank + 1,
                "token_id": idx,
                "token": tokenizer.decode([idx]),
                "logit": float(logits[idx].item()),
                "prob": float(probs[idx].item()),
            }
            for rank, idx in enumerate(ids)
        ],
    }


def _snapshot_top1(tokenizer, probs: torch.Tensor) -> dict:
    top1_id = int(torch.argmax(probs).item())
    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id]),
        "top1_prob": float(probs[top1_id].item()),
    }


def _load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Only .jsonl is supported: {path}")
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get("prompt"), str) and row["prompt"].strip():
            prompts.append(row["prompt"].strip())
        elif isinstance(row, str) and row.strip():
            prompts.append(row.strip())
    return prompts


def _load_prompt_items(dataset_root: Path, prompts_file: str) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    used_files: list[str] = []

    if prompts_file.strip():
        prompt_paths = [Path(x.strip()) for x in prompts_file.split(",") if x.strip()]
        if not prompt_paths:
            raise ValueError("--prompts-file was set but no valid file path was provided.")
        for fp in prompt_paths:
            for prompt in _load_prompts(fp):
                items.append({"prompt": prompt, "source_file": str(fp), "category": fp.parent.name or "custom"})
            used_files.append(str(fp))
        return items, used_files

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or invalid: {dataset_root}")

    for fp in sorted(dataset_root.rglob("*.jsonl")):
        prompts = _load_prompts(fp)
        if not prompts:
            continue
        rel = fp.relative_to(dataset_root)
        if len(rel.parts) >= 2 and rel.parts[0] == "by_category":
            category = rel.parts[1]
        elif len(rel.parts) >= 2:
            category = rel.parts[0]
        else:
            category = fp.parent.name or "uncategorized"
        used_files.append(str(fp))
        for prompt in prompts:
            items.append({"prompt": prompt, "source_file": str(fp), "category": category})

    if not items:
        raise ValueError(f"No prompts found from dataset root: {dataset_root}")
    return items, used_files


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_jsonl_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_key(row: dict, key_fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(k, "")) for k in key_fields)


def _load_existing_csv_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(_row_key(row, key_fields))
    return keys


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, None) is not None


def _jsonl_has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def _sort_summary_files(summary_csv_path: Path, summary_jsonl_path: Path) -> None:
    rows: list[dict] = []
    if summary_csv_path.exists() and summary_csv_path.stat().st_size > 0:
        with summary_csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif summary_jsonl_path.exists() and summary_jsonl_path.stat().st_size > 0:
        for line in summary_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return

    rows.sort(key=lambda r: float(r.get("base_token_prob_delta_mean", 0.0)))
    _save_csv(summary_csv_path, rows)
    with summary_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def _prepare_baseline(model, tokenizer, device: torch.device, prompt_items: list[dict], top_k: int) -> list[dict]:
    n_layers = model.config.num_hidden_layers
    print(f"[1/2] Baseline + donor cache for {len(prompt_items)} prompts...")
    baseline_items: list[dict] = []
    for idx, item in enumerate(prompt_items, start=1):
        input_ids = encode_prompt(tokenizer, item["prompt"], device)
        logits, probs, hidden_by_layer = _forward_with_hidden_cache(model, input_ids, n_layers=n_layers)
        baseline_items.append(
            {
                **item,
                "input_ids": input_ids,
                "baseline_probs": probs,
                "baseline_snapshot": _snapshot(tokenizer, logits, probs, top_k=top_k),
                "hidden_by_layer": hidden_by_layer,
            }
        )
        if idx % 5 == 0 or idx == len(prompt_items):
            print(f"  - baseline done: {idx}/{len(prompt_items)}")
    return baseline_items


def _validate_head_set(selected_multi_heads: list[tuple[int, int]], n_layers: int, n_heads: int) -> None:
    if not selected_multi_heads:
        raise ValueError("Head set is empty.")
    invalid = [
        (layer_idx, head_idx)
        for layer_idx, head_idx in selected_multi_heads
        if layer_idx < 0 or layer_idx >= n_layers or head_idx < 0 or head_idx >= n_heads
    ]
    if invalid:
        raise ValueError(f"Invalid heads for model shape: {invalid}")


def _evaluate_head_set(
    model,
    tokenizer,
    baseline_items: list[dict],
    selected_multi_heads: list[tuple[int, int]],
    top_k: int,
    detailed_snapshots: bool = True,
) -> tuple[list[dict], dict]:
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    if len(baseline_items) < 2:
        raise ValueError("Replace needs at least 2 prompts.")

    heads_by_layer: dict[int, list[int]] = {}
    for layer_idx, head_idx in selected_multi_heads:
        heads_by_layer.setdefault(layer_idx, []).append(head_idx)

    prompt_metrics: list[dict] = []
    for prompt_idx, base in enumerate(baseline_items):
        donor_idx = (prompt_idx + 1) % len(baseline_items)
        donor = baseline_items[donor_idx]

        handles = []
        for layer_idx, head_list in heads_by_layer.items():
            donor_vecs: dict[int, torch.Tensor] = {}
            for head_idx in head_list:
                start = head_idx * head_dim
                end = start + head_dim
                donor_vecs[head_idx] = donor["hidden_by_layer"][layer_idx][start:end]
            h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
                _replace_last_token_heads_hook(head_list, n_heads, donor_vecs)
            )
            handles.append(h)

        modified_logits, modified_probs = forward_last_token(model, base["input_ids"])
        for h in handles:
            h.remove()

        base_top1_id = base["baseline_snapshot"]["top1_id"]
        donor_top1_id = donor["baseline_snapshot"]["top1_id"]
        donor_top1_token = donor["baseline_snapshot"]["top1_token"]
        base_token_prob_delta = float(modified_probs[base_top1_id].item() - base["baseline_probs"][base_top1_id].item())
        base_token_prob_direction = "decrease" if base_token_prob_delta < 0 else "increase_or_same"
        base_token_changed = modified_probs.argmax().item() != base_top1_id
        base_top1_rank_pre_replace = _token_rank(base["baseline_probs"], base_top1_id)
        base_top1_rank_post_replace = _token_rank(modified_probs, base_top1_id)
        donor_prob_pre_replace = float(base["baseline_probs"][donor_top1_id].item())
        donor_prob_post_replace = float(modified_probs[donor_top1_id].item())
        donor_prob_delta = donor_prob_post_replace - donor_prob_pre_replace
        donor_rank_pre_replace = _token_rank(base["baseline_probs"], donor_top1_id)
        donor_rank_post_replace = _token_rank(modified_probs, donor_top1_id)
        donor_rank_change = donor_rank_post_replace - donor_rank_pre_replace

        baseline_snapshot = (
            base["baseline_snapshot"]
            if detailed_snapshots
            else {
                "top1_id": base["baseline_snapshot"]["top1_id"],
                "top1_token": base["baseline_snapshot"]["top1_token"],
                "top1_prob": base["baseline_snapshot"]["top1_prob"],
            }
        )
        replaced_snapshot = (
            _snapshot(tokenizer, modified_logits, modified_probs, top_k=top_k)
            if detailed_snapshots
            else _snapshot_top1(tokenizer, modified_probs)
        )

        prompt_metrics.append(
            {
                "category": base["category"],
                "prompt_index": prompt_idx,
                "prompt": base["prompt"],
                "source_file": base["source_file"],
                "donor_prompt_index": donor_idx,
                "donor_prompt": donor["prompt"],
                "base_token_prob_delta": base_token_prob_delta,
                "base_token_prob_direction": base_token_prob_direction,
                "base_token_changed": base_token_changed,
                "base_token_rank_pre_replace": base_top1_rank_pre_replace,
                "base_token_rank_post_replace": base_top1_rank_post_replace,
                "base_token_rank_change": base_top1_rank_post_replace - base_top1_rank_pre_replace,
                "donor_token_id": donor_top1_id,
                "donor_token": donor_top1_token,
                "donor_token_prob_pre_replace": donor_prob_pre_replace,
                "donor_token_prob_post_replace": donor_prob_post_replace,
                "donor_token_prob_delta": donor_prob_delta,
                "donor_token_rank_pre_replace": donor_rank_pre_replace,
                "donor_token_rank_post_replace": donor_rank_post_replace,
                "donor_token_rank_change": donor_rank_change,
                "baseline": baseline_snapshot,
                "replaced": replaced_snapshot,
            }
        )

    base_token_prob_deltas = [r["base_token_prob_delta"] for r in prompt_metrics]
    donor_token_prob_deltas = [r["donor_token_prob_delta"] for r in prompt_metrics]
    count = len(prompt_metrics)
    summary = {
        "prompt_count": count,
        "base_token_prob_delta_mean": _mean(base_token_prob_deltas),
        "base_token_prob_delta_variance": _variance(base_token_prob_deltas),
        "base_token_prob_decrease_ratio": sum(1 for d in base_token_prob_deltas if d < 0) / count,
        "base_token_changed_ratio": sum(1 for r in prompt_metrics if r["base_token_changed"]) / count,
        "base_token_rank_post_replace_mean": _mean([float(r["base_token_rank_post_replace"]) for r in prompt_metrics]),
        "donor_token_prob_delta_mean": _mean(donor_token_prob_deltas),
        "donor_token_prob_delta_variance": _variance(donor_token_prob_deltas),
        "donor_token_prob_increase_ratio": sum(1 for d in donor_token_prob_deltas if d > 0) / count,
        "donor_token_rank_up_ratio": sum(1 for r in prompt_metrics if r["donor_token_rank_change"] < 0) / count,
        "donor_token_rank_pre_replace_mean": _mean([float(r["donor_token_rank_pre_replace"]) for r in prompt_metrics]),
        "donor_token_rank_post_replace_mean": _mean([float(r["donor_token_rank_post_replace"]) for r in prompt_metrics]),
    }
    return prompt_metrics, summary


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-") or "uncategorized"


def _output_bucket_parts(item: dict) -> tuple[str, str]:
    category = _slug(str(item.get("category", "uncategorized")))
    source_name = _slug(Path(str(item.get("source_file", "source"))).stem)
    return category, source_name


def _build_prompt_output_rows(bucket_items: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(bucket_items):
        snap = item["baseline_snapshot"]
        rows.append(
            {
                "prompt_index": idx,
                "prompt": item["prompt"],
                "base_token": snap["top1_token"],
                "base_token_prob": snap["top1_prob"],
            }
        )
    return rows


def _write_prompt_output_maps(out_dir: Path, baseline_items: list[dict]) -> None:
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for item in baseline_items:
        by_bucket.setdefault(_output_bucket_parts(item), []).append(item)

    for bucket, bucket_items in by_bucket.items():
        category, source_name = bucket
        bucket_dir = out_dir / category / source_name
        rows = _build_prompt_output_rows(bucket_items)
        _save_csv(bucket_dir / "prompt_output_map.csv", rows)
        _save_jsonl(bucket_dir / "prompt_output_map.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace head intervention with prompt/global metrics.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT_DIR / "datasets"),
        help="Dataset root directory (.jsonl files are discovered recursively)",
    )
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional single prompt file (.jsonl only). If set, dataset-root is ignored",
    )
    parser.add_argument(
        "--multi-heads",
        default="",
        help="Comma-separated head labels (e.g. L1.H2,L3.H5). Used when not scanning all heads.",
    )
    parser.add_argument(
        "--scan-all-heads",
        action="store_true",
        help="Run every single (layer,head) and write one summary row per head.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-k to store for pre/post replace snapshots")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible runs")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "outputs"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    if args.scan_all_heads and args.multi_heads.strip():
        raise ValueError("Use either --scan-all-heads or --multi-heads, not both.")
    if not args.scan_all_heads and not args.multi_heads.strip():
        raise ValueError("Provide --multi-heads, or use --scan-all-heads.")

    device = get_device() if args.device == "auto" else torch.device(args.device)
    configure_reproducibility(args.seed, device)
    prompt_items, _ = _load_prompt_items(Path(args.dataset_root), args.prompts_file)

    print(f"Loading model: {DEFAULT_MODEL_NAME} on {device} ...")
    model, tokenizer = load_model(DEFAULT_MODEL_NAME, str(device))
    print("Model loaded.")

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    baseline_items = _prepare_baseline(model, tokenizer, device, prompt_items, top_k=max(1, args.top_k))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_prompt_output_maps(out_dir, baseline_items)
    print(f"- wrote prompt/output maps under: {out_dir}")

    if args.scan_all_heads:
        by_bucket: dict[tuple[str, str], list[dict]] = {}
        for item in baseline_items:
            by_bucket.setdefault(_output_bucket_parts(item), []).append(item)

        print(f"[2/2] Evaluating all heads ({n_layers * n_heads} rows x {len(by_bucket)} buckets)...")
        total = n_layers * n_heads
        for bucket in sorted(by_bucket.keys()):
            category, source_name = bucket
            bucket_items = by_bucket[bucket]
            if len(bucket_items) < 2:
                print(f"  - skip bucket '{category}/{source_name}': requires at least 2 prompts")
                continue

            bucket_dir = out_dir / category / source_name
            summary_csv_path = bucket_dir / "summary_by_head.csv"
            summary_jsonl_path = bucket_dir / "summary_by_head.jsonl"
            prompt_head_csv_path = bucket_dir / "prompt_by_head.csv"
            prompt_head_jsonl_path = bucket_dir / "prompt_by_head.jsonl"
            summary_key_fields = ("head_label", "prompt_count")
            prompt_key_fields = ("head_label", "prompt_index")
            had_summary_data = _jsonl_has_data_rows(summary_jsonl_path)
            existing_summary_keys = _load_existing_csv_keys(summary_csv_path, summary_key_fields)
            existing_prompt_keys = _load_existing_csv_keys(prompt_head_csv_path, prompt_key_fields)
            summary_candidates: list[dict] = []
            prompt_rows_to_append: list[dict] = []
            done = 0
            for layer in range(n_layers):
                for head in range(n_heads):
                    head_set = [(layer, head)]
                    _validate_head_set(head_set, n_layers, n_heads)
                    prompt_metrics, summary = _evaluate_head_set(
                        model=model,
                        tokenizer=tokenizer,
                        baseline_items=bucket_items,
                        selected_multi_heads=head_set,
                        top_k=max(1, args.top_k),
                        detailed_snapshots=False,
                    )
                    row = {
                        "head_label": _head_label(layer, head),
                        **summary,
                    }
                    summary_key = _row_key(row, summary_key_fields)
                    if summary_key not in existing_summary_keys:
                        summary_candidates.append(row)
                    prompt_head_rows = [
                        {
                            "head_label": _head_label(layer, head),
                            "prompt_index": r["prompt_index"],
                            "base_token_prob_delta": r["base_token_prob_delta"],
                            "base_token_prob_direction": r["base_token_prob_direction"],
                            "base_token_changed": r["base_token_changed"],
                            "base_token_rank_pre_replace": r["base_token_rank_pre_replace"],
                            "base_token_rank_post_replace": r["base_token_rank_post_replace"],
                            "base_token_rank_change": r["base_token_rank_change"],
                            "donor_token": r["donor_token"],
                            "donor_token_prob_pre_replace": r["donor_token_prob_pre_replace"],
                            "donor_token_prob_post_replace": r["donor_token_prob_post_replace"],
                            "donor_token_prob_delta": r["donor_token_prob_delta"],
                            "donor_token_rank_pre_replace": r["donor_token_rank_pre_replace"],
                            "donor_token_rank_post_replace": r["donor_token_rank_post_replace"],
                            "donor_token_rank_change": r["donor_token_rank_change"],
                            "baseline_base_token": r["baseline"]["top1_token"],
                            "baseline_base_token_prob": r["baseline"]["top1_prob"],
                            "replaced_base_token": r["replaced"]["top1_token"],
                            "replaced_base_token_prob": r["replaced"]["top1_prob"],
                        }
                        for r in prompt_metrics
                    ]
                    new_prompt_rows = []
                    for r in prompt_head_rows:
                        k = _row_key(r, prompt_key_fields)
                        if k in existing_prompt_keys:
                            continue
                        existing_prompt_keys.add(k)
                        new_prompt_rows.append(r)
                    prompt_rows_to_append.extend(new_prompt_rows)
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"  - [{category}/{source_name}] head done: {done}/{total}")

            _append_csv_rows(prompt_head_csv_path, prompt_rows_to_append)
            _append_jsonl_rows(prompt_head_jsonl_path, prompt_rows_to_append)

            selected_summary_rows: list[dict] = []
            for threshold in [SUMMARY_DECREASE_RATIO_THRESHOLD, *SUMMARY_DECREASE_RATIO_FALLBACKS]:
                tier_rows = [
                    row
                    for row in summary_candidates
                    if row["base_token_prob_decrease_ratio"] >= threshold
                    and row["base_token_prob_delta_mean"] < SUMMARY_DELTA_MEAN_THRESHOLD
                ]
                if tier_rows:
                    selected_summary_rows = tier_rows
                    break

            if selected_summary_rows:
                _append_csv_rows(summary_csv_path, selected_summary_rows)
                _append_jsonl_rows(summary_jsonl_path, selected_summary_rows)
            elif not had_summary_data:
                # Leave summary empty when no threshold tier yields candidates.
                pass

            _sort_summary_files(summary_csv_path, summary_jsonl_path)

        print("Done.")
        print(f"- appended per-category dirs under: {out_dir}")
        return

    selected_multi_heads = [_parse_head_label(x) for x in args.multi_heads.split(",") if x.strip()]
    _validate_head_set(selected_multi_heads, n_layers, n_heads)
    prompt_metrics, summary = _evaluate_head_set(
        model=model,
        tokenizer=tokenizer,
        baseline_items=baseline_items,
        selected_multi_heads=selected_multi_heads,
        top_k=max(1, args.top_k),
    )

    # Store outputs only per category (no mixed global files).
    by_bucket_metrics: dict[tuple[str, str], list[dict]] = {}
    for row in prompt_metrics:
        by_bucket_metrics.setdefault(_output_bucket_parts(row), []).append(row)
    for bucket, rows in by_bucket_metrics.items():
        category, source_name = bucket
        bucket_dir = out_dir / category / source_name
        cat_prompt_json_path = bucket_dir / f"prompt_metrics_{ts}.json"
        cat_prompt_csv_path = bucket_dir / f"prompt_metrics_{ts}.csv"
        cat_summary_json_path = bucket_dir / f"summary_{ts}.json"
        cat_summary_csv_path = bucket_dir / f"summary_{ts}.csv"
        base_token_prob_deltas = [r["base_token_prob_delta"] for r in rows]
        donor_token_prob_deltas = [r["donor_token_prob_delta"] for r in rows]
        count = len(rows)
        cat_summary = {
            "head_set": args.multi_heads,
            "prompt_count": count,
            "base_token_prob_delta_mean": _mean(base_token_prob_deltas),
            "base_token_prob_delta_variance": _variance(base_token_prob_deltas),
            "base_token_prob_decrease_ratio": sum(1 for d in base_token_prob_deltas if d < 0) / count,
            "base_token_changed_ratio": sum(1 for r in rows if r["base_token_changed"]) / count,
            "base_token_rank_post_replace_mean": _mean([float(r["base_token_rank_post_replace"]) for r in rows]),
            "donor_token_prob_delta_mean": _mean(donor_token_prob_deltas),
            "donor_token_prob_delta_variance": _variance(donor_token_prob_deltas),
            "donor_token_prob_increase_ratio": sum(1 for d in donor_token_prob_deltas if d > 0) / count,
            "donor_token_rank_up_ratio": sum(1 for r in rows if r["donor_token_rank_change"] < 0) / count,
            "donor_token_rank_pre_replace_mean": _mean([float(r["donor_token_rank_pre_replace"]) for r in rows]),
            "donor_token_rank_post_replace_mean": _mean([float(r["donor_token_rank_post_replace"]) for r in rows]),
        }
        cat_prompt_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        cat_summary_json_path.write_text(json.dumps(cat_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_csv(
            cat_prompt_csv_path,
            [
                {
                    "category": r["category"],
                    "prompt_index": r["prompt_index"],
                    "source_file": r["source_file"],
                    "base_token_prob_delta": r["base_token_prob_delta"],
                    "base_token_prob_direction": r["base_token_prob_direction"],
                    "base_token_changed": r["base_token_changed"],
                    "base_token_rank_pre_replace": r["base_token_rank_pre_replace"],
                    "base_token_rank_post_replace": r["base_token_rank_post_replace"],
                    "base_token_rank_change": r["base_token_rank_change"],
                    "donor_token": r["donor_token"],
                    "donor_token_prob_pre_replace": r["donor_token_prob_pre_replace"],
                    "donor_token_prob_post_replace": r["donor_token_prob_post_replace"],
                    "donor_token_prob_delta": r["donor_token_prob_delta"],
                    "donor_token_rank_pre_replace": r["donor_token_rank_pre_replace"],
                    "donor_token_rank_post_replace": r["donor_token_rank_post_replace"],
                    "donor_token_rank_change": r["donor_token_rank_change"],
                    "baseline_base_token": r["baseline"]["top1_token"],
                    "baseline_base_token_prob": r["baseline"]["top1_prob"],
                    "replaced_base_token": r["replaced"]["top1_token"],
                    "replaced_base_token_prob": r["replaced"]["top1_prob"],
                }
                for r in rows
            ],
        )
        _save_csv(cat_summary_csv_path, [cat_summary])

    print("Done.")
    print(f"- per-category outputs under: {out_dir}")


if __name__ == "__main__":
    main()
