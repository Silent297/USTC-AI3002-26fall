# -*- coding: utf-8 -*-
"""C1：把 world 原问题直接喂给 Qwen3-4B-Base，贪心生成经纬度（集群 GPU）。

协议（指南3 §4.1，同学参考报告，非官方）：
- prompt = 原问题句 + 固定 JSON 指令（c1_lib.PROMPT_TEMPLATE；不加聊天模板/示例/标签）；
- BF16 非量化、batch=4、贪心（do_sample=False）、max_new_tokens=64；
- 逐条保存原始回答，事后离线解析（本脚本不判解析率）。

输出 <out-dir>/：
- train.jsonl / valid.jsonl：每行一条记录，行序 = CSV 行序（结束时按 row 重排落盘）；
- run_meta.json：prompt 模板逐字 + 生成参数 + 模型文件 sha256 指纹 + CSV md5 + tier 计数。
幂等：JSONL 完整 → SKIP；缺行 → 按 row 续写（每行即时 flush，被中断可续跑）；
run_meta 与现场不一致（prompt 模板 / 指纹 / CSV）→ 硬报错退出（防混入不同协议）。

本地冒烟（CPU、tiny 模型）可用环境开关：
    C1_ALLOW_NON_A100=1  —— 放宽「必须 A100」检查（无 CUDA 时退回 CPU）；
    --expect-model-type/--expect-hidden/--expect-layers 覆盖架构断言（默认 qwen3/2560/36）。

用法（集群，仓库根）：
    python -u experiments/C1/c1_qwen_direct.py \
        --model-dir ~/models/Qwen3-4B-Base \
        --train-csv datasets/world/train.csv --valid-csv datasets/world/valid.csv \
        --out-dir results/slurm_qwen_direct/qwen_direct_v1 --batch-size 4 --max-new-tokens 64
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from c1_lib import PROMPT_TEMPLATE, build_prompt, parse_output

SCHEMA_VERSION = 1
REVISION_NOTE = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
FINGERPRINT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
RECORD_FIELDS = (
    "split", "row", "sentence", "country", "lat_true", "lon_true",
    "prompt", "raw_output", "tier", "pred_lat", "pred_lon",
    "lat_raw", "lon_raw", "in_range", "n_new_tokens", "finish", "notes",
)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=None, help="每 split 只跑前 N 行（冒烟用）")
    parser.add_argument("--expect-model-type", default="qwen3")
    parser.add_argument("--expect-hidden", type=int, default=2560)
    parser.add_argument("--expect-layers", type=int, default=36)
    return parser.parse_args()


def load_rows(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence", "country", "latitude", "longitude"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path}: CSV requires columns {sorted(required)}")
        rows = []
        for row in reader:
            if not (row.get("sentence") or "").strip():
                raise ValueError(f"{csv_path}: empty sentence at CSV row {len(rows)}")
            rows.append({
                "sentence": row["sentence"],
                "country": row["country"],
                "lat_true": float(row["latitude"]),
                "lon_true": float(row["longitude"]),
            })
    if not rows:
        raise ValueError(f"{csv_path}: no data rows")
    return rows


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprints(model_dir):
    fingerprints = {}
    for pattern in FINGERPRINT_PATTERNS:
        for path in sorted(model_dir.glob(pattern)):
            fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def check_device():
    import torch

    allow_any = os.environ.get("C1_ALLOW_NON_A100") == "1"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if "A100" in name:
            return "cuda", name
        if allow_any:
            print(f"WARNING: GPU {name!r} is not A100; C1_ALLOW_NON_A100=1 allows it", flush=True)
            return "cuda", name
        print(f"ERROR: GPU {name!r} is not A100 (RTX5090 nodes hit CUDA error 804 with this torch). "
              "Set C1_ALLOW_NON_A100=1 to override.", file=sys.stderr, flush=True)
        sys.exit(2)
    if allow_any:
        print("WARNING: no CUDA GPU; C1_ALLOW_NON_A100=1 falls back to CPU", flush=True)
        return "cpu", "cpu"
    print("ERROR: no usable CUDA GPU; a pinned A100 is required.", file=sys.stderr, flush=True)
    sys.exit(2)


def load_model(args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=dtype)
    model = model.to(device).eval()

    config = model.config
    problems = []
    if config.model_type != args.expect_model_type:
        problems.append(f"model_type={config.model_type!r} (expected {args.expect_model_type!r})")
    if getattr(config, "hidden_size", None) != args.expect_hidden:
        problems.append(f"hidden_size={getattr(config, 'hidden_size', None)} (expected {args.expect_hidden})")
    if getattr(config, "num_hidden_layers", None) != args.expect_layers:
        problems.append(f"num_hidden_layers={getattr(config, 'num_hidden_layers', None)} (expected {args.expect_layers})")
    if problems:
        raise SystemExit("Model architecture check failed: " + "; ".join(problems))
    return tokenizer, model


def generate_batch(tokenizer, model, prompts, max_new_tokens):
    import torch

    enc = tokenizer(prompts, padding=True, return_tensors="pt", truncation=False)
    width = int(enc["input_ids"].shape[1])
    if width > model.config.max_position_embeddings:
        raise ValueError(f"input width {width} exceeds max_position_embeddings")
    enc = {key: value.to(model.device) for key, value in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[:, width:]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    results = []
    for ids in new_ids.tolist():
        n = len(ids)
        finish = "length"
        for index, token_id in enumerate(ids):
            if token_id == eos_id:
                n, finish = index, "eos"
                break
            if pad_id is not None and token_id == pad_id:
                n, finish = index, "pad"
                break
        raw = tokenizer.decode(ids[:n], skip_special_tokens=True)
        results.append({"raw_output": raw, "n_new_tokens": n, "finish": finish})
    return results


def run_batch_with_fallback(tokenizer, model, prompts, max_new_tokens, context):
    """批级异常（含 OOM）→ 逐条 bs=1 重试；返回 (results, fallback_entry_or_None)。"""
    import torch

    try:
        return generate_batch(tokenizer, model, prompts, max_new_tokens), None
    except Exception as exc:
        print(f"WARNING: {context} batch failed ({exc!r}); retrying row by row", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = []
        for prompt in prompts:
            try:
                results.append(generate_batch(tokenizer, model, [prompt], max_new_tokens)[0])
            except Exception as retry_exc:
                raise RuntimeError(f"{context} single-row retry failed: {retry_exc!r}") from retry_exc
        return results, {"context": context, "error": repr(exc)}


def make_record(split, row_index, row, prompt, result):
    parsed = parse_output(result["raw_output"])
    return {
        "split": split,
        "row": row_index,
        "sentence": row["sentence"],
        "country": row["country"],
        "lat_true": row["lat_true"],
        "lon_true": row["lon_true"],
        "prompt": prompt,
        "raw_output": result["raw_output"],
        "tier": parsed["tier"],
        "pred_lat": parsed["pred_lat"],
        "pred_lon": parsed["pred_lon"],
        "lat_raw": parsed["lat_raw"],
        "lon_raw": parsed["lon_raw"],
        "in_range": parsed["in_range"],
        "n_new_tokens": result["n_new_tokens"],
        "finish": result["finish"],
        "notes": parsed["notes"],
    }


def load_existing(path, split, rows, n_use):
    """读取已有 JSONL：返回 (records{row: rec}, n_lines, n_bad)。坏行/越界行会被丢弃（稍后重写）。"""
    records = {}
    if not path.exists():
        return records, 0, 0
    with path.open(encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    n_bad = 0
    for line in lines:
        if not line.strip():
            n_bad += 1
            continue
        try:
            rec = json.loads(line)
            row_index = int(rec["row"])
            if not 0 <= row_index < n_use:
                raise ValueError(f"row {row_index} out of range")
            if rec.get("prompt") != build_prompt(rows[row_index]["sentence"]):
                raise SystemExit(
                    f"{split} row {row_index}: existing prompt does not match current template; "
                    "refusing to mix protocols (use a fresh --out-dir)"
                )
            records[row_index] = rec
        except SystemExit:
            raise
        except Exception:
            n_bad += 1
    return records, len(lines), n_bad


def append_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def rewrite_sorted(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row_index in sorted(records):
            stream.write(json.dumps(records[row_index], ensure_ascii=False) + "\n")
    tmp.replace(path)


def tier_counts(records):
    counts = {}
    for rec in records.values():
        counts[rec["tier"]] = counts.get(rec["tier"], 0) + 1
    return counts


def verify_meta(meta_path, expected):
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    mismatches = []
    if meta.get("schema_version") != SCHEMA_VERSION:
        mismatches.append("schema_version")
    for key, value in expected.items():
        if meta.get(key) != value:
            mismatches.append(key)
    if mismatches:
        raise SystemExit(
            f"{meta_path} mismatch on {mismatches}; refusing to mix runs "
            "(use a fresh --out-dir or remove stale run files)"
        )
    return meta


def run_split(split, csv_path, args, tokenizer=None, model=None):
    rows = load_rows(csv_path)
    n_total = len(rows)
    n_use = min(args.limit, n_total) if args.limit else n_total
    jsonl_path = args.out_dir / f"{split}.jsonl"

    records, n_lines, n_bad = load_existing(jsonl_path, split, rows, n_use)
    missing = [index for index in range(n_use) if index not in records]
    print(f"[{split}] rows={n_use}/{n_total} existing={len(records)} bad_lines={n_bad} missing={len(missing)}", flush=True)
    fallbacks = []
    if missing:
        started = time.time()
        done = 0
        with jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
            run_start = 0
            while run_start < len(missing):
                run_end = run_start
                while (
                    run_end + 1 < len(missing)
                    and missing[run_end + 1] == missing[run_end] + 1
                    and run_end - run_start + 1 < args.batch_size
                ):
                    run_end += 1
                chunk = missing[run_start : run_end + 1]
                prompts = [build_prompt(rows[index]["sentence"]) for index in chunk]
                results, fallback = run_batch_with_fallback(
                    tokenizer, model, prompts, args.max_new_tokens,
                    context=f"{split} rows {chunk[0]}..{chunk[-1]}",
                )
                if fallback is not None:
                    fallbacks.append(fallback)
                for index, prompt, result in zip(chunk, prompts, results):
                    record = make_record(split, index, rows[index], prompt, result)
                    records[index] = record
                    append_record(stream, record)
                done += len(chunk)
                elapsed = time.time() - started
                print(f"[{split}] {done}/{len(missing)} new rows, {elapsed:.1f}s", flush=True)
                run_start = run_end + 1
        rewrite_sorted(jsonl_path, records)
        remaining = [index for index in range(n_use) if index not in records]
        if remaining:
            raise RuntimeError(f"{split}: {len(remaining)} rows still missing after generation: {remaining[:10]}")
    else:
        print(f"[{split}] complete ({n_use} rows); skip generation", flush=True)
        if n_bad:
            rewrite_sorted(jsonl_path, records)
            print(f"[{split}] dropped {n_bad} bad/blank lines", flush=True)

    # 预览前 8 条 + tier 统计（进 slurm 日志）
    for index in range(min(8, n_use)):
        rec = records[index]
        raw_preview = rec["raw_output"][:200].replace("\n", "\\n")
        print(f"[{split}] row {index} tier={rec['tier']} pred=({rec['pred_lat']},{rec['pred_lon']}) "
              f"true=({rec['lat_true']},{rec['lon_true']}) raw={raw_preview!r}", flush=True)
    print(f"[{split}] tiers={tier_counts(records)}", flush=True)
    return rows, n_use, records, fallbacks


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_time = utcnow()

    expected_meta = {
        "prompt_template": PROMPT_TEMPLATE,
        "gen": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "dtype": args.dtype,
            "model_dir": str(args.model_dir),
        },
        "csv_md5": {
            "train": md5_file(args.train_csv),
            "valid": md5_file(args.valid_csv),
        },
    }
    meta_path = args.out_dir / "run_meta.json"
    print(f"Model fingerprints of {args.model_dir} ...", flush=True)
    expected_meta["model_fingerprints"] = model_fingerprints(args.model_dir)
    verify_meta(meta_path, expected_meta)

    device, device_name = check_device()
    tokenizer, model = load_model(args, device)
    print(f"Device: {device} ({device_name}); dtype={args.dtype}", flush=True)
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}, "
          f"chat_template={'None' if tokenizer.chat_template is None else 'set'}, "
          f"pad={tokenizer.pad_token!r}/{tokenizer.pad_token_id}, eos={tokenizer.eos_token!r}/{tokenizer.eos_token_id}", flush=True)

    train_rows, n_train, train_records, train_fallbacks = run_split(
        "train", args.train_csv, args, tokenizer, model
    )
    valid_rows, n_valid, valid_records, valid_fallbacks = run_split(
        "valid", args.valid_csv, args, tokenizer, model
    )

    import torch
    import transformers

    meta = dict(expected_meta)
    meta.update({
        "schema_version": SCHEMA_VERSION,
        "revision_note": REVISION_NOTE,
        "prompt_template_repr": repr(PROMPT_TEMPLATE),
        "argv": sys.argv,
        "limit": args.limit,
        "expected_rows": {"train": n_train, "valid": n_valid},
        "total_rows": {"train": len(train_rows), "valid": len(valid_rows)},
        "tiers": {"train": tier_counts(train_records), "valid": tier_counts(valid_records)},
        "fallback_batches": train_fallbacks + valid_fallbacks,
        "tokenizer_info": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "chat_template_is_none": tokenizer.chat_template is None,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "env": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": device,
            "device_name": device_name,
        },
        "started_utc": start_time,
        "finished_utc": utcnow(),
    })
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
    print(f"Wrote {meta_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
