# -*- coding: utf-8 -*-
"""C2：Prompt 变体特征提取（Qwen3-4B-Base，单进程一次前向提取全部 13 层）。

对每个变体 × split 的 CSV，逐 batch（默认 8，与 extract_intrep.py 一致）前向，
在 13 个 decoder block 的输出上挂 hook，一次前向取全部层；token 选择：
    origcountry / nameonly -> country   （国家名最后一个 subtoken）
    origlast / nocountry / rephrase -> last（句末标点）
其余细节与仓库 extract_intrep.py 完全一致（rev 906bfd4b、bf16、AutoModel、
无 chat template、左 padding、use_cache=False、block 输出 RMSNorm 之前）。

自检：origcountry 的提取结果与 datasets/world 现有特征逐层比较
（逐行余弦均值 / 最大值误差）；余弦 < 0.999 视为管线错配并中止。
输出：<out_dir>/<variant>/<split>.layer<L>.npy（float16，CSV 行序）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

LAYERS_DEFAULT = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 35)
VARIANTS = ("origcountry", "origlast", "nocountry", "nameonly", "rephrase")
TOKEN_RULE = {
    "origcountry": "country",
    "origlast": "last",
    "nocountry": "last",
    "nameonly": "country",
    "rephrase": "last",
}
COSINE_MIN = 0.999


def load_rows(path: Path, selection: str) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence", "country"} if selection == "country" else {"sentence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: CSV requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows or any(not (row.get("sentence") or "").strip() for row in rows):
        raise ValueError(f"{path}: CSV must contain nonempty sentences")
    return rows


def country_span(row: dict[str, str]) -> tuple[int, int]:
    sentence, country = row["sentence"], row.get("country") or ""
    start = sentence.rfind(country) if country else -1
    end = start + len(country)
    # 原句/换问法句尾是 "?"；nameonly 的句尾就是国家名本身（无标点）
    if start < 0 or sentence[end:].strip() not in ("", "?"):
        raise ValueError(f"Country must occur at the end of the sentence ({sentence!r})")
    return start, end


def select_token(row: dict[str, str], offsets: list[list[int]], selection: str) -> int:
    if selection == "last":
        candidates = [i for i, (start, end) in enumerate(offsets) if end > start]
    else:
        start, end = country_span(row)
        candidates = [
            i
            for i, (left, right) in enumerate(offsets)
            if right > left and left < end and right > start
        ]
    if not candidates:
        raise ValueError(f"No token found for selection {selection!r}")
    return candidates[-1]


def load_model(model_dir: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; run this script in a GPU job")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for token offsets")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to("cuda").eval()
    if model.config.model_type != "qwen3" or model.config.hidden_size != 2560 or len(model.layers) != 36:
        raise ValueError("Expected Qwen3-4B-Base with 36 blocks and width 2560")
    return tokenizer, model


def extract_variant(rows, tokenizer, model, *, layers, selection, batch_size):
    import torch

    result = {layer: np.empty((len(rows), 2560), dtype=np.float16) for layer in layers}
    state = {"indices": []}

    def make_hook(layer):
        def hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            batch_indices = torch.arange(len(state["indices"]), device=hidden.device)
            selected = hidden[batch_indices, state["indices"], :]
            selected = selected.detach().float().cpu().numpy()
            result[layer][offset : offset + len(state["indices"])] = selected

        return hook

    handles = [model.layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    token_ids = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                tokens = tokenizer(
                    [row["sentence"] for row in batch],
                    padding=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=False,
                )
                spans = tokens.pop("offset_mapping").tolist()
                state["indices"] = [
                    select_token(row, span, selection) for row, span in zip(batch, spans, strict=True)
                ]
                token_ids.extend(int(tokens.input_ids[i, j]) for i, j in enumerate(state["indices"]))
                if tokens.input_ids.shape[1] > model.config.max_position_embeddings:
                    raise ValueError("Input exceeds the model context length")
                model(**tokens.to("cuda"), use_cache=False)
                print(f"    {offset + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for layer in layers:
        if not np.isfinite(result[layer]).all():
            raise ValueError(f"non-finite features in layer {layer}")
    return result, token_ids


def save_features(out_dir: Path, variant: str, split: str, features: dict, layers) -> int:
    target_dir = out_dir / variant
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for layer in layers:
        path = target_dir / f"{split}.layer{layer}.npy"
        if path.exists():
            continue
        with path.open("xb") as stream:
            np.save(stream, features[layer], allow_pickle=False)
        saved += 1
    return saved


def check_against_world(out_dir: Path, world_dir: Path, layers, splits) -> dict:
    report = {}
    problems = []
    for split in splits:
        for layer in layers:
            ref_path = world_dir / f"{split}.layer{layer}.npy"
            new_path = out_dir / "origcountry" / f"{split}.layer{layer}.npy"
            if not ref_path.exists() or not new_path.exists():
                continue
            ref = np.load(ref_path).astype(np.float64)
            new = np.load(new_path).astype(np.float64)
            if ref.shape != new.shape:
                problems.append(f"{split} L{layer}: shape {new.shape} != {ref.shape}")
                continue
            ref_norm = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
            new_norm = new / np.clip(np.linalg.norm(new, axis=1, keepdims=True), 1e-12, None)
            cos = np.sum(ref_norm * new_norm, axis=1)
            max_abs = float(np.max(np.abs(ref - new)))
            entry = {
                "min_cosine": float(np.min(cos)),
                "mean_cosine": float(np.mean(cos)),
                "max_abs_diff": max_abs,
            }
            report[f"{split}/L{layer}"] = entry
            if entry["mean_cosine"] < COSINE_MIN:
                problems.append(
                    f"{split} L{layer}: mean cosine {entry['mean_cosine']:.6f} < {COSINE_MIN}"
                )
    return {"layers": report, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).with_name("features"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in LAYERS_DEFAULT))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--check_world_dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets" / "world")
    parser.add_argument("--skip_check", action="store_true")
    args = parser.parse_args()

    variants = tuple(args.variants.split(","))
    for variant in variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant: {variant}")
    splits = tuple(args.splits.split(","))
    layers = tuple(int(value) for value in args.layers.split(","))

    tokenizer, model = load_model(args.model_dir)
    report = {"variants": {}, "token_id_checks": {}}
    token_ids_by_variant = {}
    for variant in variants:
        selection = TOKEN_RULE[variant]
        for split in splits:
            rows = load_rows(args.data_dir / variant / f"{split}.csv", selection)
            target_dir = args.out_dir / variant
            wanted = [target_dir / f"{split}.layer{layer}.npy" for layer in layers]
            if all(path.exists() for path in wanted):
                print(f"SKIP  {variant}/{split} (all layers exist)", flush=True)
                continue
            print(f"----- {variant}/{split} (n={len(rows)}, selection={selection}) -----", flush=True)
            features, token_ids = extract_variant(
                rows, tokenizer, model, layers=layers, selection=selection, batch_size=args.batch_size
            )
            saved = save_features(args.out_dir, variant, split, features, layers)
            report["variants"][f"{variant}/{split}"] = {
                "n": len(rows),
                "selection": selection,
                "layers": list(layers),
                "saved": saved,
            }
            token_ids_by_variant[(variant, split)] = token_ids
            print(f"  saved {saved} layer files", flush=True)

    # 同一行上 nameonly 与 origcountry 选中的 token 应完全一致（同样的国家名后缀）
    for split in splits:
        left = token_ids_by_variant.get(("origcountry", split))
        right = token_ids_by_variant.get(("nameonly", split))
        if left and right and len(left) == len(right):
            mismatch = sum(1 for a, b in zip(left, right) if a != b)
            report["token_id_checks"][split] = {"n": len(left), "mismatch": mismatch}

    if not args.skip_check and "origcountry" in variants:
        check = check_against_world(args.out_dir, args.check_world_dir, layers, splits)
        report["world_baseline_check"] = check
        for key, entry in check["layers"].items():
            print(
                f"check {key}: min_cos={entry['min_cosine']:.6f} "
                f"mean_cos={entry['mean_cosine']:.6f} max_abs={entry['max_abs_diff']:.6f}",
                flush=True,
            )
        if check["problems"]:
            (args.out_dir / "extract_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise SystemExit("Feature check failed: " + "; ".join(check["problems"][:3]))

    (args.out_dir / "extract_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out_dir / 'extract_report.json'}")


if __name__ == "__main__":
    main()
