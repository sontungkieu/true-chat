#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 131072)
DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    layers: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    source: str = "fallback"


FALLBACK_QWEN25_SPECS = {
    "Qwen/Qwen2.5-0.5B": ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B",
        layers=24,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=896,
        num_attention_heads=14,
    ),
    "Qwen/Qwen2.5-1.5B": ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B",
        layers=28,
        num_key_value_heads=2,
        head_dim=128,
        hidden_size=1536,
        num_attention_heads=12,
    ),
    "Qwen/Qwen2.5-3B": ModelSpec(
        model_id="Qwen/Qwen2.5-3B",
        layers=36,
        num_key_value_heads=2,
        head_dim=128,
        hidden_size=2048,
        num_attention_heads=16,
    ),
    "Qwen/Qwen2.5-7B": ModelSpec(
        model_id="Qwen/Qwen2.5-7B",
        layers=28,
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=3584,
        num_attention_heads=28,
    ),
    "Qwen/Qwen2.5-14B": ModelSpec(
        model_id="Qwen/Qwen2.5-14B",
        layers=48,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=5120,
        num_attention_heads=40,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate local Qwen KV-cache memory from config metadata.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated Hugging Face model ids.")
    parser.add_argument(
        "--seq-lens",
        default=",".join(str(value) for value in DEFAULT_SEQUENCE_LENGTHS),
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype-bytes", type=int, default=2, help="2 for fp16/bf16, 4 for fp32, 1 for int8 KV.")
    parser.add_argument("--use-auto-config", action="store_true", help="Try transformers.AutoConfig before fallback specs.")
    parser.add_argument("--out-md", type=Path, default=Path("docs/reports/local_qwen_kv_estimates.md"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    model_ids = _parse_csv(args.models)
    seq_lens = [int(value) for value in _parse_csv(args.seq_lens)]
    rows = estimate_table(
        model_ids=model_ids,
        seq_lens=seq_lens,
        batch_size=args.batch_size,
        dtype_bytes=args.dtype_bytes,
        use_auto_config=args.use_auto_config,
    )
    summary = {
        "formula": "2 * layers * num_key_value_heads * head_dim * seq_len * batch_size * dtype_bytes",
        "batch_size": args.batch_size,
        "dtype_bytes": args.dtype_bytes,
        "rows": rows,
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_csv, rows)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "row_count": len(rows)}, indent=2))
    return 0


def estimate_table(
    *,
    model_ids: list[str],
    seq_lens: list[int],
    batch_size: int,
    dtype_bytes: int,
    use_auto_config: bool = False,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        spec = load_model_spec(model_id, use_auto_config=use_auto_config)
        for seq_len in seq_lens:
            if seq_len <= 0:
                raise ValueError("seq_lens must be positive")
            kv_bytes = estimate_kv_bytes(
                spec,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype_bytes=dtype_bytes,
            )
            rows.append(
                {
                    **asdict(spec),
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "dtype_bytes": dtype_bytes,
                    "kv_bytes": kv_bytes,
                    "kv_mib": kv_bytes / (1024**2),
                    "kv_gib": kv_bytes / (1024**3),
                }
            )
    return rows


def load_model_spec(model_id: str, *, use_auto_config: bool = False) -> ModelSpec:
    if use_auto_config:
        try:
            from transformers import AutoConfig  # type: ignore

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return spec_from_config(model_id, config)
        except Exception:
            pass
    if model_id not in FALLBACK_QWEN25_SPECS:
        raise ValueError(f"No fallback spec for {model_id}; use --use-auto-config or add a fallback spec.")
    return FALLBACK_QWEN25_SPECS[model_id]


def spec_from_config(model_id: str, config: Any) -> ModelSpec:
    layers = int(getattr(config, "num_hidden_layers"))
    hidden_size = int(getattr(config, "hidden_size"))
    attention_heads = int(getattr(config, "num_attention_heads"))
    key_value_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // attention_heads))
    return ModelSpec(
        model_id=model_id,
        layers=layers,
        num_key_value_heads=key_value_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_attention_heads=attention_heads,
        source="AutoConfig",
    )


def estimate_kv_bytes(spec: ModelSpec, *, seq_len: int, batch_size: int, dtype_bytes: int) -> int:
    return 2 * spec.layers * spec.num_key_value_heads * spec.head_dim * seq_len * batch_size * dtype_bytes


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    lines = [
        "# Local Qwen KV-Cache Estimates",
        "",
        "This report estimates decoder KV-cache memory without loading model weights.",
        "",
        f"- Formula: `{summary['formula']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Dtype bytes: `{summary['dtype_bytes']}`",
        "",
        "| Model | Layers | KV heads | Head dim | Seq len | KV GiB | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['model_id']}`",
                    str(row["layers"]),
                    str(row["num_key_value_heads"]),
                    str(row["head_dim"]),
                    f"{row['seq_len']:,}",
                    f"{row['kv_gib']:.3f}",
                    f"`{row['source']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "These are analytical KV-cache estimates only. They exclude model weights, activations, framework overhead, allocator fragmentation, and paged-attention block overhead.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
