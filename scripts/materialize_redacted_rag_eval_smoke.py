from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rag_bench.dictionary import DictionaryEntry, load_dictionary_artifact
from rag_bench.privacy import normalize_data_tier


PLACEHOLDER_BY_SLOT = {
    "ALPHA": ("TERM_ALPHA", "DICT_ALPHA"),
    "BETA": ("TERM_BETA", "DICT_BETA"),
    "GAMMA": ("TERM_GAMMA", "DICT_GAMMA"),
    "UNRELATED": ("TERM_UNRELATED", "DICT_UNRELATED"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize redacted RAG smoke templates against a local dictionary artifact.")
    parser.add_argument("--dictionary-artifact", type=Path, required=True)
    parser.add_argument("--eval-template", type=Path, required=True)
    parser.add_argument("--structured-template", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--structured-output", type=Path, required=True)
    parser.add_argument("--data-tier", default="semi_private", choices=("public", "semi_private", "private"))
    args = parser.parse_args()

    entries = _select_entries(load_dictionary_artifact(args.dictionary_artifact), count=len(PLACEHOLDER_BY_SLOT))
    replacements = _replacement_map(entries)
    materialize_jsonl(args.eval_template, args.eval_output, replacements=replacements, data_tier=args.data_tier)
    materialize_jsonl(args.structured_template, args.structured_output, replacements=replacements, data_tier=args.data_tier)
    manifest = {
        "dictionary_artifact": str(args.dictionary_artifact),
        "data_tier": normalize_data_tier(args.data_tier).value,
        "eval_output": str(args.eval_output),
        "structured_output": str(args.structured_output),
        "selected_entry_count": len(entries),
        "selected_entry_ids": [entry.id for entry in entries],
    }
    manifest_path = args.eval_output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eval_output": str(args.eval_output), "structured_output": str(args.structured_output)}, indent=2))
    return 0


def materialize_jsonl(path: Path, output_path: Path, *, replacements: dict[str, str], data_tier: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            rows.append(json.dumps(_replace_value(row, replacements, data_tier=data_tier), ensure_ascii=False))
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _select_entries(entries: list[DictionaryEntry], *, count: int) -> list[DictionaryEntry]:
    selected: list[DictionaryEntry] = []
    seen_headwords: set[str] = set()
    for entry in entries:
        headword = _safe_headword(entry.headword)
        if not headword:
            continue
        key = _normalize_key(headword)
        if key in seen_headwords:
            continue
        seen_headwords.add(key)
        selected.append(entry)
        if len(selected) >= count:
            return selected
    raise ValueError(f"dictionary artifact needs at least {count} usable entries for smoke materialization")


def _replacement_map(entries: list[DictionaryEntry]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for entry, (_slot, (term_placeholder, id_placeholder)) in zip(entries, PLACEHOLDER_BY_SLOT.items(), strict=True):
        replacements[term_placeholder] = _safe_headword(entry.headword)
        replacements[id_placeholder] = entry.id
    return replacements


def _replace_value(value: Any, replacements: dict[str, str], *, data_tier: str) -> Any:
    if isinstance(value, str):
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, replacement)
        return result
    if isinstance(value, list):
        return [_replace_value(item, replacements, data_tier=data_tier) for item in value]
    if isinstance(value, dict):
        replaced = {
            key: _replace_value(item, replacements, data_tier=data_tier)
            for key, item in value.items()
        }
        if "data_tier" in replaced:
            replaced["data_tier"] = normalize_data_tier(data_tier).value
        return replaced
    return value


def _safe_headword(value: str) -> str:
    text = " ".join(str(value or "").split())
    if not text or len(text) > 72:
        return ""
    if not re.search(r"[\wĐđ]", text, flags=re.UNICODE):
        return ""
    return text


def _normalize_key(value: str) -> str:
    return re.sub(r"\W+", "", value.lower(), flags=re.UNICODE)


if __name__ == "__main__":
    raise SystemExit(main())
