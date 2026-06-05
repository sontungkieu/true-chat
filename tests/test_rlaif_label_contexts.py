from __future__ import annotations

import json
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.rlaif_label_contexts import RlaifContextLabelConfig, label_rlaif_contexts


class FakeJudgeClient:
    def __init__(self, responses: list[GenerationResult]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ) -> GenerationResult:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No fake response configured")
        return self.responses.pop(0)


def test_label_contexts_dry_run_writes_ambiguous_placeholder(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])

    summary = label_rlaif_contexts(
        RlaifContextLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            dry_run=True,
            limit=1,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["processed_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert labels[0]["schema_version"] == "rlaif-context-label-v1"
    assert labels[0]["action_id"] == "a1"
    assert labels[0]["ambiguous"] is True
    assert labels[0]["context_quality_score"] is None
    assert labels[0]["available_chunk_ids"] == ["doc-1", "doc-2"]
    assert labels[0]["metadata"]["dry_run"] is True


def test_label_contexts_retries_invalid_json_and_filters_unknown_chunk_ids(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])
    client = FakeJudgeClient(
        [
            _generation("not json"),
            _generation(
                json.dumps(
                    {
                        "sufficient": True,
                        "selected_chunk_ids": ["doc-1", "missing-doc"],
                        "redundant_chunk_ids": ["doc-2"],
                        "irrelevant_chunk_ids": ["doc-2", "doc-2"],
                        "missing_evidence": False,
                        "minimality_score": 0.6,
                        "evidence_support_score": 0.9,
                        "context_quality_score": 0.8,
                        "ambiguous": False,
                        "short_rationale": "Evidence is present but one chunk is redundant.",
                    }
                )
            ),
        ]
    )

    summary = label_rlaif_contexts(
        RlaifContextLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            client=client,
            json_retries=1,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert summary["invalid_json_count"] == 0
    assert len(client.calls) == 2
    assert label["sufficient"] is True
    assert label["missing_evidence"] is False
    assert label["selected_chunk_ids"] == ["doc-1"]
    assert label["redundant_chunk_ids"] == ["doc-2"]
    assert label["irrelevant_chunk_ids"] == ["doc-2"]
    assert label["context_quality_score"] == 0.8
    assert label["ambiguous"] is False
    assert label["metadata"]["json_retry_count"] == 1
    assert label["metadata"]["dropped_unknown_chunk_ids"]["selected_chunk_ids"] == ["missing-doc"]


def test_label_contexts_missing_context_is_ambiguous_not_zero(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    action = _action("a1", "q1")
    action["retrieved"] = []
    _write_jsonl(actions_path, [action])

    label_rlaif_contexts(RlaifContextLabelConfig(actions_path=actions_path, output_path=output_path, dry_run=True))

    label = _read_jsonl(output_path)[0]
    assert label["provenance"] == "missing"
    assert label["missing_reason"] == "missing_context"
    assert label["ambiguous"] is True
    assert label["context_quality_score"] is None
    assert label["selected_chunk_ids"] == []


def test_label_contexts_records_final_invalid_json(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])
    client = FakeJudgeClient([_generation(""), _generation("")])

    summary = label_rlaif_contexts(
        RlaifContextLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            client=client,
            json_retries=1,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert summary["invalid_json_count"] == 1
    assert label["invalid_json"] is True
    assert label["context_quality_score"] is None
    assert label["metadata"]["attempt"] == 1


def test_label_contexts_resume_skips_completed_actions(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1"), _action("a2", "q2")])
    _write_jsonl(output_path, [{"action_id": "a1", "schema_version": "rlaif-context-label-v1"}])

    summary = label_rlaif_contexts(
        RlaifContextLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            dry_run=True,
            resume=True,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["processed_count"] == 1
    assert summary["skipped_resume_count"] == 1
    assert [row["action_id"] for row in labels] == ["a1", "a2"]


def test_label_contexts_stops_after_max_errors(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "context_labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1"), _action("a2", "q2")])
    client = FakeJudgeClient([_generation("", error="judge unavailable"), _generation("{}")])

    summary = label_rlaif_contexts(
        RlaifContextLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            client=client,
            max_errors=1,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "max_errors"
    assert summary["processed_count"] == 1
    assert labels[0]["error"] == "judge unavailable"


def _action(action_id: str, query_id: str) -> dict[str, object]:
    return {
        "action_id": action_id,
        "query_id": query_id,
        "benchmark": "scifact",
        "question": "Does the claim follow from the context?",
        "answer": "Yes, the answer is supported by the context.",
        "retrieved": [
            {
                "rank": 1,
                "doc_id": "doc-1",
                "title": "Evidence",
                "text": "The context states that the answer is supported.",
            },
            {
                "rank": 2,
                "doc_id": "doc-2",
                "title": "Duplicate",
                "text": "This chunk repeats the same support.",
            },
        ],
    }


def _generation(answer: str, *, error: str | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        key_alias="fake",
        attempted_aliases=["fake"],
        latency_s=0.01,
        retry_count=0,
        error=error,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
