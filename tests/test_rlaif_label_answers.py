from __future__ import annotations

import json
from pathlib import Path

from rag_bench.groq_client import GenerationResult
from rag_bench.rlaif_label_answers import RlaifAnswerLabelConfig, _parse_judge_json, label_rlaif_answers


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


def test_label_answers_dry_run_writes_ambiguous_placeholder(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            dry_run=True,
            limit=1,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["processed_count"] == 1
    assert labels[0]["action_id"] == "a1"
    assert labels[0]["ambiguous"] is True
    assert labels[0]["quality_score"] is None
    assert labels[0]["metadata"]["dry_run"] is True


def test_label_answers_retries_invalid_json_and_parses_valid_response(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])
    client = FakeJudgeClient(
        [
            _generation("not json"),
            _generation(
                json.dumps(
                    {
                        "answer_correctness": 0.8,
                        "evidence_support": 0.7,
                        "unsupported_claim_penalty": 0.1,
                        "refusal_correctness": None,
                        "citation_faithfulness": 0.6,
                        "conciseness": 0.9,
                        "overall_quality": 0.75,
                        "ambiguous": False,
                        "short_rationale": "Mostly supported.",
                    }
                )
            ),
        ]
    )

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            client=client,
            json_retries=1,
        )
    )

    labels = _read_jsonl(output_path)
    assert summary["invalid_json_count"] == 0
    assert len(client.calls) == 2
    assert labels[0]["quality_score"] == 0.75
    assert labels[0]["answer_correctness"] == 0.8
    assert labels[0]["unsupported_claim_penalty"] == 0.1
    assert labels[0]["ambiguous"] is False
    assert labels[0]["metadata"]["json_retry_count"] == 1


def test_label_answers_extracts_json_from_markdown_and_commentary(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])
    client = FakeJudgeClient(
        [
            _generation(
                "Sure.\n```json\n"
                '{"answer_correctness": 0.9, "evidence_support": 0.8, '
                '"unsupported_claim_penalty": 0, "refusal_correctness": null, '
                '"citation_faithfulness": 0.7, "conciseness": 0.6, '
                '"overall_quality": 0.85, "ambiguous": false, '
                '"short_rationale": "Supported."}\n```'
            ),
        ]
    )

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(actions_path=actions_path, output_path=output_path, client=client)
    )

    label = _read_jsonl(output_path)[0]
    assert summary["invalid_json_count"] == 0
    assert label["quality_score"] == 0.85
    assert label["ambiguous"] is False


def test_label_answers_records_generation_metadata_for_final_invalid_json(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1")])
    client = FakeJudgeClient([_generation(""), _generation("")])

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(
            actions_path=actions_path,
            output_path=output_path,
            client=client,
            json_retries=1,
        )
    )

    label = _read_jsonl(output_path)[0]
    assert summary["invalid_json_count"] == 1
    assert label["invalid_json"] is True
    assert label["metadata"]["raw_response_preview"] == ""
    assert label["metadata"]["json_retry_count"] == 1
    assert label["metadata"]["attempt"] == 1


def test_parse_judge_json_accepts_trailing_commas_and_singleton_lists() -> None:
    assert _parse_judge_json('prefix {"overall_quality": 0.7,} suffix') == {"overall_quality": 0.7}
    assert _parse_judge_json('[{"overall_quality": 0.6,}]') == {"overall_quality": 0.6}


def test_label_answers_resume_skips_completed_actions(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1"), _action("a2", "q2")])
    _write_jsonl(output_path, [{"action_id": "a1", "schema_version": "rlaif-answer-label-v1"}])

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(
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


def test_label_answers_missing_context_is_ambiguous_not_zero(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    action = _action("a1", "q1")
    action["retrieved"] = []
    _write_jsonl(actions_path, [action])

    label_rlaif_answers(RlaifAnswerLabelConfig(actions_path=actions_path, output_path=output_path, dry_run=True))

    label = _read_jsonl(output_path)[0]
    assert label["provenance"] == "missing"
    assert label["missing_reason"] == "missing_context"
    assert label["ambiguous"] is True
    assert label["quality_score"] is None


def test_label_answers_stops_after_max_errors(tmp_path: Path) -> None:
    actions_path = tmp_path / "rlaif_actions.jsonl"
    output_path = tmp_path / "labels.jsonl"
    _write_jsonl(actions_path, [_action("a1", "q1"), _action("a2", "q2")])
    client = FakeJudgeClient([_generation("", error="judge unavailable"), _generation("{}")])

    summary = label_rlaif_answers(
        RlaifAnswerLabelConfig(
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
            }
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
