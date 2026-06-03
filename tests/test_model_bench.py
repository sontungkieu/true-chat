from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import CompletedProcess

from rag_bench.model_bench import (
    ModelBenchConfig,
    OpenAICompletionClient,
    Scenario,
    aggregate_scenario,
    collect_hardware_samples,
    collect_hardware_snapshot,
    enrich_scenario_rows_with_hardware,
    parse_nvidia_gpu_samples,
    parse_nvidia_gpu_snapshot,
    run_model_benchmark,
    run_scenario,
    validate_model_bench_config,
)


class _CompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.requests.append(payload)
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"content": "hello "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = json.dumps(
            {
                "choices": [{"message": {"content": "plain response"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        return


def _serve_completion_api() -> tuple[ThreadingHTTPServer, str]:
    _CompletionHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def test_openai_completion_client_reads_streaming_response() -> None:
    server, endpoint = _serve_completion_api()
    try:
        client = OpenAICompletionClient(endpoint=endpoint, model="test-model", stream=True)
        result = client.complete(
            [{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.0,
            request_id="r1",
            scenario="s1",
            suite="synthetic",
        )
    finally:
        server.shutdown()

    assert result.error is None
    assert result.ttft_s is not None
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.total_tokens == 7
    assert result.output_tokens_per_s is not None
    assert _CompletionHandler.requests[0]["model"] == "test-model"
    assert _CompletionHandler.requests[0]["stream_options"] == {"include_usage": True}


def test_openai_completion_client_reads_non_streaming_response() -> None:
    server, endpoint = _serve_completion_api()
    try:
        client = OpenAICompletionClient(endpoint=endpoint, model="test-model", stream=False)
        result = client.complete([{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.0)
    finally:
        server.shutdown()

    assert result.error is None
    assert result.ttft_s is None
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 6
    assert result.total_tokens == 11


def test_run_scenario_aggregates_fake_server_results() -> None:
    server, endpoint = _serve_completion_api()
    try:
        client = OpenAICompletionClient(endpoint=endpoint, model="test-model", stream=False)
        scenario = Scenario(
            name="unit",
            suite="synthetic",
            messages=[{"role": "user", "content": "hello"}],
            max_output_tokens=8,
        )
        result = run_scenario(
            client,
            scenario,
            concurrency=2,
            requests_per_scenario=4,
            max_output_tokens=8,
            temperature=0.0,
        )
    finally:
        server.shutdown()

    assert len(result["requests"]) == 4
    assert result["metrics"]["scenario"] == "unit"
    assert result["metrics"]["concurrency"] == 2
    assert result["metrics"]["success_count"] == 4
    assert result["metrics"]["error_rate"] == 0.0
    assert result["metrics"]["latency_p50_s"] is not None


def test_run_model_benchmark_emits_progress_for_existing_endpoint(tmp_path: Path, capsys) -> None:
    server, endpoint = _serve_completion_api()
    try:
        summary = run_model_benchmark(
            ModelBenchConfig(
                model=None,
                endpoint=endpoint,
                served_model_name="test-model",
                preset="smoke",
                concurrency=(1,),
                requests_per_scenario=1,
                warmup_requests=0,
                output_dir=tmp_path,
                host="127.0.0.1",
                port=8000,
                tensor_parallel_size="1",
                max_model_len=None,
                max_output_tokens=8,
                temperature=0.0,
                startup_timeout_s=1,
                sample_interval_s=0.1,
                stream=False,
            )
        )
    finally:
        server.shutdown()

    captured = capsys.readouterr()
    assert summary["scenario_count"] == 1
    assert summary["request_count"] == 1
    assert "[model-bench " in captured.err
    assert "using existing OpenAI-compatible endpoint" in captured.err
    assert "run 1/1: scenario=synthetic_short concurrency=1 requests=1" in captured.err
    assert "writing benchmark artifacts" in captured.err


def test_aggregate_scenario_records_error_rate_and_percentiles() -> None:
    scenario = Scenario(name="unit", suite="synthetic", messages=[], max_output_tokens=8)
    rows = [
        _row("a", latency_s=1.0, output_tokens_per_s=10.0),
        _row("b", latency_s=2.0, output_tokens_per_s=20.0),
        _row("c", latency_s=3.0, error="failed"),
    ]

    metrics = aggregate_scenario(scenario, rows, concurrency=1, duration_s=6.0)

    assert metrics["request_count"] == 3
    assert metrics["success_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == 1 / 3
    assert metrics["latency_p50_s"] == 1.5
    assert metrics["avg_output_tokens_per_s"] == 15.0


def test_nvidia_parsers_handle_csv_rows() -> None:
    snapshot = parse_nvidia_gpu_snapshot("0, NVIDIA A100-SXM4-40GB, 40960, 550.54\n")
    samples = parse_nvidia_gpu_samples("0, 91, 20480, 40960, 250.5, 67\n")

    assert snapshot == [
        {
            "gpu_index": 0,
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "gpu_memory_total_mb": 40960.0,
            "driver_version": "550.54",
        }
    ]
    assert samples[0]["gpu_util_percent"] == 91.0
    assert samples[0]["gpu_power_w"] == 250.5


def test_enrich_scenario_rows_with_hardware_adds_peak_metrics() -> None:
    scenario_rows = [
        {
            "scenario": "unit",
            "suite": "synthetic",
            "concurrency": 1,
            "scenario_started_at": "2026-01-01T00:00:00+00:00",
            "scenario_ended_at": "2026-01-01T00:00:03+00:00",
        }
    ]
    hardware_rows = [
        {
            "sampled_at": "2026-01-01T00:00:01+00:00",
            "gpu_util_percent": 50.0,
            "gpu_memory_used_mb": 1000.0,
            "gpu_memory_total_mb": 2000.0,
            "gpu_power_w": 80.0,
            "gpu_temperature_c": 55.0,
            "ram_used_mb": 8000.0,
            "ram_total_mb": 16000.0,
            "cpu_load_1m": 1.0,
        },
        {
            "sampled_at": "2026-01-01T00:00:02+00:00",
            "gpu_util_percent": 90.0,
            "gpu_memory_used_mb": 1500.0,
            "gpu_memory_total_mb": 2000.0,
            "gpu_power_w": 120.0,
            "gpu_temperature_c": 65.0,
            "ram_used_mb": 9000.0,
            "ram_total_mb": 16000.0,
            "cpu_load_1m": 2.0,
        },
    ]

    enriched = enrich_scenario_rows_with_hardware(scenario_rows, hardware_rows)

    assert enriched[0]["hardware_sample_count"] == 2
    assert enriched[0]["gpu_peak_memory_used_mb"] == 1500.0
    assert enriched[0]["gpu_peak_memory_used_percent"] == 75.0
    assert enriched[0]["gpu_peak_util_percent"] == 90.0
    assert enriched[0]["gpu_avg_util_percent"] == 70.0
    assert enriched[0]["gpu_peak_power_w"] == 120.0
    assert enriched[0]["gpu_peak_temperature_c"] == 65.0
    assert enriched[0]["ram_peak_used_percent"] == 56.25


def test_hardware_collection_falls_back_without_nvidia_smi() -> None:
    def missing(_command: list[str]) -> CompletedProcess[str]:
        return CompletedProcess(_command, 127, "", "not found")

    snapshot = collect_hardware_snapshot(command_runner=missing)
    samples = collect_hardware_samples(command_runner=missing)

    assert snapshot["nvidia_smi"]["available"] is False
    assert snapshot["gpus"] == []
    assert samples[0]["gpu_index"] is None


def test_model_bench_config_validation_rejects_bad_values(tmp_path: Path) -> None:
    config = ModelBenchConfig(
        model="model",
        endpoint=None,
        served_model_name=None,
        preset="smoke",
        concurrency=(1,),
        requests_per_scenario=0,
        warmup_requests=0,
        output_dir=tmp_path,
        host="127.0.0.1",
        port=8000,
        tensor_parallel_size="auto",
        max_model_len=None,
        max_output_tokens=None,
        temperature=0.0,
        startup_timeout_s=1,
        sample_interval_s=1.0,
        stream=True,
    )

    try:
        validate_model_bench_config(config)
    except ValueError as exc:
        assert "--requests-per-scenario must be positive" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def _row(request_id: str, *, latency_s: float, output_tokens_per_s: float | None = None, error: str | None = None):
    from rag_bench.model_bench import CompletionResult

    return CompletionResult(
        request_id=request_id,
        scenario="unit",
        suite="synthetic",
        concurrency=1,
        started_at="2026-01-01T00:00:00+00:00",
        latency_s=latency_s,
        ttft_s=None,
        prompt_tokens=None,
        completion_tokens=10 if error is None else None,
        total_tokens=None,
        output_tokens_per_s=output_tokens_per_s,
        generated_chars=10,
        error=error,
    )
