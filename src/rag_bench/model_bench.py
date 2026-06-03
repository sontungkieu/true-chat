from __future__ import annotations

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from rag_bench.io import write_csv, write_json, write_jsonl


MODEL_BENCH_PRESETS = ("smoke", "standard", "all")
DEFAULT_MODEL_BENCH_OUTPUT_DIR = Path("runs/model_bench")
ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class ModelBenchConfig:
    model: str | None
    endpoint: str | None
    served_model_name: str | None
    preset: str
    concurrency: tuple[int, ...] | None
    requests_per_scenario: int | None
    warmup_requests: int
    output_dir: Path
    host: str
    port: int
    tensor_parallel_size: str
    max_model_len: int | None
    max_output_tokens: int | None
    temperature: float
    startup_timeout_s: int
    sample_interval_s: float
    stream: bool
    vllm_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    suite: str
    messages: list[dict[str, str]]
    max_output_tokens: int


@dataclass(frozen=True)
class CompletionResult:
    request_id: str
    scenario: str
    suite: str
    concurrency: int
    started_at: str
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    output_tokens_per_s: float | None
    generated_chars: int
    error: str | None


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def progress_logger() -> ProgressLogger:
    def log(message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[model-bench {stamp}] {message}", file=sys.stderr, flush=True)

    return log


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


@dataclass
class HardwareSampler:
    interval_s: float = 1.0
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = field(default=_run_command)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._sample_loop, name="hardware-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 0.5))
        with self._lock:
            return list(self._rows)

    def sample_once(self) -> None:
        rows = collect_hardware_samples(command_runner=self.command_runner)
        with self._lock:
            self._rows.extend(rows)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)


def run_model_benchmark(config: ModelBenchConfig) -> dict[str, Any]:
    validate_model_bench_config(config)
    started = time.perf_counter()
    run_id = build_run_id(config)
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_dir / "server.log"
    progress = progress_logger()
    progress(f"run_id={run_id}")
    progress(f"output_dir={run_dir}")

    served_model_name = config.served_model_name or config.model
    if served_model_name is None:
        raise ValueError("--served-model-name is required when --endpoint is used without --model")

    tensor_parallel_size = resolve_tensor_parallel_size(config.tensor_parallel_size)
    endpoint = config.endpoint.rstrip("/") if config.endpoint else f"http://{config.host}:{config.port}/v1"
    progress(f"model={served_model_name} endpoint={endpoint} preset={config.preset} tensor_parallel_size={tensor_parallel_size}")
    vllm_command: list[str] | None = None
    server_process: subprocess.Popen[str] | None = None
    server_log_handle = None
    progress("collecting hardware snapshot")
    hardware_snapshot = collect_hardware_snapshot()
    sampler = HardwareSampler(interval_s=config.sample_interval_s)
    request_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    try:
        if config.endpoint is None:
            assert config.model is not None
            vllm_command = build_vllm_command(config, tensor_parallel_size=tensor_parallel_size, served_model_name=served_model_name)
            progress(f"starting vLLM: {shell_join(vllm_command)}")
            progress(f"server_log={server_log_path}")
            server_log_handle = server_log_path.open("w", encoding="utf-8", buffering=1)
            server_process = subprocess.Popen(
                vllm_command,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_for_server_health(
                endpoint,
                server_process,
                timeout_s=config.startup_timeout_s,
                log_path=server_log_path,
                progress=progress,
            )
            progress("vLLM health check passed")
        else:
            progress("using existing OpenAI-compatible endpoint; no local vLLM process will be started")
            server_log_path.write_text("Using existing OpenAI-compatible endpoint; no local vLLM process was started.\n", encoding="utf-8")

        scenarios = scenarios_for_preset(config.preset)
        concurrency_values = config.concurrency or default_concurrency(config.preset)
        requests_per_scenario = config.requests_per_scenario or default_requests_per_scenario(config.preset)
        client = OpenAICompletionClient(endpoint=endpoint, model=served_model_name, stream=config.stream)
        progress(
            "benchmark plan: "
            f"{len(scenarios)} scenario(s), concurrency={','.join(str(value) for value in concurrency_values)}, "
            f"requests_per_scenario={requests_per_scenario}, warmup_requests={config.warmup_requests}, "
            f"stream={config.stream}"
        )

        progress(f"starting hardware sampler interval_s={config.sample_interval_s}")
        sampler.start()
        try:
            total_runs = len(scenarios) * len(concurrency_values)
            run_index = 0
            for scenario_index, scenario in enumerate(scenarios, start=1):
                max_output_tokens = config.max_output_tokens or scenario.max_output_tokens
                progress(
                    f"scenario {scenario_index}/{len(scenarios)} {scenario.name} "
                    f"suite={scenario.suite} max_output_tokens={max_output_tokens}"
                )
                for warmup_index in range(config.warmup_requests):
                    progress(f"warmup {warmup_index + 1}/{config.warmup_requests} for {scenario.name}")
                    client.complete(
                        scenario.messages,
                        max_tokens=max_output_tokens,
                        temperature=config.temperature,
                    )
                for concurrency in concurrency_values:
                    run_index += 1
                    progress(
                        f"run {run_index}/{total_runs}: scenario={scenario.name} concurrency={concurrency} "
                        f"requests={requests_per_scenario}"
                    )
                    sampler.sample_once()
                    scenario_started_at = datetime.now(timezone.utc)
                    scenario_result = run_scenario(
                        client,
                        scenario,
                        concurrency=concurrency,
                        requests_per_scenario=requests_per_scenario,
                        max_output_tokens=max_output_tokens,
                        temperature=config.temperature,
                    )
                    scenario_ended_at = datetime.now(timezone.utc)
                    sampler.sample_once()
                    request_rows.extend(row.__dict__ for row in scenario_result["requests"])
                    metrics = scenario_result["metrics"]
                    scenario_rows.append(
                        {
                            **metrics,
                            "scenario_started_at": scenario_started_at.isoformat(),
                            "scenario_ended_at": scenario_ended_at.isoformat(),
                        }
                    )
                    progress(
                        f"done scenario={scenario.name} concurrency={concurrency}: "
                        f"success={metrics['success_count']}/{metrics['request_count']} "
                        f"p50_latency={fmt_number(metrics.get('latency_p50_s'))}s "
                        f"avg_tok_s={fmt_number(metrics.get('avg_output_tokens_per_s'))}"
                    )
        finally:
            hardware_rows = sampler.stop()
            progress(f"hardware sampler stopped; samples={len(hardware_rows)}")

        progress("aggregating hardware metrics by scenario")
        scenario_rows = enrich_scenario_rows_with_hardware(scenario_rows, hardware_rows)

        manifest = build_manifest(
            config,
            run_id=run_id,
            endpoint=endpoint,
            served_model_name=served_model_name,
            tensor_parallel_size=tensor_parallel_size,
            vllm_command=vllm_command,
            hardware_snapshot=hardware_snapshot,
            elapsed_s=time.perf_counter() - started,
        )
        progress("writing benchmark artifacts")
        write_outputs(run_dir, manifest=manifest, request_rows=request_rows, scenario_rows=scenario_rows, hardware_rows=hardware_rows)
        progress(f"done; summary={run_dir / 'summary.md'}")
        return {
            "run_id": run_id,
            "output_dir": str(run_dir),
            "scenario_count": len(scenario_rows),
            "request_count": len(request_rows),
        }
    finally:
        if server_process is not None:
            progress("stopping local vLLM process")
            terminate_process(server_process)
        if server_log_handle is not None:
            server_log_handle.close()


def validate_model_bench_config(config: ModelBenchConfig) -> None:
    if config.preset not in MODEL_BENCH_PRESETS:
        raise ValueError(f"--preset must be one of: {', '.join(MODEL_BENCH_PRESETS)}")
    if config.endpoint is None and not config.model:
        raise ValueError("--model is required when starting vLLM locally")
    if config.endpoint is not None and not (config.served_model_name or config.model):
        raise ValueError("--served-model-name or --model is required with --endpoint")
    if config.concurrency is not None and any(value <= 0 for value in config.concurrency):
        raise ValueError("--concurrency values must be positive")
    if config.requests_per_scenario is not None and config.requests_per_scenario <= 0:
        raise ValueError("--requests-per-scenario must be positive")
    if config.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    if config.port <= 0:
        raise ValueError("--port must be positive")
    if config.max_model_len is not None and config.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if config.max_output_tokens is not None and config.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if config.startup_timeout_s <= 0:
        raise ValueError("--startup-timeout-s must be positive")
    if config.sample_interval_s <= 0:
        raise ValueError("--sample-interval-s must be positive")
    if config.tensor_parallel_size != "auto":
        try:
            parsed = int(config.tensor_parallel_size)
        except ValueError as exc:
            raise ValueError("--tensor-parallel-size must be 'auto' or a positive integer") from exc
        if parsed <= 0:
            raise ValueError("--tensor-parallel-size must be 'auto' or a positive integer")


def build_vllm_command(
    config: ModelBenchConfig,
    *,
    tensor_parallel_size: int,
    served_model_name: str,
) -> list[str]:
    assert config.model is not None
    command = [
        "vllm",
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    command.extend(config.vllm_args)
    return command


def wait_for_server_health(
    endpoint: str,
    process: subprocess.Popen[str],
    *,
    timeout_s: int,
    log_path: Path,
    progress: ProgressLogger | None = None,
) -> None:
    health_url = endpoint.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url = f"{health_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    next_progress_at = started
    last_error: str | None = None
    if progress is not None:
        progress(f"waiting for vLLM health at {health_url} timeout_s={timeout_s}")
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"vLLM exited before health check passed with code {exit_code}: {tail_file(log_path)}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001 - health endpoint can fail while vLLM loads.
            last_error = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if progress is not None and now >= next_progress_at:
            elapsed = int(now - started)
            remaining = max(0, int(deadline - now))
            progress(f"vLLM still starting: elapsed={elapsed}s remaining={remaining}s last_error={last_error}")
            next_progress_at = now + 30
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s; last error: {last_error}; log: {tail_file(log_path)}")


class OpenAICompletionClient:
    def __init__(self, *, endpoint: str, model: str, stream: bool = True, timeout_s: float = 600.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.stream = stream
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        request_id: str = "warmup",
        scenario: str = "warmup",
        suite: str = "warmup",
        concurrency: int = 1,
    ) -> CompletionResult:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.stream,
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        try:
            response = self._post(payload)
            if self.stream:
                text, ttft_s, usage = self._read_stream(response, started=started)
            else:
                text, ttft_s, usage = self._read_json(response)
            latency_s = time.perf_counter() - started
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                completion_tokens = estimate_text_tokens(text)
            return CompletionResult(
                request_id=request_id,
                scenario=scenario,
                suite=suite,
                concurrency=concurrency,
                started_at=started_at,
                latency_s=latency_s,
                ttft_s=ttft_s,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens"),
                output_tokens_per_s=tokens_per_second(completion_tokens, latency_s),
                generated_chars=len(text),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark rows should capture provider/server errors.
            latency_s = time.perf_counter() - started
            return CompletionResult(
                request_id=request_id,
                scenario=scenario,
                suite=suite,
                concurrency=concurrency,
                started_at=started_at,
                latency_s=latency_s,
                ttft_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                output_tokens_per_s=None,
                generated_chars=0,
                error=safe_error(exc),
            )

    def _post(self, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=self.timeout_s)

    def _read_json(self, response: Any) -> tuple[str, None, dict[str, int | None]]:
        parsed = json.loads(response.read().decode("utf-8"))
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                text = content
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        return text, None, normalize_usage(usage)

    def _read_stream(self, response: Any, *, started: float) -> tuple[str, float | None, dict[str, int | None]]:
        parts: list[str] = []
        ttft_s: float | None = None
        usage: dict[str, int | None] = {}
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            parsed = json.loads(data)
            if isinstance(parsed.get("usage"), dict):
                usage.update(normalize_usage(parsed["usage"]))
            choices = parsed.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - started
                parts.append(content)
        return "".join(parts), ttft_s, usage


def run_scenario(
    client: OpenAICompletionClient,
    scenario: Scenario,
    *,
    concurrency: int,
    requests_per_scenario: int,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[CompletionResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                client.complete,
                scenario.messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                request_id=f"{scenario.name}-{concurrency}-{index + 1}",
                scenario=scenario.name,
                suite=scenario.suite,
                concurrency=concurrency,
            )
            for index in range(requests_per_scenario)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    duration_s = time.perf_counter() - started
    return {
        "requests": sorted(results, key=lambda item: item.request_id),
        "metrics": aggregate_scenario(scenario, results, concurrency=concurrency, duration_s=duration_s),
    }


def aggregate_scenario(
    scenario: Scenario,
    rows: list[CompletionResult],
    *,
    concurrency: int,
    duration_s: float,
) -> dict[str, Any]:
    successes = [row for row in rows if row.error is None]
    latencies = [row.latency_s for row in successes]
    ttfts = [row.ttft_s for row in successes if row.ttft_s is not None]
    throughputs = [row.output_tokens_per_s for row in successes if row.output_tokens_per_s is not None]
    completion_tokens = [row.completion_tokens for row in successes if row.completion_tokens is not None]
    return {
        "scenario": scenario.name,
        "suite": scenario.suite,
        "concurrency": concurrency,
        "request_count": len(rows),
        "success_count": len(successes),
        "error_count": len(rows) - len(successes),
        "error_rate": safe_ratio(len(rows) - len(successes), len(rows)),
        "duration_s": duration_s,
        "requests_per_s": safe_ratio(len(rows), duration_s),
        "completion_tokens_per_s": safe_ratio(sum(completion_tokens), duration_s),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "avg_output_tokens_per_s": mean(throughputs) if throughputs else None,
        "max_output_tokens_per_s": max(throughputs) if throughputs else None,
    }


def enrich_scenario_rows_with_hardware(
    scenario_rows: list[dict[str, Any]],
    hardware_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            **aggregate_hardware_for_window(
                hardware_rows,
                started_at=row.get("scenario_started_at"),
                ended_at=row.get("scenario_ended_at"),
            ),
        }
        for row in scenario_rows
    ]


def aggregate_hardware_for_window(
    hardware_rows: list[dict[str, Any]],
    *,
    started_at: str | None,
    ended_at: str | None,
) -> dict[str, Any]:
    window_rows = hardware_rows_in_window(hardware_rows, started_at=started_at, ended_at=ended_at)
    if not window_rows:
        return {
            "hardware_sample_count": 0,
            "gpu_peak_memory_used_mb": None,
            "gpu_peak_memory_used_percent": None,
            "gpu_peak_util_percent": None,
            "gpu_avg_util_percent": None,
            "gpu_peak_power_w": None,
            "gpu_avg_power_w": None,
            "gpu_peak_temperature_c": None,
            "ram_peak_used_mb": None,
            "ram_peak_used_percent": None,
            "cpu_load_1m_peak": None,
        }

    gpu_memory_used = numeric_values(window_rows, "gpu_memory_used_mb")
    gpu_memory_total = numeric_values(window_rows, "gpu_memory_total_mb")
    gpu_util = numeric_values(window_rows, "gpu_util_percent")
    gpu_power = numeric_values(window_rows, "gpu_power_w")
    gpu_temperature = numeric_values(window_rows, "gpu_temperature_c")
    ram_used = numeric_values(window_rows, "ram_used_mb")
    ram_total = numeric_values(window_rows, "ram_total_mb")
    cpu_load = numeric_values(window_rows, "cpu_load_1m")

    peak_gpu_memory_used = max(gpu_memory_used) if gpu_memory_used else None
    peak_gpu_memory_total = max(gpu_memory_total) if gpu_memory_total else None
    peak_ram_used = max(ram_used) if ram_used else None
    peak_ram_total = max(ram_total) if ram_total else None

    return {
        "hardware_sample_count": len(window_rows),
        "gpu_peak_memory_used_mb": peak_gpu_memory_used,
        "gpu_peak_memory_used_percent": safe_ratio(100 * peak_gpu_memory_used, peak_gpu_memory_total)
        if peak_gpu_memory_used is not None and peak_gpu_memory_total
        else None,
        "gpu_peak_util_percent": max(gpu_util) if gpu_util else None,
        "gpu_avg_util_percent": mean(gpu_util) if gpu_util else None,
        "gpu_peak_power_w": max(gpu_power) if gpu_power else None,
        "gpu_avg_power_w": mean(gpu_power) if gpu_power else None,
        "gpu_peak_temperature_c": max(gpu_temperature) if gpu_temperature else None,
        "ram_peak_used_mb": peak_ram_used,
        "ram_peak_used_percent": safe_ratio(100 * peak_ram_used, peak_ram_total)
        if peak_ram_used is not None and peak_ram_total
        else None,
        "cpu_load_1m_peak": max(cpu_load) if cpu_load else None,
    }


def hardware_rows_in_window(
    hardware_rows: list[dict[str, Any]],
    *,
    started_at: str | None,
    ended_at: str | None,
) -> list[dict[str, Any]]:
    if not started_at or not ended_at:
        return hardware_rows
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
    except ValueError:
        return hardware_rows
    rows = []
    for row in hardware_rows:
        sampled_at = row.get("sampled_at")
        if not isinstance(sampled_at, str):
            continue
        try:
            sampled = datetime.fromisoformat(sampled_at)
        except ValueError:
            continue
        if start <= sampled <= end:
            rows.append(row)
    return rows or hardware_rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def scenarios_for_preset(preset: str) -> list[Scenario]:
    synthetic = [
        synthetic_scenario("synthetic_short", prompt_target_tokens=96, max_output_tokens=64),
        synthetic_scenario("synthetic_medium", prompt_target_tokens=768, max_output_tokens=128),
        synthetic_scenario("synthetic_long", prompt_target_tokens=3000, max_output_tokens=256),
    ]
    if preset == "smoke":
        return synthetic[:1]
    if preset == "standard":
        return synthetic
    return [
        *synthetic,
        synthetic_scenario("synthetic_long_8k", prompt_target_tokens=7600, max_output_tokens=256),
        *chat_scenarios(),
    ]


def synthetic_scenario(name: str, *, prompt_target_tokens: int, max_output_tokens: int) -> Scenario:
    unit = (
        "Benchmark context block: measure inference throughput, latency, and stable instruction following. "
        "Return concise factual prose without lists unless requested. "
    )
    repetitions = max(1, prompt_target_tokens // max(1, estimate_text_tokens(unit)))
    content = (unit * repetitions).strip()
    return Scenario(
        name=name,
        suite="synthetic",
        messages=[
            {"role": "system", "content": "You are a deterministic benchmark assistant. Answer briefly and directly."},
            {"role": "user", "content": f"{content}\n\nSummarize the benchmark context in one paragraph."},
        ],
        max_output_tokens=max_output_tokens,
    )


def chat_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="chat_quick_qa",
            suite="chat",
            messages=[{"role": "user", "content": "Give three practical checks before deploying a local LLM server."}],
            max_output_tokens=96,
        ),
        Scenario(
            name="chat_vietnamese_explain",
            suite="chat",
            messages=[{"role": "user", "content": "Giải thích ngắn gọn vì sao cần đo TTFT và tok/s khi benchmark mô hình."}],
            max_output_tokens=128,
        ),
        Scenario(
            name="chat_summary",
            suite="chat",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize this deployment note for an engineer: start vLLM, verify health, "
                        "run warmup requests, measure latency percentiles, record GPU memory and power, "
                        "then save reproducible artifacts for comparison."
                    ),
                }
            ],
            max_output_tokens=160,
        ),
        Scenario(
            name="chat_multiturn",
            suite="chat",
            messages=[
                {"role": "user", "content": "We need to compare two GPU machines for one model."},
                {"role": "assistant", "content": "Use identical prompts, concurrency, output lengths, and collect hardware samples."},
                {"role": "user", "content": "Now give the final checklist in Vietnamese."},
            ],
            max_output_tokens=160,
        ),
    ]


def default_concurrency(preset: str) -> tuple[int, ...]:
    if preset == "smoke":
        return (1,)
    return (1, 2, 4, 8)


def default_requests_per_scenario(preset: str) -> int:
    return 2 if preset == "smoke" else 6


def collect_hardware_snapshot(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),
        "ram_total_mb": ram_total_mb(),
        "nvidia_smi": nvidia_smi_overview(command_runner=command_runner),
        "gpus": nvidia_gpu_snapshot(command_runner=command_runner),
    }


def collect_hardware_samples(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> list[dict[str, Any]]:
    sampled_at = datetime.now(timezone.utc).isoformat()
    base = {
        "sampled_at": sampled_at,
        "cpu_load_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        "ram_used_mb": ram_used_mb(),
        "ram_total_mb": ram_total_mb(),
    }
    gpu_rows = nvidia_gpu_samples(command_runner=command_runner)
    if not gpu_rows:
        return [{**base, "gpu_index": None}]
    return [{**base, **row} for row in gpu_rows]


def nvidia_smi_overview(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> dict[str, Any]:
    result = command_runner(["nvidia-smi"])
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or result.stdout.strip()}
    text = result.stdout
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
    return {
        "available": True,
        "driver_version": driver_match.group(1) if driver_match else None,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
    }


def nvidia_gpu_snapshot(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> list[dict[str, Any]]:
    result = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    return parse_nvidia_gpu_snapshot(result.stdout)


def nvidia_gpu_samples(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> list[dict[str, Any]]:
    result = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []
    return parse_nvidia_gpu_samples(result.stdout)


def parse_nvidia_gpu_snapshot(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "gpu_index": to_int(parts[0]),
                "gpu_name": parts[1],
                "gpu_memory_total_mb": to_float(parts[2]),
                "driver_version": parts[3],
            }
        )
    return rows


def parse_nvidia_gpu_samples(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "gpu_index": to_int(parts[0]),
                "gpu_util_percent": to_float(parts[1]),
                "gpu_memory_used_mb": to_float(parts[2]),
                "gpu_memory_total_mb": to_float(parts[3]),
                "gpu_power_w": to_float(parts[4]),
                "gpu_temperature_c": to_float(parts[5]),
            }
        )
    return rows


def resolve_tensor_parallel_size(value: str) -> int:
    if value != "auto":
        return int(value)
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in {"", "-1"}:
        return max(1, len([item for item in visible.split(",") if item.strip()]))
    gpu_count = len(nvidia_gpu_snapshot())
    return max(1, gpu_count)


def build_manifest(
    config: ModelBenchConfig,
    *,
    run_id: str,
    endpoint: str,
    served_model_name: str,
    tensor_parallel_size: int,
    vllm_command: list[str] | None,
    hardware_snapshot: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "config": serializable_config(config),
        "endpoint": endpoint,
        "served_model_name": served_model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "vllm_command": vllm_command,
        "inference_engine": build_inference_engine_metadata(
            config,
            endpoint=endpoint,
            vllm_command=vllm_command,
        ),
        "serving_config": build_serving_config_metadata(
            config,
            tensor_parallel_size=tensor_parallel_size,
            served_model_name=served_model_name,
        ),
        "git": git_metadata(),
        "hardware": hardware_snapshot,
    }


def build_inference_engine_metadata(
    config: ModelBenchConfig,
    *,
    endpoint: str,
    vllm_command: list[str] | None,
) -> dict[str, Any]:
    local_vllm = vllm_command is not None
    return {
        "name": "vllm" if local_vllm else "openai-compatible",
        "version": package_version("vllm") if local_vllm else None,
        "mode": "local_vllm" if local_vllm else "external_endpoint",
        "endpoint": endpoint,
        "openai_compatible": True,
        "streaming": config.stream,
        "command": vllm_command,
    }


def build_serving_config_metadata(
    config: ModelBenchConfig,
    *,
    tensor_parallel_size: int,
    served_model_name: str,
) -> dict[str, Any]:
    vllm_options = parse_vllm_options(config.vllm_args)
    model_hint = model_quantization_hint(config.model or served_model_name)
    quantization_arg = vllm_options.get("quantization")
    quantization = quantization_arg if quantization_arg not in {None, "", "auto"} else "auto/model-config"
    return {
        "served_model_name": served_model_name,
        "model": config.model,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": config.max_model_len,
        "max_output_tokens_override": config.max_output_tokens,
        "temperature": config.temperature,
        "vllm_args": list(config.vllm_args),
        "vllm_options": vllm_options,
        "quantization": {
            "argument": quantization_arg,
            "effective": quantization,
            "model_hint": model_hint,
        },
        "dtype": vllm_options.get("dtype", "auto"),
        "kv_cache_dtype": vllm_options.get("kv_cache_dtype", "auto"),
        "attention_backend": vllm_options.get("attention_backend", "auto"),
        "speculative_decoding": speculative_decoding_metadata(vllm_options),
        "cpu_offload_gb": vllm_options.get("cpu_offload_gb", 0),
        "gpu_memory_utilization": vllm_options.get("gpu_memory_utilization"),
        "max_num_seqs": vllm_options.get("max_num_seqs"),
        "max_num_batched_tokens": vllm_options.get("max_num_batched_tokens"),
        "enforce_eager": bool(vllm_options.get("enforce_eager", False)),
    }


def parse_vllm_options(args: tuple[str, ...] | list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if not arg.startswith("--") or arg == "--":
            index += 1
            continue
        key_value = arg[2:]
        if "=" in key_value:
            key, raw_value = key_value.split("=", 1)
            value: Any = parse_scalar(raw_value)
        elif index + 1 < len(args) and not args[index + 1].startswith("--"):
            key = key_value
            value = parse_scalar(args[index + 1])
            index += 1
        else:
            key = key_value
            value = True
        options[normalize_cli_key(key)] = value
        index += 1
    return options


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[-+]?\d+\.\d+", stripped):
            return float(stripped)
    except ValueError:
        pass
    return value


def normalize_cli_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def speculative_decoding_metadata(vllm_options: dict[str, Any]) -> dict[str, Any]:
    raw_config = vllm_options.get("speculative_config")
    if raw_config is None or raw_config is False or raw_config == "" or (
        isinstance(raw_config, str) and raw_config.lower() in {"off", "none"}
    ):
        return {
            "enabled": False,
            "method": "none",
            "config": None,
            "raw": raw_config,
        }

    parsed_config: Any = raw_config
    if isinstance(raw_config, str):
        try:
            parsed_config = json.loads(raw_config)
        except json.JSONDecodeError:
            parsed_config = raw_config

    method = "configured"
    num_speculative_tokens = None
    draft_model = None
    if isinstance(parsed_config, dict):
        method = str(parsed_config.get("method") or parsed_config.get("type") or method)
        num_speculative_tokens = parsed_config.get("num_speculative_tokens")
        draft_model = parsed_config.get("model") or parsed_config.get("draft_model")

    return {
        "enabled": True,
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_model": draft_model,
        "config": parsed_config,
        "raw": raw_config,
    }


def model_quantization_hint(model_name: str | None) -> str | None:
    if not model_name:
        return None
    lowered = model_name.lower()
    if "awq" in lowered:
        return "awq"
    if "gptq" in lowered:
        return "gptq"
    if "int8" in lowered or "8bit" in lowered or "8-bit" in lowered:
        return "int8"
    if "4bit" in lowered or "4-bit" in lowered:
        return "4bit"
    return None


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def write_outputs(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    request_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    hardware_rows: list[dict[str, Any]],
) -> None:
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "requests.jsonl", request_rows)
    write_json(run_dir / "scenario_metrics.json", {"metrics": scenario_rows})
    write_csv(run_dir / "scenario_metrics.csv", scenario_rows)
    write_csv(run_dir / "hardware_samples.csv", hardware_rows)
    (run_dir / "summary.md").write_text(render_summary(manifest, scenario_rows), encoding="utf-8")


def render_summary(manifest: dict[str, Any], scenario_rows: list[dict[str, Any]]) -> str:
    engine = manifest.get("inference_engine", {})
    serving = manifest.get("serving_config", {})
    quantization = serving.get("quantization", {})
    speculative = serving.get("speculative_decoding", {})
    lines = [
        f"# Model Benchmark {manifest['run_id']}",
        "",
        f"- Model: `{manifest['served_model_name']}`",
        f"- Endpoint: `{manifest['endpoint']}`",
        f"- Host: `{manifest['hardware'].get('hostname')}`",
        f"- Git: `{manifest['git'].get('commit')}` dirty={manifest['git'].get('dirty')}",
        f"- Inference engine: `{engine.get('name')}` version=`{engine.get('version') or 'unknown'}` mode=`{engine.get('mode')}`",
        f"- Streaming: `{engine.get('streaming')}`",
        f"- Quantization: `{quantization.get('effective')}` model_hint=`{quantization.get('model_hint') or 'none'}` argument=`{quantization.get('argument') or 'auto'}`",
        f"- dtype: `{serving.get('dtype')}` kv_cache_dtype=`{serving.get('kv_cache_dtype')}` attention_backend=`{serving.get('attention_backend')}`",
        f"- Speculative decoding: enabled=`{speculative.get('enabled')}` method=`{speculative.get('method')}` num_tokens=`{speculative.get('num_speculative_tokens') or 'n/a'}` draft_model=`{speculative.get('draft_model') or 'n/a'}`",
        f"- vLLM execution: tensor_parallel_size=`{serving.get('tensor_parallel_size')}` max_model_len=`{serving.get('max_model_len')}` max_num_seqs=`{serving.get('max_num_seqs') or 'auto'}` max_num_batched_tokens=`{serving.get('max_num_batched_tokens') or 'auto'}` enforce_eager=`{serving.get('enforce_eager')}`",
        f"- GPU memory utilization: `{serving.get('gpu_memory_utilization') or 'auto'}` CPU offload GB: `{serving.get('cpu_offload_gb') or 0}`",
        "",
        "| Scenario | Suite | Concurrency | Success | Error % | p50 latency | p95 latency | p50 TTFT | avg tok/s | req/s | peak VRAM MB | peak GPU % | peak W | peak C |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["suite"]),
                    str(row["concurrency"]),
                    str(row["success_count"]),
                    fmt_number(100 * float(row["error_rate"] or 0.0)),
                    fmt_number(row.get("latency_p50_s")),
                    fmt_number(row.get("latency_p95_s")),
                    fmt_number(row.get("ttft_p50_s")),
                    fmt_number(row.get("avg_output_tokens_per_s")),
                    fmt_number(row.get("requests_per_s")),
                    fmt_number(row.get("gpu_peak_memory_used_mb")),
                    fmt_number(row.get("gpu_peak_util_percent")),
                    fmt_number(row.get("gpu_peak_power_w")),
                    fmt_number(row.get("gpu_peak_temperature_c")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_run_id(config: ModelBenchConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hostname = slugify(socket.gethostname())
    model_name = config.served_model_name or config.model or "endpoint"
    return f"{stamp}_{hostname}_{slugify(model_name)}"


def git_metadata() -> dict[str, Any]:
    return {
        "branch": git_output(["git", "branch", "--show-current"]),
        "commit": git_output(["git", "rev-parse", "HEAD"]),
        "dirty": bool(git_output(["git", "status", "--short"])),
    }


def git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def serializable_config(config: ModelBenchConfig) -> dict[str, Any]:
    output = dict(config.__dict__)
    output["output_dir"] = str(config.output_dir)
    output["concurrency"] = list(config.concurrency) if config.concurrency is not None else None
    output["vllm_args"] = list(config.vllm_args)
    return output


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def tail_file(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return "(log file does not exist)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def ram_total_mb() -> float | None:
    info = meminfo()
    total_kb = info.get("MemTotal")
    return total_kb / 1024 if total_kb is not None else None


def ram_used_mb() -> float | None:
    info = meminfo()
    total_kb = info.get("MemTotal")
    available_kb = info.get("MemAvailable")
    if total_kb is None or available_kb is None:
        return None
    return (total_kb - available_kb) / 1024


def meminfo() -> dict[str, float]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        match = re.search(r"([0-9.]+)", value)
        if match:
            values[key] = float(match.group(1))
    return values


def normalize_usage(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return {}
    return {
        "prompt_tokens": to_int(value.get("prompt_tokens")),
        "completion_tokens": to_int(value.get("completion_tokens")),
        "total_tokens": to_int(value.get("total_tokens")),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def tokens_per_second(tokens: int | None, latency_s: float) -> float | None:
    if tokens is None or tokens <= 0 or latency_s <= 0:
        return None
    return tokens / latency_s


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def to_int(value: Any) -> int | None:
    try:
        if value in {None, "", "N/A"}:
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        if value in {None, "", "N/A"}:
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def safe_error(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort error detail.
            text = str(exc)
        return f"HTTP {exc.code}: {text[:1000]}"
    return f"{type(exc).__name__}: {text[:1000]}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()
    return slug[:80] or "model"


def fmt_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4g}"
