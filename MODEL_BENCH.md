# Hướng Dẫn Benchmark Model vLLM

File này hướng dẫn chạy benchmark một model trên từng máy theo workflow thủ công: mở máy, clone repo, setup môi trường, nhập tên model, chạy benchmark, rồi so sánh kết quả giữa các máy.

## 1. Chuẩn bị trên mỗi máy

Clone repo và chuyển sang branch benchmark:

```bash
git clone <repo-url> true-chat
cd true-chat
git checkout bench/vllm-model-bench
```

Setup Python env và cài vLLM:

```bash
scripts/setup_vllm_bench.sh
```

Script setup chỉ cài dependency Python và vLLM trong `.venv`. Script không cài hoặc sửa driver NVIDIA, CUDA, package hệ thống, hay cấu hình GPU. Nếu máy cần version vLLM cụ thể:

```bash
VLLM_VERSION=0.21.0 scripts/setup_vllm_bench.sh
```

Kiểm tra nhanh GPU trước khi chạy:

```bash
nvidia-smi
```

Nếu `nvidia-smi` không chạy được, benchmark vẫn có thể gọi lệnh nhưng vLLM GPU gần như chắc chắn sẽ lỗi cho tới khi driver/CUDA được chuẩn bị đúng.

## 2. Chạy benchmark

### Smoke test

Dùng để kiểm tra model có load được và server vLLM có trả lời được không:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset smoke \
  --tensor-parallel-size auto
```

### Benchmark chuẩn

Dùng để so sánh nhanh giữa nhiều máy, gồm synthetic short/medium/long với concurrency `1,2,4,8`:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset standard \
  --tensor-parallel-size auto
```

### Benchmark đầy đủ

Dùng khi máy ổn định và model fit VRAM. Suite này thêm workload long-context khoảng 8k và prompt chat thực tế, gồm tiếng Việt và multi-turn:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset all \
  --tensor-parallel-size auto \
  --max-model-len 8192
```

Mặc định lệnh sẽ:

- start `vllm serve` cục bộ;
- chờ endpoint `/health`;
- chạy warmup;
- chạy benchmark;
- lấy mẫu phần cứng trong khi chạy;
- ghi artifacts vào `runs/model_bench/...`;
- stop process vLLM khi xong.

## 3. Dùng endpoint có sẵn

Nếu bạn đã tự start vLLM hoặc có OpenAI-compatible endpoint khác:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --endpoint http://127.0.0.1:8000/v1 \
  --served-model-name my-model \
  --preset standard
```

Mode này không start/stop process vLLM. File `server.log` sẽ ghi rõ là benchmark dùng endpoint có sẵn.

## 4. Tuỳ chỉnh quan trọng

Đổi concurrency:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset standard \
  --concurrency 1,4,16
```

Tăng số request mỗi scenario:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset standard \
  --requests-per-scenario 12
```

Giới hạn output token chung cho mọi scenario:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset standard \
  --max-output-tokens 256
```

Truyền thêm argument thô cho vLLM:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --preset smoke \
  --vllm-arg=--dtype \
  --vllm-arg auto
```

Ghi chú: nếu extra arg bắt đầu bằng `--`, dùng dạng `--vllm-arg=--ten-arg` để argparse không hiểu nhầm đó là flag của `rag-bench`.

## 5. Đọc kết quả

Mỗi run ghi vào:

```text
runs/model_bench/<timestamp>_<hostname>_<model_slug>/
```

Các file chính:

- `summary.md`: bảng ngắn để copy so sánh giữa máy.
- `scenario_metrics.csv`: số aggregate dễ mở bằng spreadsheet.
- `scenario_metrics.json`: số aggregate dạng JSON.
- `requests.jsonl`: từng request riêng lẻ, gồm latency, TTFT, token usage, tok/s, lỗi.
- `hardware_samples.csv`: GPU util, VRAM, power, temperature, CPU load, RAM trong lúc chạy.
- `manifest.json`: model, command, git commit, dirty flag, platform, hardware snapshot.
- `server.log`: log vLLM khi benchmark tự start server.

Metric nên nhìn đầu tiên:

- `latency_p50_s`, `latency_p95_s`, `latency_p99_s`: độ trễ tổng.
- `ttft_p50_s`, `ttft_p95_s`: time-to-first-token, chỉ có khi streaming bật.
- `avg_output_tokens_per_s`, `max_output_tokens_per_s`: tốc độ sinh token theo request thành công.
- `requests_per_s`: throughput theo scenario/concurrency.
- `completion_tokens_per_s`: tổng completion token/s toàn scenario.
- `error_rate`: nếu khác `0`, xem `requests.jsonl` và `server.log` trước khi so sánh tốc độ.

Khi so sánh nhiều máy, ưu tiên so cùng:

- cùng branch/commit trong `manifest.json`;
- cùng model và `served_model_name`;
- cùng preset, concurrency, request count, max output tokens;
- cùng `max_model_len` và extra vLLM args;
- cùng trạng thái GPU không bị nghẽn nhiệt/power trong `hardware_samples.csv`.

## 6. Lỗi thường gặp

Model không load được:

- xem `server.log`;
- kiểm tra VRAM bằng `nvidia-smi`;
- thử giảm `--max-model-len`;
- thử thêm vLLM arg như `--vllm-arg=--dtype --vllm-arg auto`;
- chạy `--preset smoke` trước.

Server không healthy trong thời gian chờ:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model <model> \
  --preset smoke \
  --startup-timeout-s 1800
```

Benchmark endpoint có sẵn nhưng trả lỗi model:

- kiểm tra `--served-model-name` có đúng với model id mà server expose không;
- thử gọi endpoint `/v1/models`;
- nếu server không hỗ trợ streaming usage, chạy `--no-stream` để đo latency và tok/s ước lượng từ text.

Kết quả tok/s bất thường:

- kiểm tra `error_rate` trước;
- xem `hardware_samples.csv` để tìm GPU util thấp, VRAM full, power/temperature throttle;
- tăng `--requests-per-scenario` để giảm nhiễu;
- so sánh theo cùng concurrency, không trộn concurrency `1` với concurrency cao.
