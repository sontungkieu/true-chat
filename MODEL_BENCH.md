# Hướng Dẫn Benchmark Model vLLM

File này hướng dẫn chạy benchmark một model trên từng máy theo workflow thủ công: mở máy, clone repo, setup môi trường, nhập tên model, chạy benchmark, rồi so sánh kết quả giữa các máy.

## Quickstart Vast AI RTX 5060 Ti 16GB

Khi thuê Vast AI, ưu tiên chọn instance RTX 5060 Ti 16GB có driver đủ mới. Trong container, kiểm tra:

```bash
nvidia-smi
```

Chọn profile theo `Driver Version`:

- Driver Linux `>= 580.65.06`: ưu tiên profile CUDA 13.0.
- Driver Linux `>= 575.57.08` nhưng `< 580.65.06`: dùng fallback CUDA 12.9.
- Nếu Vast không thuê được host đủ driver cho CUDA 13, chuyển sang CUDA 12.9.

Clone repo:

```bash
git clone <repo-url> true-chat
cd true-chat
git checkout bench/vllm-model-bench
```

## Copy/Paste Lệnh Bench Theo Model

Sau khi chọn đúng setup CUDA cho máy, các lệnh bench model nằm cùng một chỗ ở đây. Các lệnh chuẩn dùng `standard`, tức chạy synthetic short/medium/long ở concurrency `1,2,4,8`. Với model lớn trên RTX 5060 Ti 16GB, dùng `max_model_len=4096` để `synthetic_long` không bị reject. Chỉ đổi preset thành `smoke` khi cần health check nhanh xem model có load được không.

Mặc định các lệnh Vast chạy:

- inference engine: vLLM OpenAI-compatible server qua `vllm serve`;
- speculative decoding: tắt, trừ khi set `BENCH_VLLM_SPECULATIVE_CONFIG`;
- attention backend: vLLM auto-selection, trừ khi set `BENCH_VLLM_ATTENTION_BACKEND`;
- quantization: đọc theo model config, ví dụ model id AWQ sẽ được ghi `model_hint=awq`; set `BENCH_VLLM_QUANTIZATION=awq` nếu muốn ép explicit arg;
- serving: local vLLM process tự start/stop, không dùng server ngoài;
- output: `manifest.json` và `summary.md` ghi rõ engine, version vLLM, quantization, KV cache dtype, attention backend, speculative method, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `enforce_eager`, GPU utilization, và CPU offload.
- cache flow: kiểm tra dung lượng trống trong `HF_HOME`, dọn cache model khác nếu thiếu disk, prefetch/resume model đích trước khi start vLLM, rồi mới kiểm tra VRAM và chạy benchmark.

Setup CUDA 13.0:

```bash
scripts/setup_vast_5060ti_cuda130.sh
```

Bench từng model trên CUDA 13.0:

```bash
# Qwen 2.5 7B AWQ, baseline 16GB dễ chạy nhất
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard

# Qwen 3.5 9B AWQ 4-bit
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda130.sh cyankiwi/Qwen3.5-9B-AWQ-4bit standard

# Qwen 2.5 14B AWQ 4-bit, không offload theo mặc định
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard

# Qwen 3.5 9B AWQ 8-bit/BF16-INT8
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 BENCH_GPU_MEMORY_UTILIZATION=0.94 BENCH_VLLM_KV_CACHE_DTYPE=turboquant_4bit_nc BENCH_VLLM_CPU_OFFLOAD_GB=2 \
  scripts/bench_vast_5060ti_cuda130.sh cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8 standard

# Llama-3 16B AWQ
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda130.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard

# Chạy suite chính: Qwen3.5 9B 4-bit + Qwen2.5 14B AWQ + Llama-3 16B AWQ
scripts/bench_vast_5060ti_model_suite_cuda130.sh standard

# Opt-in Qwen 3.5 9B 8-bit, chậm vì cần CPU offload trên 16GB
BENCH_INCLUDE_QWEN35_8BIT=1 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

A/B speculative decoding nhẹ nhất bằng n-gram, không cần draft model:

```bash
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=0 \
  BENCH_VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

A/B sweep speculative decoding cho Llama-3 16B AWQ, mặc định chạy baseline no-SD rồi n-gram SD 2 và 4 token:

```bash
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_sd_sweep_cuda130.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard
```

Nếu baseline đang hoặc đã chạy riêng rồi, bỏ baseline trong sweep để chỉ chạy SD:

```bash
env BENCH_SD_INCLUDE_BASELINE=0 BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_sd_sweep_cuda130.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard
```

A/B attention backend, chỉ dùng khi muốn so auto-selection với backend cụ thể:

```bash
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=0 \
  BENCH_VLLM_ATTENTION_BACKEND=FLASHINFER \
  scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

Fallback CUDA 12.9 nếu driver chưa đủ cho CUDA 13:

```bash
scripts/setup_vast_5060ti_cuda129.sh
```

Bench từng model trên CUDA 12.9:

```bash
# Qwen 2.5 7B AWQ, baseline 16GB dễ chạy nhất
scripts/bench_vast_5060ti_cuda129.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard

# Qwen 3.5 9B AWQ 4-bit
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda129.sh cyankiwi/Qwen3.5-9B-AWQ-4bit standard

# Qwen 2.5 14B AWQ 4-bit, không offload theo mặc định
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda129.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard

# Qwen 3.5 9B AWQ 8-bit/BF16-INT8
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 BENCH_GPU_MEMORY_UTILIZATION=0.94 BENCH_VLLM_KV_CACHE_DTYPE=turboquant_4bit_nc BENCH_VLLM_CPU_OFFLOAD_GB=2 \
  scripts/bench_vast_5060ti_cuda129.sh cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8 standard

# Llama-3 16B AWQ
env BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda129.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard

# Chạy suite chính: Qwen3.5 9B 4-bit + Qwen2.5 14B AWQ + Llama-3 16B AWQ
scripts/bench_vast_5060ti_model_suite_cuda129.sh standard

# Opt-in Qwen 3.5 9B 8-bit, chậm vì cần CPU offload trên 16GB
BENCH_INCLUDE_QWEN35_8BIT=1 scripts/bench_vast_5060ti_model_suite_cuda129.sh standard
```

A/B sweep speculative decoding CUDA 12.9 tương tự CUDA 13.0:

```bash
env BENCH_SD_INCLUDE_BASELINE=0 BENCH_MAX_MODEL_LEN=4096 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=4096 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_sd_sweep_cuda129.sh solidrust/Llama-3-16B-Instruct-v0.1-AWQ standard
```

Llama 4 Scout 17B không bật mặc định vì thường cần HF access token và không 16GB-safe. Nếu vẫn muốn thử có kiểm soát:

```bash
BENCH_INCLUDE_LLAMA4=1 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard

BENCH_INCLUDE_LLAMA4=1 \
BENCH_LLAMA4_MODEL=unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit \
scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

Nếu storage `/workspace` nhỏ, ép wrapper xoá cache trước khi tải model kế tiếp:

```bash
BENCH_MODEL_CACHE_CLEANUP=always scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
BENCH_MODEL_CACHE_CLEANUP=always scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

Mặc định wrapper dùng `BENCH_MODEL_CACHE_CLEANUP=auto`: trước mỗi model, script kiểm tra dung lượng trống trong `HF_HOME`; nếu thấp hơn `BENCH_MIN_CACHE_FREE_GB=35`, single-model script xoá cache của các model khác nhưng giữ cache model đích, còn suite script xoá cache của model vừa bench trước đó. Đặt `BENCH_MODEL_CACHE_CLEANUP=never` nếu muốn giữ toàn bộ model cache, hoặc `BENCH_PREFETCH_MODEL=0` nếu muốn để vLLM tự tải model khi debug startup.

### Profile CUDA 13.0

Setup:

```bash
scripts/setup_vast_5060ti_cuda130.sh
```

Script CUDA 13.0 sẽ:

- dùng `HF_HOME=/workspace/hf-cache` nếu `/workspace` ghi được;
- dùng `XDG_CACHE_HOME=/workspace/vllm-cache` nếu `/workspace` ghi được;
- ép `UV_PROJECT_ENVIRONMENT=$PWD/.venv` để không bị shell active `(main)` cài nhầm vào `/venv/main`;
- pin mặc định `VLLM_VERSION=0.22.0`;
- ép backend `cu130`;
- clean stack `vllm/torch/...` cũ trong `.venv`;
- cài vLLM với backend `cu130`, để resolver chọn đúng PyTorch build một lần;
- fail sớm nếu driver thấp hơn mức cần cho CUDA 13.0;
- verify `torch.version.cuda == 13.0`.

Chạy health check mặc định, model AWQ phù hợp hơn với VRAM 16GB:

```bash
scripts/bench_vast_5060ti_cuda130.sh
```

Chạy model/preset cụ thể:

```bash
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-7B-Instruct-AWQ smoke
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Override cấu hình an toàn bằng env:

```bash
BENCH_MAX_MODEL_LEN=6144 \
BENCH_GPU_MEMORY_UTILIZATION=0.88 \
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Chạy suite chính gồm Qwen3.5 9B AWQ 4-bit, Qwen2.5 14B AWQ, và Llama-3 16B AWQ. Không truyền preset thì suite mặc định chạy `standard` với `max_model_len=4096` để có synthetic long:

```bash
scripts/bench_vast_5060ti_model_suite_cuda130.sh
```

Qwen3.5 9B 8-bit là case opt-in vì trên 16GB nó cần CPU offload và rất chậm:

```bash
BENCH_INCLUDE_QWEN35_8BIT=1 scripts/bench_vast_5060ti_model_suite_cuda130.sh
```

### Fallback CUDA 12.9

Setup:

```bash
scripts/setup_vast_5060ti_cuda129.sh
```

Script CUDA 12.9 dùng cùng cache `/workspace`, ép `UV_PROJECT_ENVIRONMENT=$PWD/.venv` để không bị shell active `(main)` cài nhầm vào `/venv/main`, pin mặc định `VLLM_VERSION=0.22.0`, ép backend `cu129`, clean stack cũ trong `.venv`, cài vLLM với backend `cu129` để resolver chọn đúng PyTorch build một lần, fail sớm nếu driver thấp hơn mức cần cho CUDA 12.9, và verify `torch.version.cuda == 12.9`.

Chạy health check mặc định:

```bash
scripts/bench_vast_5060ti_cuda129.sh
```

Chạy benchmark chuẩn:

```bash
scripts/bench_vast_5060ti_cuda129.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Override cấu hình giống profile CUDA 13.0:

```bash
BENCH_MAX_MODEL_LEN=6144 \
BENCH_GPU_MEMORY_UTILIZATION=0.88 \
scripts/bench_vast_5060ti_cuda129.sh Qwen/Qwen2.5-7B-Instruct-AWQ standard
```

Chạy suite chính gồm Qwen3.5 9B AWQ 4-bit, Qwen2.5 14B AWQ, và Llama-3 16B AWQ. Không truyền preset thì suite mặc định chạy `standard` với `max_model_len=4096` để có synthetic long:

```bash
scripts/bench_vast_5060ti_model_suite_cuda129.sh
```

Qwen3.5 9B 8-bit là case opt-in vì trên 16GB nó cần CPU offload và rất chậm:

```bash
BENCH_INCLUDE_QWEN35_8BIT=1 scripts/bench_vast_5060ti_model_suite_cuda129.sh
```

Nếu muốn xác nhận lỗi AWQ Marlin trên fallback CUDA 12.9, có thể ép kernel AWQ thường:

```bash
BENCH_VLLM_QUANTIZATION=awq scripts/bench_vast_5060ti_cuda129.sh
```

Không dùng flag này cho số benchmark chuẩn nếu driver đã đúng; để vLLM tự chọn kernel thường cho tốc độ tốt hơn.

### Model suite 5060 Ti

Các suite script mặc định chạy:

| Nhãn | Model id | Ghi chú |
| --- | --- | --- |
| Qwen3.5 9B AWQ | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | Bản AWQ 4-bit, hợp lý hơn bản full cho VRAM 16GB. |
| Qwen2.5 14B AWQ | `Qwen/Qwen2.5-14B-Instruct-AWQ` | Bản official AWQ 4-bit, dùng để đo model 14B không CPU offload trên 16GB trước khi thử quant nhỏ hơn. |
| Llama-3 16B AWQ | `solidrust/Llama-3-16B-Instruct-v0.1-AWQ` | Community merge AWQ, có thể sát VRAM hơn; lệnh đo chính dùng `standard`, đổi về `smoke` khi chỉ cần debug load. |

Qwen3.5 9B AWQ 8-bit `cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8` không chạy mặc định. Bật bằng `BENCH_INCLUDE_QWEN35_8BIT=1` khi cần kiểm tra riêng; kết quả này đo cả CPU/PCIe offload nên không nên trộn với bảng tốc độ GPU sạch.

Suite model lớn dùng defaults an toàn hơn wrapper single-model:

- `BENCH_MAX_MODEL_LEN=4096`;
- `BENCH_MAX_NUM_SEQS=1`;
- `BENCH_MAX_NUM_BATCHED_TOKENS=4096`;
- `BENCH_ENFORCE_EAGER=1`;
- `BENCH_VLLM_ATTENTION_BACKEND` bỏ trống, nghĩa là để vLLM tự chọn backend;
- `BENCH_VLLM_SPECULATIVE_CONFIG` bỏ trống, nghĩa là không dùng speculative decoding;
- `BENCH_QWEN35_8BIT_KV_CACHE_DTYPE=turboquant_4bit_nc` cho riêng Qwen3.5 9B 8-bit khi opt-in;
- `BENCH_QWEN35_8BIT_GPU_MEMORY_UTILIZATION=0.94` cho riêng Qwen3.5 9B 8-bit khi opt-in nếu không set global `BENCH_GPU_MEMORY_UTILIZATION`;
- `BENCH_QWEN35_8BIT_CPU_OFFLOAD_GB=2` cho riêng Qwen3.5 9B 8-bit khi opt-in nếu không set global `BENCH_VLLM_CPU_OFFLOAD_GB`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Nếu bật speculative decoding, truyền JSON thẳng qua env để wrapper đưa vào `vllm serve --speculative-config`:

```bash
BENCH_VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

Hoặc chạy sweep n-gram có sẵn:

```bash
scripts/bench_vast_5060ti_sd_sweep_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

Các biến cho SD sweep:

- `BENCH_SD_INCLUDE_BASELINE=0` nếu đã có baseline và chỉ muốn chạy SD;
- `BENCH_SD_NGRAM_TOKENS="2 4"` để chọn số token speculative cần thử;
- `BENCH_SD_PROMPT_LOOKUP_MIN=2` và `BENCH_SD_PROMPT_LOOKUP_MAX=5` cho prompt lookup n-gram.

Với draft model/MTP/DFlash, chỉ bật khi model/speculator thật sự tương thích và còn VRAM. Trên RTX 5060 Ti 16GB, sweep mặc định chỉ dùng `ngram` vì không tải thêm draft model; Qwen3.5 9B 8-bit có CPU offload thì không nên dùng làm baseline speculative.

Trước mỗi model, Vast wrapper cũng:

- kill stale GPU process có tên chứa `VLLM`/`vllm` nếu `BENCH_KILL_STALE_VLLM=1` (default);
- chờ tổng `memory.used` của GPU xuống dưới `BENCH_GPU_READY_MAX_USED_MB=512`;
- timeout sau `BENCH_GPU_READY_TIMEOUT_S=90`;
- có thể bỏ qua check bằng `BENCH_SKIP_GPU_READY_CHECK=1` nếu cần debug thủ công.

Wrapper còn quản lý disk cache khi chạy model:

- `BENCH_MODEL_CACHE_CLEANUP=auto` là default, chỉ xoá cache khi dung lượng trống trong `HF_HOME` thấp hơn ngưỡng;
- `BENCH_MIN_CACHE_FREE_GB=35` là ngưỡng default;
- `BENCH_MODEL_CACHE_CLEANUP=always` xoá cache model khác trước mỗi single-model run hoặc cache model vừa chạy trước model kế tiếp trong suite;
- `BENCH_MODEL_CACHE_CLEANUP=never` giữ toàn bộ cache.
- `BENCH_PREFETCH_MODEL=1` là default, tải/resume model đích trước khi start vLLM; dùng `BENCH_PREFETCH_MODEL=0` nếu muốn để vLLM tự tải.

Lý do: preset `standard` có `synthetic_long` khoảng 3000 prompt tokens, nên default suite dùng `max_model_len=4096`. Nếu model lớn bị OOM ở 4096, fallback về 2048 để chạy short/medium/smoke trước, nhưng khi đó `synthetic_long` sẽ không còn hợp lệ:

```bash
BENCH_MAX_MODEL_LEN=2048 \
BENCH_MAX_NUM_BATCHED_TOKENS=2048 \
scripts/bench_vast_5060ti_cuda130.sh cyankiwi/Qwen3.5-9B-AWQ-4bit smoke
```

Nếu log vLLM báo `No available memory for the cache blocks`, nghĩa là sau khi load/profile model vẫn không còn đủ chỗ tạo KV cache block. Qwen3.5 9B 8-bit khi opt-in dùng `gpu_memory_utilization=0.94`, `turboquant_4bit_nc`, và `cpu_offload_gb=2`; nếu vẫn gặp lỗi này trên máy sạch VRAM, tăng offload trước khi tăng `gpu_memory_utilization`:

```bash
BENCH_INCLUDE_QWEN35_8BIT=1 BENCH_QWEN35_8BIT_CPU_OFFLOAD_GB=4 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

RAM không đầy không có nghĩa offload nhanh. `cpu_offload_gb` chỉ cho phép một phần weight đi qua CPU/RAM; tốc độ thường bị giới hạn bởi CPU/PCIe và kernel scheduling, nên tok/s thấp là kỳ vọng trên 5060 Ti 16GB.

Với Qwen2.5 14B AWQ và Llama-3 16B AWQ, ưu tiên giữ không offload. Nếu 4096 context bị cache-block/OOM, giảm context trước:

```bash
env BENCH_MAX_MODEL_LEN=3072 BENCH_MAX_NUM_SEQS=1 BENCH_MAX_NUM_BATCHED_TOKENS=3072 BENCH_ENFORCE_EAGER=1 \
  scripts/bench_vast_5060ti_cuda130.sh Qwen/Qwen2.5-14B-Instruct-AWQ standard
```

Nếu muốn so sánh Qwen3.5 9B 8-bit bằng KV-cache dtype khác, override riêng biến này:

```bash
BENCH_INCLUDE_QWEN35_8BIT=1 BENCH_QWEN35_8BIT_KV_CACHE_DTYPE=fp8 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
BENCH_INCLUDE_QWEN35_8BIT=1 BENCH_QWEN35_8BIT_KV_CACHE_DTYPE=none scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

Llama 4 Scout 17B:

- model chính thức: `meta-llama/Llama-4-Scout-17B-16E-Instruct`;
- đây là MoE lớn và thường cần quyền access/HF token;
- không 16GB-safe để chạy mặc định trong vLLM;
- chỉ bật khi muốn thử có kiểm soát:

```bash
BENCH_INCLUDE_LLAMA4=1 scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

Nếu muốn thử repo Llama 4 khác, override:

```bash
BENCH_INCLUDE_LLAMA4=1 \
BENCH_LLAMA4_MODEL=unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit \
scripts/bench_vast_5060ti_model_suite_cuda130.sh standard
```

## 1. Chuẩn bị trên mỗi máy

Clone repo và chuyển sang branch benchmark:

```bash
git clone <repo-url> true-chat
cd true-chat
git checkout bench/vllm-model-bench
```

Setup Python env và cài vLLM. Với RTX 5060 Ti 16GB, ưu tiên script CUDA 13.0 khi driver đủ mới:

```bash
scripts/setup_vllm_bench_cuda130.sh
```

Nếu máy/driver chưa đủ cho CUDA 13.0, dùng fallback CUDA 12.9:

```bash
scripts/setup_vllm_bench_cuda129.sh
```

Script tự chọn backend theo `uv` cũng vẫn có sẵn:

```bash
scripts/setup_vllm_bench.sh
```

Script setup chỉ cài dependency Python và vLLM trong `.venv`. Script không cài hoặc sửa driver NVIDIA, CUDA, package hệ thống, hay cấu hình GPU. Hai script CUDA-specific sẽ gỡ stack vLLM/PyTorch CUDA hiện có trong `.venv` trước khi cài lại để tránh lẫn wheel CUDA 13 với torch CUDA 12.9. Torch không được cài bằng một bước riêng; vLLM resolver chọn build PyTorch tương thích với backend đã chọn.

Các script CUDA-specific kiểm tra driver tối thiểu trước khi cài:

- CUDA 12.9: driver Linux `>= 575.57.08`.
- CUDA 13.0: driver Linux `>= 580.65.06`.

Nếu driver thấp hơn, script sẽ dừng sớm. Có thể override bằng `VLLM_SKIP_DRIVER_CHECK=1` cho môi trường đặc biệt, nhưng không nên dùng để chạy benchmark chuẩn vì lỗi kernel như `cudaErrorUnsupportedPtxVersion` thường sẽ xuất hiện khi load model.

Ghi chú quan trọng: dòng `CUDA Version` trong `nvidia-smi` là mức CUDA runtime tối đa mà driver hỗ trợ, không phải version CUDA mà Python package đang dùng. Version cần kiểm tra sau setup là:

```bash
.venv/bin/python - <<'PY'
import torch, vllm
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

Nếu log lỗi có `libcudart.so.13`, env đang có binary CUDA 13 nhưng runtime chưa khớp. Cài lại đúng backend bằng `scripts/setup_vllm_bench_cuda130.sh` nếu driver đủ mới; nếu không, fallback bằng `scripts/setup_vllm_bench_cuda129.sh`.

Nếu máy cần version vLLM cụ thể:

```bash
VLLM_VERSION=0.22.0 scripts/setup_vllm_bench_cuda130.sh
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

Trong lúc chạy, terminal sẽ hiện progress theo từng bước:

- `[vast-setup ...]`: setup profile Vast, cache, backend CUDA;
- `[vllm-setup ...]`: tạo/sync `.venv`, clean package cũ, cài vLLM, verify Torch CUDA;
- `[vast-bench ...]`: wrapper Vast, kiểm tra cache/disk, prefetch model, dọn VRAM, chạy từng model hoặc từng suite;
- `[model-bench ...]`: core benchmark, start/chờ vLLM, warmup, từng scenario/concurrency, ghi artifact.

JSON kết quả cuối vẫn in ở stdout; progress log của core benchmark in ở stderr để dễ theo dõi khi chạy dài mà không phá output JSON.

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

- `summary.md`: bảng ngắn để copy so sánh giữa máy, gồm engine/config inference, latency/tok/s và peak VRAM/GPU/power/temp.
- `scenario_metrics.csv`: số aggregate dễ mở bằng spreadsheet, gồm cả hardware aggregate theo scenario.
- `scenario_metrics.json`: số aggregate dạng JSON, gồm cả hardware aggregate theo scenario.
- `requests.jsonl`: từng request riêng lẻ, gồm latency, TTFT, token usage, tok/s, lỗi.
- `hardware_samples.csv`: raw samples GPU util, VRAM, power, temperature, CPU load, RAM trong lúc chạy.
- `manifest.json`: model, command, git commit, dirty flag, platform, hardware snapshot, `inference_engine`, và `serving_config` gồm quantization, KV-cache dtype, attention backend, speculative decoding, eager/CUDA graph mode, batch/cache limits, GPU utilization, CPU offload.
- `server.log`: log vLLM khi benchmark tự start server.

Metric nên nhìn đầu tiên:

- `latency_p50_s`, `latency_p95_s`, `latency_p99_s`: độ trễ tổng.
- `ttft_p50_s`, `ttft_p95_s`: time-to-first-token, chỉ có khi streaming bật.
- `avg_output_tokens_per_s`, `max_output_tokens_per_s`: tốc độ sinh token theo request thành công.
- `requests_per_s`: throughput theo scenario/concurrency.
- `completion_tokens_per_s`: tổng completion token/s toàn scenario.
- `error_rate`: nếu khác `0`, xem `requests.jsonl` và `server.log` trước khi so sánh tốc độ.
- `gpu_peak_memory_used_mb`, `gpu_peak_memory_used_percent`: VRAM peak theo scenario.
- `gpu_peak_util_percent`, `gpu_avg_util_percent`: mức dùng GPU peak/trung bình theo scenario.
- `gpu_peak_power_w`, `gpu_avg_power_w`, `gpu_peak_temperature_c`: điện năng và nhiệt peak/trung bình theo scenario.
- `ram_peak_used_mb`, `ram_peak_used_percent`, `cpu_load_1m_peak`: RAM và CPU host theo scenario.

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

Import lỗi `libcudart.so.13` hoặc mismatch CUDA:

- nếu muốn chạy CUDA 13.0, dùng `scripts/setup_vllm_bench_cuda130.sh`;
- nếu driver chưa đủ CUDA 13.0, dùng fallback `scripts/setup_vllm_bench_cuda129.sh`;
- không dùng `uv pip uninstall -y`; `uv pip uninstall` không có flag `-y`;
- sau setup, kiểm tra `torch.version.cuda` bằng đoạn Python ở phần chuẩn bị.

Lỗi `CUDA error: the provided PTX was compiled with an unsupported toolchain`:

- đây thường là driver quá cũ so với CUDA backend/kernel mà vLLM đang dùng;
- với CUDA 12.9, cập nhật driver lên ít nhất `575.57.08`;
- với CUDA 13.0, cập nhật driver lên ít nhất `580.65.06`;
- nếu đang dùng AWQ và muốn thử workaround tạm, ép vLLM không tự chọn AWQ Marlin:

```bash
uv run --frozen --no-sync rag-bench model-bench \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --preset smoke \
  --tensor-parallel-size auto \
  --max-model-len 4096 \
  --vllm-arg=--gpu-memory-utilization \
  --vllm-arg 0.85 \
  --vllm-arg=--quantization \
  --vllm-arg awq
```

Workaround này chỉ để xác nhận lỗi nằm ở kernel quantization; benchmark chuẩn vẫn nên chạy trên driver đúng.

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
