# Local Qwen KV-Cache Estimates

This report estimates decoder KV-cache memory without loading model weights.

- Formula: `2 * layers * num_key_value_heads * head_dim * seq_len * batch_size * dtype_bytes`
- Batch size: `1`
- Dtype bytes: `2`

| Model | Layers | KV heads | Head dim | Seq len | KV GiB | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 1,024 | 0.012 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 2,048 | 0.023 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 4,096 | 0.047 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 8,192 | 0.094 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 16,384 | 0.188 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 32,768 | 0.375 | `fallback` |
| `Qwen/Qwen2.5-0.5B` | 24 | 2 | 64 | 131,072 | 1.500 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 1,024 | 0.027 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 2,048 | 0.055 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 4,096 | 0.109 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 8,192 | 0.219 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 16,384 | 0.438 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 32,768 | 0.875 | `fallback` |
| `Qwen/Qwen2.5-1.5B` | 28 | 2 | 128 | 131,072 | 3.500 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 1,024 | 0.035 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 2,048 | 0.070 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 4,096 | 0.141 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 8,192 | 0.281 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 16,384 | 0.562 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 32,768 | 1.125 | `fallback` |
| `Qwen/Qwen2.5-3B` | 36 | 2 | 128 | 131,072 | 4.500 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 1,024 | 0.055 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 2,048 | 0.109 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 4,096 | 0.219 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 8,192 | 0.438 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 16,384 | 0.875 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 32,768 | 1.750 | `fallback` |
| `Qwen/Qwen2.5-7B` | 28 | 4 | 128 | 131,072 | 7.000 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 1,024 | 0.188 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 2,048 | 0.375 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 4,096 | 0.750 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 8,192 | 1.500 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 16,384 | 3.000 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 32,768 | 6.000 | `fallback` |
| `Qwen/Qwen2.5-14B` | 48 | 8 | 128 | 131,072 | 24.000 | `fallback` |

These are analytical KV-cache estimates only. They exclude model weights, activations, framework overhead, allocator fragmentation, and paged-attention block overhead.
