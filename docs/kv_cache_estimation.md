# KV-Cache Estimation

BudgetRAG Phase 1B reports analytical KV-cache estimates from estimated context token counts.

Formula:

```text
kv_cache_bytes ~= 2 * layers * heads * head_dim * sequence_length * dtype_bytes
```

The factor `2` accounts for keys and values. Built-in profiles:

- `generic-small`: 32 layers, 32 heads, head dimension 128, 2-byte dtype.
- `qwen2.5-14b`: approximate 48 layers, 40 heads, head dimension 128, 2-byte dtype.

These estimates are useful for comparing context policies under the same model profile. They are not measured VRAM usage and do not mean this repo prunes `past_key_values` or modifies runtime attention caches.
