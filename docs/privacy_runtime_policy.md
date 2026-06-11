# Runtime Privacy Policy For RAG Chat

This document describes the first production-safety layer for chat-based RAG runtime routing. It uses only metadata and synthetic labels; it must not contain raw private or semi-private source text.

## Data Tiers

Runtime retrieval artifacts use a monotonic tier order:

```text
public < semi_private < private
```

- `public`: benchmark corpora and explicitly public demo indexes.
- `semi_private`: local domain or dictionary artifacts unless source metadata marks them private.
- `private`: sources whose path or metadata indicates private, secret, classified, top-secret, or equivalent trusted-private-backend handling.

Malformed runtime tier values are treated conservatively as private-risk. Missing tier on raw retrieved hits is also private-risk unless metadata explicitly marks the source public. Dictionary/domain artifacts default to semi-private when no safer source metadata is present.

## Conversation Taint

Privacy is tracked per chat session, not per isolated turn. Each request may provide a `session_id`; the server keeps an in-memory `ConversationPrivacyState` for that id and returns the updated state in the response.

Effective turn tier is computed from:

- previous session taint,
- message or attachment tier metadata, when provided,
- retrieved context tiers,
- memory/tool-output tier metadata, when provided in later extensions.

Once a session sees private context, the session remains private-tainted until the user starts a new session id or explicitly resets privacy state for a clean session. Later public or semi-private retrieval does not downgrade the session.

## History Toggle

`memory=false` or `history_enabled=false` only removes prior messages from the prompt context. It does not clear privacy taint. A session that previously retrieved private context still requires a trusted private inference backend even when memory is off.

## Provider Routing Matrix

| session taint | allowed inference boundary |
| --- | --- |
| public | private-capable backend or external SaaS |
| semi_private | private-capable backend, or external SaaS only when `allow_external_semi_private=true` |
| private | trusted private backend and trusted private model only |

Groq, MiMo, DeepSeek, OpenAI, Anthropic, Gemini, and similar cloud providers are external SaaS by default. Private-tainted sessions block external generation, query rewrite, multi-query expansion, rerank, judge, and summarizer calls before text is sent to a provider.

Private data is allowed only when both the backend and model are trusted:

- backend kind is private-capable: `local_process`, `self_hosted_private`, `private_lan`, or `private_vpc`,
- backend id is listed in `trusted_private_backends` / `--private-backend`,
- model id is listed in `trusted_private_models` / `--trusted-private-models`, or in a per-backend allowlist through `--private-backend-model BACKEND:MODEL[,MODEL...]`.

`trusted_local_models` / `--trusted-local-models` remains a backward-compatible alias for `trusted_private_models`, but the model name alone is never enough. A model id that is trusted for a private backend is still blocked when routed to Groq, MiMo, DeepSeek, or any other external SaaS backend. API-key authentication also does not make an external SaaS backend private-safe.

## Source Payload Redaction

Private retrieved sources are redacted in response payloads by default:

- `text` is returned as `null`,
- `title` is hidden,
- raw fields such as `plain_text`, `raw_docx_text`, `rich_blocks`, and `evidence_text` are removed or set to `null`,
- safe metadata such as `doc_id`, `rank`, `score`, `data_tier`, `doc_type`, and `source_id` may remain.

Raw private source payloads require all of:

- a trusted route,
- a trusted private backend and trusted private model when the tier is private,
- an authorized request path,
- explicit `RAG_BENCH_INCLUDE_PRIVATE_SOURCE_TEXT=1`.

This is disabled by default.

## Clean Session

To clear taint, create a new chat with a new `session_id`, or send an explicit privacy reset for a session that does not reuse prior history, memory, or source payloads. `reset_privacy=true` is rejected for a private-tainted session when the same request still carries earlier assistant/system messages or multi-turn history.

## Not Implemented Yet

This layer does not implement query planning, rule/case schemas, fine-tuning, RL, DPO, PPO, GRPO, online bandits, or runtime KV pruning. It also does not verify that a trusted backend id is truly inside the private inference boundary; deployment config must wire trusted backend ids only to controlled local, LAN, VPC, or self-hosted inference.
