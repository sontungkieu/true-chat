# Semantic Corner Case Tests

Snapshot: 2026-05-24.

File này tổng hợp các corner case ngữ nghĩa hiện đang được bảo vệ bằng test. Mục tiêu là giúp nhìn nhanh những lỗi đã từng gặp trong RAG/chat/dictionary pipeline và biết test nào sẽ báo regressions nếu hành vi bị hỏng.

## Retrieval

| Nhóm | Case đang test | Kỳ vọng | Test |
| --- | --- | --- | --- |
| Token khoa học trong câu tiếng Việt | `giải thích BH1 bằng tiếng Việt` | `multi-query` phải giữ token khoa học `BH1`, không để phần hướng dẫn tiếng Việt làm nhiễu retrieval. | `tests/test_retrievers.py::test_multi_query_keeps_scientific_token_from_vietnamese_instruction` |
| Exact keyword | `yellow banana` | `keyword-match` đưa tài liệu banana lên đầu. | `tests/test_retrievers.py::test_keyword_match_retriever_returns_exact_keyword_match_first` |
| Graph expansion | Query `alpha` với seed doc có term `bridge/kinase/pathway` | `graph-bm25` không chỉ trả seed doc mà còn kéo được tài liệu second-hop qua shared terms. | `tests/test_retrievers.py::test_graph_bm25_expands_from_seed_document_neighbors` |
| LLM multi-query metadata | Query `Which animal purrs?` | LLM query expansion sinh biến thể, merge đúng tài liệu, và giữ metadata về LLM retrieval call. | `tests/test_retrievers.py::test_llm_multi_query_retriever_records_retrieval_llm_metadata` |
| Keyword bằng LLM trước search | `giải thích BH1 bằng tiếng Việt` với `keyword-match` trong chat | LLM lọc ra các keyword `BH1`, `BH1 Bcl-2`, `BH1 domain apoptosis` rồi mới search, giúp không search nguyên câu dài. | `tests/test_chat_service.py::test_keyword_match_uses_llm_keywords_before_search` |

## Dictionary

| Nhóm | Case đang test | Kỳ vọng | Test |
| --- | --- | --- | --- |
| Rich metadata dictionary | `AMONIT` | Dictionary retriever giữ `kind=dictionary` và `rich_blocks` để UI render bản gốc. | `tests/test_retrievers.py::test_dictionary_graph_retriever_preserves_dictionary_metadata` |
| Biến thể dấu và gạch nối | `hexogen`, `hêxôgen`, `hê-xô-gen`, `he-xo-gen` | Tất cả phải resolve về cùng entry canonical `base:H-0011 / HEXOGEN`; các entry liên quan như `AMONIT`, `TRẠM NỔ` vẫn có thể đứng sau. | `tests/test_retrievers.py::test_dictionary_graph_retriever_matches_accent_folded_headword` |
| Viết tắt/alias | `pb`, `pbbc` | `PB` phải ưu tiên `PHÁO BINH`; `PBBC` phải ưu tiên `PHÁO BINH BIÊN CHẾ`, không để match chuỗi thô như `THƯỚC PB-74` vượt canonical match. | `tests/test_retrievers.py::test_dictionary_graph_retriever_matches_abbreviation_alias_to_headword` |
| Alias/concept từ graph artifact | Graph có edge `has_alias` và `has_concept` cho `base:P-0023` | Loader phải attach `aliases=["PB"]` và `concepts=["lực lượng tác chiến"]` vào dictionary document metadata. | `tests/test_dictionary.py::test_dictionary_artifact_loader_attaches_graph_aliases_and_concepts` |
| Legacy artifact | Artifact cũ chỉ có `entries.jsonl` không rich schema | Loader vẫn đọc được, đánh `schema_version=1`, không crash. | `tests/test_dictionary.py::test_dictionary_artifact_loader_accepts_plain_legacy_entries` |
| Source namespace | Base và supplement cùng local id `B-0001` | Parser có thể namespace thành `base:B-0001` và `supp2021:B-0001` để không collision. | `tests/test_dictionary.py::test_dictionary_parser_can_namespace_supplement_source_ids` |
| DOCX fidelity | `AMONIT`, italic `thuốc nổ phá`, subscript `NH4NO3`, màu đỏ | Parser giữ casing, bold, italic, subscript, color trong rich blocks. | `tests/test_dictionary.py::test_docx_dictionary_parser_preserves_run_formatting_and_casing` |

## Chat And Answer Behavior

| Nhóm | Case đang test | Kỳ vọng | Test |
| --- | --- | --- | --- |
| Force ngôn ngữ trả lời | Request `language="vi"` | System prompt phải có `Required response language: Vietnamese` và metadata ghi `language=vi`. | `tests/test_chat_service.py::test_rag_chat_service_forces_selected_response_language` |
| Default ngôn ngữ | Không gửi language | System prompt mặc định force English để output không trôi theo ngôn ngữ câu hỏi. | `tests/test_chat_service.py::test_rag_chat_service_answers_with_retrieved_context_and_history` |
| API language validation | `language="fr"` | API trả 400 với lỗi `language must be one of: en, vi`. | `tests/test_api.py::test_chat_completion_rejects_unknown_language` |
| Dictionary command | `/dict AMONIT` | Route sang `dictionary-graph`, giữ rich metadata, prompt dictionary cũng force language. | `tests/test_chat_service.py::test_dict_command_routes_to_dictionary_retriever_with_rich_metadata` |
| Cited low-score source | Một source score `0.0` được LLM cite, một source score `0.0` không được cite | Source score thấp không cite bị ẩn; source score thấp có cite vẫn giữ để audit citation. | `tests/test_chat_service.py::test_uncited_zero_score_sources_are_hidden_but_cited_zero_score_sources_remain` |
| Text mode chống route sai | User chọn text nhưng request retriever là `image-digits` | Backend vẫn dùng text retriever mặc định, không để image retriever chen vào text answer. | `tests/test_chat_service.py::test_text_mode_ignores_image_retriever_request` |
| Image command không tốn LLM | `/img digit 7` | Route sang `image-digits`, trả image metadata, không gọi LLM generation. | `tests/test_chat_service.py::test_img_command_routes_to_image_retriever_without_llm_generation` |
| Image rewrite | Image mode bật rewrite với model Qwen | Dùng selected model để rewrite query ảnh, rồi search image. | `tests/test_chat_service.py::test_image_mode_can_rewrite_query_with_selected_model` |
| Text + image | `response_mode=text_image` | Trả text RAG trước, sau đó append image results liên quan. | `tests/test_chat_service.py::test_text_image_mode_appends_image_results_after_text_retrieval` |

## UI Contract Smoke Tests

| Nhóm | Case đang test | Kỳ vọng | Test |
| --- | --- | --- | --- |
| UI có dictionary mode | HTML chứa `Dictionary`, `Từ điển`, `/dict`, `dictionary-graph` | Frontend vẫn expose mode dictionary và command routing. | `tests/test_api.py::test_chat_page_renders_built_in_ui` |
| Rich dictionary render | HTML chứa `renderRichBlocks`, `dictionary-inline-list`, `dictionaryAnswerParts` | UI vẫn có code render rich dictionary card trong main answer và side panel. | `tests/test_api.py::test_chat_page_renders_built_in_ui` |
| Citation UX | HTML chứa `renderTextWithCitations`, `citation-ref`, `ragDetailsOpen` | Citation vẫn clickable và disclosure state không reset khi mở nguồn. | `tests/test_api.py::test_chat_page_renders_built_in_ui` |
| Request profile | HTML chứa `captureRequestOptions`, `formatUserRequestMeta`, `languageMode` | Dev mode vẫn show config đã dùng tại thời điểm gửi câu hỏi, gồm mode/search/model/language/rewrite. | `tests/test_api.py::test_chat_page_renders_built_in_ui` |

## Retriever Registry Aliases

| Alias | Canonical strategy | Test |
| --- | --- | --- |
| `lexical` | `bm25` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `dense` | `vector` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `find` | `keyword-match` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `graph`, `graph-rag` | `graph-bm25` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `img` | `image-digits` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `dict`, `dictionary` | `dictionary-graph` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |
| `rerank` | `vector-rerank` | `tests/test_retriever_registry.py::test_registry_normalizes_aliases_and_creates_retrievers` |

## Gaps Nên Bổ Sung

- Chưa có test trực tiếp trên full artifact thật trong automated suite cho `hê-xô-gen`, `PB`, `PBBC`; hiện mới có fixture nhỏ và manual script đã chạy trên `runs/pb_dictionary_base_supp2021_prod_graph`.
- Chưa có semantic eval tự động cho dictionary disambiguation khi một viết tắt có nhiều nghĩa, ví dụ `PB = pháo binh`, `phóng bồi`, `phát bắn`.
- Chưa có test cho typo/fuzzy edit distance, ví dụ `hexgogen`, `pháo bihn`.
- Chưa có test cho synonym không nằm trong alias graph, ví dụ một thuật ngữ dân gian/phi chuẩn trỏ tới mục từ chính.
- Chưa có test đo chất lượng answer cuối cùng theo reference cho dictionary mode; hiện chủ yếu test retrieval/prompt/routing.
- Chưa có test semantic cho Graph RAG dựa trên entity edges thật trong `dictionary_graph.sqlite`; `graph-bm25` hiện được test bằng graph term expansion fixture.
