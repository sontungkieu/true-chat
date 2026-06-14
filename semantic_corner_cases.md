# Semantic Corner Case Tests

Snapshot: 2026-06-14.

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
| Viết tắt/alias | `pb`, `pbbc`, `qspb`, `QS PB` | `PB` phải ưu tiên `PHÁO BINH`; `PBBC` phải ưu tiên `PHÁO BINH BIÊN CHẾ`; `QSPB` phải match được cụm spaced acronym `QS PB` trong nguồn, không để match lỏng sang riêng `PB` hoặc chuỗi thô như `THƯỚC PB-74` vượt canonical match. | `tests/test_retrievers.py::test_dictionary_graph_retriever_matches_abbreviation_alias_to_headword` |
| Viết tắt trong câu hỏi ngắn | `PB`, `P B`, `pblagi`, `giaithichpb`, `CVHL nghĩa là gì?`, `cvhlnghialagi`, `KHCN xuất hiện ở đâu?`, `khcnxuathienodau`, `Q S P B` | Planner phải normalize các biến thể lookup ngắn về cùng target acronym/term gốc, rồi dictionary retrieval dùng target đó để giữ kết quả nhất quán trong cả Dictionary mode và Text-only dictionary fallback; không để câu có hậu tố `là gì`/`nghĩa là gì`/`mean`/`xuất hiện ở đâu`, tiền tố `giải thích`, viết dính, hoặc thừa khoảng trắng rơi sang lexical weak match hoặc benchmark fallback. | `tests/test_dictionary_query_planner.py::test_definition_plan_strips_question_noise_for_short_acronyms`, `tests/test_chat_service.py::test_dictionary_mode_normalizes_short_acronym_definition_queries`, `tests/test_chat_service.py::test_text_mode_dictionary_fallback_uses_normalized_lookup_target_for_mentions` |
| Cụm địa danh nhiều từ | `Pháo đài Xuân Canh`, `pháo đài Láng, pháo đài Xuân Tảo` | Entry có nhắc đúng cụm trong định nghĩa phải đứng trên headword một từ như `PHÁO`; metadata trả `query_highlights` để UI bôi vàng cụm match trong rich dictionary card/source panel. | `tests/test_retrievers.py::test_dictionary_graph_retriever_prefers_exact_phrase_mentions_over_generic_headwords` |
| Highlight bỏ dấu nhưng không match trong từ dài | `thạ` | UI có thể fold dấu để `thạ` tương đương `tha`, nhưng chỉ bôi vàng khi đó là token/cụm độc lập; không bôi `THA` trong `THANG` hoặc `tha` trong `tham gia`. | `tests/test_api.py::test_chat_page_renders_built_in_ui` smoke-checks `isHighlightBoundary`; cần thêm JS/unit test riêng nếu frontend test runner được tách ra. |
| Alias/concept từ graph artifact | Graph có edge `has_alias` và `has_concept` cho `base:P-0023` | Loader phải attach `aliases=["PB"]` và `concepts=["lực lượng tác chiến"]` vào dictionary document metadata. | `tests/test_dictionary.py::test_dictionary_artifact_loader_attaches_graph_aliases_and_concepts` |
| Legacy artifact | Artifact cũ chỉ có `entries.jsonl` không rich schema | Loader vẫn đọc được, đánh `schema_version=1`, không crash. | `tests/test_dictionary.py::test_dictionary_artifact_loader_accepts_plain_legacy_entries` |
| Source namespace | Base và supplement cùng local id `B-0001` | Parser có thể namespace thành `base:B-0001` và `supp2021:B-0001` để không collision. | `tests/test_dictionary.py::test_dictionary_parser_can_namespace_supplement_source_ids` |
| DOCX fidelity | `AMONIT`, italic `thuốc nổ phá`, subscript `NH4NO3`, màu đỏ | Parser giữ casing, bold, italic, subscript, color trong rich blocks. | `tests/test_dictionary.py::test_docx_dictionary_parser_preserves_run_formatting_and_casing` |

### Ví Dụ Dictionary Dễ Gây Nhầm

Các ví dụ dưới đây ghi lại lỗi từng gặp hoặc lỗi có xác suất cao khi mở rộng dictionary retrieval. Chúng nên được dùng lại khi tune prompt, tune reranker, hoặc viết eval thủ công cho model.

| Case | Ví dụ query | Hành vi sai cần tránh | Hành vi đúng mong muốn | Ý nghĩa cho prompt/eval |
| --- | --- | --- | --- | --- |
| Biến thể phiên âm và dấu | `hexogen`, `hêxôgen`, `hê-xô-gen`, `he-xo-gen` | `hêxôgen` chỉ kéo các mục liên quan như `TRẠM NỔ` hoặc `AMONIT` vì trong định nghĩa có nhắc đến hêxôgen, nhưng bỏ qua mục từ chính `HEXOGEN`. | `HEXOGEN` đứng đầu; `TRẠM NỔ`, `AMONIT` có thể đứng sau như related mentions. | Model cần phân biệt “định nghĩa trực tiếp” với “nguồn có nhắc tới thuật ngữ”. Khi chỉ có related mention, answer phải nói rõ đó không phải mục từ chính. |
| Viết tắt quân sự | `pb`, `pbbc`, `qspb`, `QS PB` | `pb` bị match vào chuỗi kỹ thuật như `THƯỚC PB-74`, `pbbc` bị tách thành token rời và kéo nhầm các mục có chữ `PB`, hoặc `QSPB` không tìm thấy nguồn đang ghi dạng `QS PB`. | `PB` map về `PHÁO BINH`; `PBBC` map về `PHÁO BINH BIÊN CHẾ`; `QS PB` và `QSPB` map về cùng entry nếu artifact có alias hoặc text mention spaced acronym tương ứng. | Prompt/eval nên yêu cầu giữ nguyên viết tắt, giải thích alias nếu có bằng chứng, normalize khoảng trắng trong acronym nhiều cụm, và không tự mở rộng viết tắt khi dictionary không hỗ trợ. |
| Viết tắt kèm phrasing lookup | `PB`, `PB là gì?`, `giải thích PB`, `CVHL nghĩa là gì?`, `cho tôi biết CVHL là gì`, `what does SPG9 mean?` | Các câu cho kết quả khác nhau: câu trần có direct alias/headword hit, còn câu có wrapper hỏi đáp bị tách target thành cả cụm câu hỏi nên rơi sang lexical weak match hoặc context benchmark cũ. | Mọi biến thể lookup ngắn phải được coi là cùng một target acronym/term gốc; kết quả phải ưu tiên canonical abbreviation/headword evidence trước alias phụ hoặc lexical mention. | Eval nên kiểm cùng một thuật ngữ dưới nhiều phrasing ngắn, không chỉ test bare acronym; answer không được đổi nghĩa chỉ vì người dùng thêm “là gì”, “nghĩa là gì”, “giải thích”, hoặc “mean”. |
| Viết tắt chỉ xuất hiện trong cụm liên quan | `KHCN`, `KHCN là gì?`, `KHCN xuất hiện ở đâu?` | Model có thể bịa một định nghĩa chính thức cho `KHCN`, hoặc bỏ qua nguồn vì không có headword/alias trực tiếp. | Không coi đây là định nghĩa chính thức nếu từ điển không có mục/alias explicit; chỉ nêu rằng `KHCN` xuất hiện trong cụm `KHCN&MT` ở các mục liên quan như `CÔNG TÁC KHOA HỌC QUÂN SỰ` và `TRƯỞNG PHÒNG KHOA HỌC QUÂN SỰ`, rồi trích nguồn. | Đây là mention/occurrence lookup: answer phải phân biệt “từ viết tắt được nhắc trong nguồn” với “mục từ chính thức có định nghĩa”. |
| Địa danh có từ chung | `pháo đài Láng`, `pháo đài Xuân Tảo`, `Pháo đài Xuân Canh` | Vì query chứa từ `pháo`, hệ thống đưa mục `PHÁO` lên cao dù người dùng đang hỏi một địa danh/công trình cụ thể. | Mục từ địa danh hoặc mục có nhắc đúng cụm `pháo đài ...` đứng trên; `PHÁO` chỉ là liên quan xa. | Model nên nói “nguồn đang nói về địa danh/pháo đài cụ thể”, không chuyển trọng tâm sang định nghĩa chung của pháo. |
| Nguồn nhắc cụm nhưng không phải headword | `Pháo đài Xuân Canh` | Bỏ qua `NGÀY TRUYỀN THỐNG PHÁO BINH` vì headword không phải `Pháo đài Xuân Canh`. | Vẫn giữ `NGÀY TRUYỀN THỐNG PHÁO BINH` nếu định nghĩa có nhắc đúng cụm, nhưng xếp sau mục từ trực tiếp nếu có. | Eval nên chấp nhận related evidence có trích cụm chính xác, nhưng answer phải nêu đây là nguồn liên quan, không phải entry canonical nếu không phải headword. |
| Query năm hoặc số rộng | `1948` | UI hiện text thô kiểu `Mục từ gốc [base:L-0015]` và citation id nội bộ như `[base:T-0153]`, hoặc bị kẹt pending vì lưu quá nhiều rich source vào `localStorage`. | UI hoàn tất request, render các mục từ liên quan dưới dạng rich dictionary card, citation trong giải thích thành pill thứ tự, và chỉ lưu compact source metadata vào history. | Broad numeric query có thể trả nhiều mục dài; eval cần kiểm tra cả retrieval quality lẫn khả năng render/persist mà không làm browser đơ. |
| Citation dùng local id không namespace | Model trả `[Đ-0025]` trong giải thích, còn source runtime có `doc_id=base:Đ-0025` hoặc `source_entry_id=Đ-0025` | UI để nguyên `[Đ-0025]` dưới dạng text thô, người dùng không bấm mở được nguồn. | UI map được cả full id, suffix sau namespace, `source_entry_id`, và rank sang cùng source để render pill `[1]`/`[n]`. | Eval UI citation nên dùng cả citation full id và local id vì model thường bỏ namespace khi cite. |
| Header mục từ trong chat cũ | Chat cũ lưu text `Mục từ gốc [base:P-0001]: ...` | Sau khi mở rộng citation lookup, `[base:P-0001]` trong header bị đổi thành citation pill, làm header trông như citation trong câu trả lời. | Không render citation token nằm ngay trong header `Mục từ gốc [...]` hoặc `Original entry [...]`; chỉ render citation trong phần giải thích/nội dung. | Cần test backward compatibility với lịch sử chat cũ vì format message đã thay đổi qua nhiều mốc UI. |
| Chat cũ đã compact source | Chat cũ lưu dictionary source chỉ còn `text/raw_docx_text`, không còn `rich_blocks` | UI không nhận ra đây là dictionary card, nên mất box mục từ và mất highlight cụm query. | Vẫn render dictionary source thành card bằng text fallback và dùng `query_highlights` để bôi vàng nếu rich blocks không còn trong localStorage. | Persisted history cần compact để tránh quota, nhưng UI phải degrade gracefully khi thiếu rich DOCX blocks. |
| Phân biệt match/liên quan | Query `SPG9`; kết quả đầu có highlight `SPG9`, các mục như `A-12` chỉ được kéo theo vì liên quan ngữ nghĩa/graph | Người đọc không biết source nào khớp trực tiếp với query và source nào chỉ là liên quan rộng. | Card dictionary có pill xanh `Khớp` khi có direct/highlight match, pill vàng `Liên quan` khi chỉ là nguồn liên quan. | UI nên giúp người đọc đánh giá độ trực tiếp của bằng chứng mà không bắt họ đọc score thô. |
| Ref chéo trong panel từ điển | Bật `Ref chéo từ điển`, đang mở mục `ĐKZ SPG9`; trong định nghĩa có cụm được bôi vàng `ĐKZ` hoặc người dùng chọn cụm `SPG9` rồi bấm | Panel chỉ là text tĩnh, muốn xem mục `ĐKZ` phải quay lại chat và gõ query mới; nếu tính năng luôn bật thì dễ gây nhảy panel ngoài ý muốn. | Khi toggle bật, bấm highlight hoặc bôi đen cụm trong panel gọi lookup từ điển và thay panel bằng mục từ top-1 nếu tìm thấy; khi toggle tắt, panel chỉ để đọc. | Ref chéo nên là lookup local không gọi LLM để giữ nhanh, rẻ và dễ audit, nhưng phải nằm sau công tắc vì còn là thử nghiệm UX. |
| Highlight bỏ dấu quá rộng | Query `thạ`; nguồn có `BẮN BẬC THANG` và `tham gia` | Do fold dấu `thạ -> tha`, UI bôi vàng `THA` trong `THANG` và `tha` trong `tham gia`, làm người đọc tưởng hai từ này liên quan trực tiếp. | Chỉ highlight khi `tha/thạ` là token độc lập hoặc cụm nguyên vẹn; không highlight substring nằm trong từ dài hơn. | Khi làm prompt/eval, cần phân biệt “match để recall” và “match để giải thích cho người dùng”; highlight/UI phải bảo thủ hơn retrieval. |

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
- Các case trong file này nên được dùng làm nguyên liệu prompt tuning/eval set cho model: yêu cầu model giữ nguyên thuật ngữ nguồn, không diễn giải quá đà khi retrieval chỉ là related mention, và nêu rõ khi kết quả chỉ là mục từ liên quan chứ không phải định nghĩa trực tiếp.
