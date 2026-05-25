# True Chat RAG Benchmark - Vietnamese Presentation Script

## Slide 1 - Title

Hôm nay em trình bày project True Chat RAG Benchmark. Trọng tâm là lõi thuật toán retrieval-augmented generation, cách project tổ chức nhiều chiến lược truy xuất, và kết quả benchmark cho thấy chiến lược nào hiệu quả nhất trong từng tập dữ liệu.

## Slide 2 - Project Goal

Mục tiêu của project không chỉ là làm chatbot RAG, mà là biến phần retrieval thành thứ có thể đo lường và so sánh. Cùng một benchmark, cùng một tập query, project chạy nhiều retriever khác nhau rồi ghi lại metric để quyết định bằng dữ liệu.

## Slide 3 - System Architecture

Kiến trúc chính đi từ dữ liệu benchmark sang registry retriever, sau đó qua vòng truy xuất và tùy chọn sinh câu trả lời bằng LLM. Điểm quan trọng là registry dùng chung, nên một strategy mới chỉ cần đăng ký một lần là dùng được ở CLI, proxy và UI.

## Slide 4 - Benchmark Algorithm

Vòng benchmark được thiết kế để tách riêng retrieval và generation. Khi bật `--skip-generation`, project chỉ đo chất lượng truy xuất, nhờ vậy kết quả không bị nhiễu bởi chất lượng câu trả lời của LLM.

## Slide 5 - Retriever Families

Project không cố định một thuật toán duy nhất. Nó xây dựng một bộ retriever có cùng interface để so sánh: từ lexical truyền thống, embedding vector, fusion, graph expansion, cho tới LLM-assisted query expansion và dictionary graph.

## Slide 6 - Sparse Baselines

BM25 là baseline quan trọng nhất vì nhanh, dễ giải thích và kết quả rất mạnh trên SciFact. TF-IDF có latency thấp hơn nhưng chất lượng thấp hơn BM25 ở SciFact. Keyword match phù hợp cho truy vấn định danh, nhưng nếu dùng đơn độc thì dễ bỏ sót ngữ cảnh.

## Slide 7 - Expansion and Fusion

Multi-query không thay đổi bản chất BM25, mà tạo nhiều cách diễn đạt query rồi trộn kết quả. RRF giúp trộn các danh sách có thang điểm khác nhau, vì nó dựa vào thứ hạng thay vì so raw score trực tiếp.

## Slide 8 - Graph BM25

Graph BM25 bắt đầu bằng BM25 để lấy seed document. Sau đó project xây một graph nhẹ giữa document và term, chọn term có IDF tốt, mở rộng candidate, rồi kết hợp điểm lexical và graph với trọng số 65 phần trăm và 35 phần trăm.

## Slide 9 - Dense and Hybrid Retrieval

Vector retrieval dùng embedding để bắt quan hệ ngữ nghĩa, nhưng project vẫn đưa tín hiệu lexical vào fusion hoặc rerank. Kết quả benchmark cho thấy hybrid và vector-rerank mạnh hơn khi query cần cả semantic similarity và exact scientific terms.

## Slide 10 - Dictionary Graph Pipeline

Dictionary graph là phần mở rộng theo domain. Dữ liệu DOCX được chuyển thành entry, node và edge có schema rõ ràng, sau đó runtime kết hợp direct lookup, BM25 và graph expansion. Vì vậy các biến thể như có dấu, không dấu, alias hoặc viết tắt vẫn có thể trỏ về mục từ đúng.

## Slide 11 - Operational Design

Project ghi nhận chi phí vận hành của LLM thay vì giấu nó trong thuật toán. Với LLM query rewrite hoặc LLM multi-query, benchmark vẫn đo token, latency, retry và lỗi riêng cho retrieval, nhờ đó ta thấy được trade-off giữa chất lượng và chi phí.

## Slide 12 - Benchmark Setup

Hai tập chính trong báo cáo là SciFact và NFCorpus, đều chạy 50 query với top-k bằng 3. Phần lớn kết quả là retrieval-only để đánh giá retriever trực tiếp bằng qrels, còn RAGAS chỉ là mẫu nhỏ để kiểm tra chất lượng câu trả lời.

## Slide 13 - SciFact Results

Trên SciFact, hybrid-rrf đạt hit@3 cao nhất là 0.86, nghĩa là 86 phần trăm query có ít nhất một tài liệu liên quan trong top 3. Vector-rerank có nDCG@3 cao nhất 0.8011, nhưng BM25 vẫn là baseline rất mạnh với build time thấp hơn nhiều.

## Slide 14 - NFCorpus Results

Trên NFCorpus, vector retrieval đạt hit@3 cao nhất là 0.68, tốt hơn BM25 khá rõ. Tuy nhiên nếu xét nDCG@3, hybrid-rrf nhỉnh hơn một chút, cho thấy việc kết hợp semantic signal và lexical signal vẫn có lợi.

## Slide 15 - LLM-Assisted Retrieval

Phần này là trade-off rõ nhất. LLM-assisted retrieval tốn khoảng 1.4 đến 2.4 giây mỗi query và thêm token cost, nhưng không vượt BM25 trong các run SciFact này. Đặc biệt query rewrite bằng Qwen làm giảm mạnh nDCG, có thể vì rewrite làm mất hoặc thay đổi thuật ngữ khoa học quan trọng.

## Slide 16 - Conclusions

Kết luận chính là không có retriever tốt nhất cho mọi trường hợp. BM25 vẫn rất mạnh và rẻ, vector tốt hơn ở dữ liệu semantic, còn hybrid thường là lựa chọn cân bằng. Với production chat, cần đo cả chất lượng, latency và token cost trước khi bật các strategy dùng LLM.
