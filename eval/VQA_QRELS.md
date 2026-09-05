# Ground truth và evaluator cho VQA

Evaluator VQA đo đồng thời khả năng tìm đúng video, trả lời đúng và dẫn chứng đúng. Nhãn phải được kiểm tra thủ công; không lấy kết quả của chính hệ thống làm ground truth.

## Cấu trúc JSONL

Mỗi dòng trong `vqa_qrels.jsonl` là một JSON object:

```json
{
  "query_id": "vqa_001",
  "query": "Mô tả chuỗi sự kiện dùng để tìm video",
  "question": "X là con gì?",
  "context": "Ngữ cảnh bổ sung, có thể để trống",
  "acceptable_answers": ["nghêu", "con nghêu"],
  "relevant_videos": [
    {"video_name": "L26_V254.mp4"}
  ],
  "relevant_frame_ids": [],
  "relevant_moments": [],
  "group": "object-sequence"
}
```

Các trường bắt buộc là `query_id`, `query` và `question`. Một dòng phải có ít nhất một loại nhãn: đáp án, video, frame hoặc moment.

- `acceptable_answers`: mọi cách viết ngắn được chấp nhận. So sánh không phân biệt hoa thường và dấu câu, nhưng vẫn giữ dấu tiếng Việt.
- `relevant_videos`: mỗi phần tử có `video_id`, `video_name`, hoặc cả hai. Nên dùng tên video khi chưa biết internal hash ID.
- `relevant_frame_ids`: chỉ thêm frame ID đã được kiểm tra thật.
- `relevant_moments`: chỉ thêm khi đã xem video và biết khoảng đúng; mỗi phần tử có `video_id` hoặc `video_name`, cùng `start_sec` và `end_sec` với `end_sec > start_sec`.
- `group`: nhóm câu hỏi dùng để xem metric riêng, ví dụ `object-sequence`, `ocr`, `asr`, `counting`, `temporal`.

Không điền frame ID hoặc timestamp giả để đủ schema. Metric tương ứng chỉ được tính khi qrels có nhãn đó. File `vqa_qrels.example.jsonl` đã seed trường hợp `L26_V254 → nghêu` nhưng cố ý để trống frame và moment.

## Chạy evaluator

```bash
cp eval/vqa_qrels.example.jsonl eval/vqa_qrels.jsonl

uv run python -m evaluation.vqa \
  --experiment-name result \
  --qrels eval/vqa_qrels.jsonl \
  --top-k 20,50 \
  --pipeline-mode grounded
```

Evaluator tự nạp file `.env` ở thư mục project. Hãy bảo đảm
`OPENROUTER_API_KEY` và `VQA_OPENROUTER_MODEL` (hoặc `OPENROUTER_MODEL`) đã
được cấu hình; model dùng cho VQA phải hỗ trợ nhiều ảnh.

Các tùy chọn hữu ích:

```bash
# Chọn model embedding giống UI
--enabled-models jina-clip-v2,siglip2-so400m,vietnamese-embedding

# Đánh giá khi tắt query planner hoặc reranker
--no-llm
--no-reranker

# Chỉ định file báo cáo
--output runs/result/evaluation/vqa_manual.json
```

## Metric

- `video_recall@3`: tỷ lệ video ground truth xuất hiện trong ba candidate đầu.
- `answer_em`: exact match sau chuẩn hóa; các tiền tố như `X là` hoặc `Đáp án là` được bỏ.
- `answer_f1`: token F1 tốt nhất so với các đáp án chấp nhận.
- `evidence_accuracy`: có ít nhất một supporting frame đúng frame/moment đã gán nhãn.
- `temporal_iou`: IoU cao nhất giữa selected moment và answer-bearing moment.
- `answer_top_k_stability`: đáp án có giữ nguyên giữa các Top K hay không.
- `video_top_k_stability`: selected video có giữ nguyên giữa các Top K hay không.
- `evidence_top_k_jaccard`: mức trùng supporting frame giữa các Top K.
- `latency_p50_ms`, `latency_p95_ms`: độ trễ đo bên ngoài toàn request.
- `openrouter_calls_total`: tổng số HTTP request thực sự gửi đến OpenRouter,
  bao gồm retry. API còn trả `usage.openrouter_operations` để phân biệt số tác
  vụ logic (planner/verifier) với số lần gọi mạng.
- `openrouter_cost_total`: tổng chi phí OpenRouter do API usage báo về (nếu provider trả trường này).

`summary` dùng kết quả ở Top K lớn nhất làm số liệu accuracy chuẩn, còn `by_top_k` cho phép so từng Top K. Độ ổn định được tính trên toàn bộ danh sách Top K đã truyền vào.
