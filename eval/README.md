# Intelligent Search qrels

`intelligent_qrels.jsonl` phải được gán nhãn thủ công; không dùng kết quả của
chính hệ thống làm ground truth. Sao chép cấu trúc trong
`intelligent_qrels.example.jsonl`, sau đó gán nhãn đúng 10 query cho mỗi nhóm:
`visual-only`, `ocr-only`, `asr-only`, `caption-only`, `visual-ocr`,
`visual-asr`, `ocr-asr` và `temporal` (tổng cộng 80 query). Mỗi nhóm nên có cả
query tiếng Việt có dấu và không dấu. Báo cáo evaluator gồm số liệu tổng và
riêng cho từng giá trị `group` để so sánh các ablation.

Chạy đánh giá:

```bash
uv run python -m evaluation.intelligent \
  --experiment-name result \
  --qrels eval/intelligent_qrels.jsonl \
  --top-k 5,10,20 \
  --ablation all
```
