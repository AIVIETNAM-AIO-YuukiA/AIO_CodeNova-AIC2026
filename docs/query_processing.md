# Hướng dẫn xử lý Truy vấn (Query Processing) & Cấu hình LLM

Dự án hỗ trợ một module xử lý truy vấn thông minh (`QueryProcessor`) giúp dịch thuật, phân tích ý định (intent parsing) và làm giàu câu lệnh (prompt enrichment) trước khi thực hiện tìm kiếm vector.

---

## 1. Cơ chế hoạt động (How it works)

Hệ thống hỗ trợ 2 chế độ xử lý truy vấn:

1.  **Chế độ thông thường (Pass-through Mode):**
    *   Kích hoạt khi **không có** LLM API Key.
    *   Câu truy vấn của người dùng được giữ nguyên và gửi trực tiếp đến CLIP để tìm kiếm.
2.  **Chế độ LLM (LLM Processing Mode):**
    *   Kích hoạt khi **có** cấu hình LLM API Key (biến môi trường `GEMINI_API_KEY`).
    *   Hệ thống sử dụng mô hình Gemini để:
        *   Tự động dịch câu truy vấn từ tiếng Việt sang tiếng Anh.
        *   Làm giàu câu lệnh (thêm mô tả chi tiết về bố cục, màu sắc, ánh sáng) giúp CLIP khớp tốt hơn.
        *   Bóc tách từ khóa OCR (chữ trên màn hình), ASR (giọng nói) và các metadata khác để chuẩn bị cho việc kết hợp tìm kiếm sau này.

Hệ thống sử dụng cơ chế **Tự động hạ cấp (Graceful Fallback)**: Nếu không có API Key hoặc kết nối mạng bị lỗi, hệ thống sẽ tự động chuyển về Chế độ thông thường mà không gây gián đoạn hay báo lỗi chương trình.

---

## 2. Cấu hình LLM API Key

Để kích hoạt chế độ xử lý thông minh bằng LLM, bạn cần thiết lập biến môi trường `GEMINI_API_KEY`.

### Bước 1: Lấy API Key miễn phí
Truy cập [Google AI Studio](https://aistudio.google.com/) và tạo một API Key mới.

### Bước 2: Thiết lập biến môi trường
Trước khi khởi động UI hoặc chạy lệnh tìm kiếm, hãy đặt biến môi trường trong terminal của bạn:

#### Trên Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
```

#### Trên Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
```

#### Trên Linux / macOS / Google Colab / Kaggle:
```bash
export GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
```

---

## 3. Cách sử dụng (Usage)

Khi biến môi trường đã được thiết lập, mọi lệnh tìm kiếm (CLI và UI) sẽ tự động sử dụng LLM để xử lý câu truy vấn:

### Tìm kiếm qua CLI:
```bash
# Mặc định hệ thống sẽ tự động phát hiện GPU (auto), bạn có thể chỉ định rõ thiết bị nếu cần
uv run codenova search --experiment-name "ten-experiment" --device <device> "cảnh nấu ăn"
```
*Lưu ý: `<device>` có thể là `auto` (mặc định), `cuda` (nếu chạy GPU) hoặc `cpu` (nếu chạy CPU).*

### Chạy giao diện Web UI:
```bash
uv run codenova serve-ui --experiment-name "ten-experiment" --device <device>
```
Khi bạn nhập câu hỏi tiếng Việt trên giao diện Web, hệ thống sẽ tự động dịch, mở rộng câu lệnh sang tiếng Anh chuẩn xác ở phía backend trước khi truy xuất dữ liệu từ FAISS.