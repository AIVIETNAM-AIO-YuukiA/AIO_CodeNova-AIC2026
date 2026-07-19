# Prompt: Preprocess KIS Multi-Scene

## Mục tiêu

Chia một câu query mô tả nhiều scene xảy ra tuần tự trong video thành các scene riêng biệt. Mỗi scene là một câu mô tả ngắn gọn, đã được chuyển động từ hành động → trạng thái tĩnh.

## Bối cảnh

Hệ thống tìm kiếm video sử dụng **CLIP** (text-to-image) — chỉ hiểu được trạng thái tĩnh trong một frame, không hiểu diễn biến theo thời gian. Do đó, mọi động từ hành động cần được chuyển đổi thành mô tả tĩnh.

## Đầu vào

Một câu query dài mô tả video, có nhiều scene/sự kiện xảy ra tuần tự.

Ví dụ:
- "Phân cảnh bắt đầu là một chiếc đĩa sứ màu trắng trên khay gỗ. Bên cạnh là chén đựng dâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ. Sau đó đầu bếp rưới chocolate lên mặt bánh."
- "Đoạn clip bắt đầu với cảnh tác phẩm điêu khắc cát. Nền phía sau là họa tiết vòm cong. Tiếp theo có cảnh nhiều tác phẩm khác bằng cát và 2 cột khói màu hồng."

## Nguyên tắc xử lý

### 1. Xác định các scene
Dựa vào các từ khóa chỉ thứ tự: "bắt đầu", "tiếp theo", "sau đó", "cuối cùng", "kế tiếp", "mở đầu", "kết thúc", "phân cảnh đầu", "phân cảnh sau", ...

### 2. Chuyển động từ → mô tả tĩnh
Mỗi động từ hành động phải được chuyển thành mô tả về vị trí / tư thế / trạng thái trong frame:

| Động từ (hành động) | Mô tả tĩnh |
|---------------------|------------|
| "đặt bánh xuống bàn" | "bàn tay ở phía trên bánh, bánh nằm trên mặt bàn" |
| "rưới chocolate lên bánh" | "chocolate đang chảy trên mặt bánh, tay cầm chai phía trên" |
| "cô gái múa kiếm" | "cô gái trong tư thế dang tay cầm kiếm" |
| "người đang chạy" | "người ở tư thế chạy, một chân trước một chân sau" |
| "người phụ nữ bế chó" | "người phụ nữ ôm chú chó trên tay" |
| "các kị sĩ cưỡi ngựa chạy" | "kị sĩ ngồi trên lưng ngựa, ngựa đang phi" |
| "tháo ống kính khỏi máy ảnh" | "ống kính đã tháo rời, đặt trên khăn, bên cạnh thân máy" |
| "ngồi xuống ghế" | "người đang ngồi trên ghế" |
| "đứng dậy" | "người đang đứng, tay vịn vào bàn/ghế" |
| "cầm vật gì đó" | "tay đang cầm vật, vật ở trong lòng bàn tay" |
| "ném/quăng/thả" | "vật đang ở trên không giữa đường bay, tay vừa rời vật" |

### 3. Giữ nguyên
- Màu sắc, số lượng, vị trí không gian
- Tên riêng, con số, biểu tượng
- Bối cảnh nền (trong nhà/ngoài trời, ngày/đêm)
- Quan hệ giữa các object (bên trái, bên phải, phía trên, phía sau)

### 4. Không thêm
Chi tiết không có trong query gốc.

## Đầu ra JSON

```json
{
  "type": "multi_scene",
  "scenes": [
    {
      "order": 1,
      "description": "Mô tả scene 1 (đã tĩnh hóa)",
      "original_verbs": ["đặt", "rưới"]
    },
    {
      "order": 2,
      "description": "Mô tả scene 2 (đã tĩnh hóa)",
      "original_verbs": ["cầm"]
    }
  ]
}
```
