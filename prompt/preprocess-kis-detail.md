# Prompt: Preprocess KIS Detail

## Mục tiêu

Chia một câu query mô tả **một scene duy nhất nhưng có rất nhiều chi tiết** thành các atomic detail. Mỗi detail là một chi tiết cụ thể có thể xuất hiện trong cùng một frame. Đồng thời chuyển mọi động từ hành động thành mô tả tĩnh.

## Bối cảnh

Hệ thống tìm kiếm video sử dụng **CLIP** (text-to-image) — chỉ hiểu được trạng thái tĩnh trong một frame. Khi một frame có nhiều chi tiết (nhiều người, nhiều object, nhiều màu sắc, chữ viết...), cần tách thành các subquery riêng để search từng cái, sau đó fusion kết quả.

## Đầu vào

Một câu query mô tả 1 scene duy nhất nhưng có nhiều chi tiết cùng tồn tại.

Ví dụ:
- "Hai người đàn ông (áo hồng và áo trắng) đứng hai bên, ở giữa là bốn em nhỏ (áo đỏ, trắng, váy hồng, áo xanh). Phía sau là phông nền đỏ với khẩu hiệu. Trước mặt mỗi em là túi quà lớn."
- "Ở giữa là robot vẽ tranh: hai cánh tay robot gắn trên khung kim loại vẽ trên khung vẽ hình vuông. Bức tranh có mảng màu đen-xám-trắng. Xung quanh là phòng trưng bày. Cuối có quyển sách bìa xanh chữ MILK."

## Nguyên tắc xử lý

### 1. Tách thành atomic detail
Mỗi detail là một ý độc lập, có thể search riêng:
- "người đàn ông mặc áo hồng"
- "người đàn ông mặc áo trắng"
- "bốn em nhỏ mặc áo đủ màu"
- "phông nền đỏ có khẩu hiệu"
- "túi quà lớn trước mặt mỗi em"

### 2. Chuyển động từ → mô tả tĩnh
Giống quy tắc ở prompt multi-scene.

### 3. Giữ nguyên
Màu sắc, số lượng, vị trí, chữ viết, tên riêng.

### 4. Không thêm
Chi tiết không có trong query gốc.

## Đầu ra JSON

```json
{
  "type": "detail",
  "main_scene": "Mô tả tổng quan scene (đã tĩnh hóa)",
  "sub_details": [
    {
      "id": "detail_1",
      "text": "atomic detail 1 (đã tĩnh hóa)",
      "category": "person|object|background|text"
    },
    {
      "id": "detail_2",
      "text": "atomic detail 2 (đã tĩnh hóa)",
      "category": "person|object|background|text"
    }
  ]
}
```
