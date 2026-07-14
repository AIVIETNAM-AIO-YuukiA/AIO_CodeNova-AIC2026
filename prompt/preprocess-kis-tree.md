# Prompt: Preprocess KIS Tree (Scene + Sub-detail)

## Mục tiêu

Kết hợp Multi-Scene và Detail trong một lần xử lý: phân tích query thành các scene tuần tự, và với mỗi scene tiếp tục tách thành các atomic sub-detail. Đồng thời chuyển động từ hành động → mô tả tĩnh.

## Bối cảnh

Dành cho bài toán mà mỗi scene trong chuỗi có thể chứa nhiều chi tiết phức tạp. Cần cấu trúc cây: Level 1 là danh sách scene tuần tự, Level 2 là danh sách sub-detail trong mỗi scene.

## Đầu vào

Query dài có cả đặc điểm multi-scene lẫn multi-detail.

Ví dụ:
"Phân cảnh bắt đầu là chiếc đĩa sứ trắng trên khay gỗ. Bên cạnh là chén đựng dâu, chén chuối cắt sẵn và thìa màu nâu. Phân cảnh tiếp theo cho thấy đầu bếp đặt 2 chiếc bánh rán lên đĩa sứ. Sau đó đầu bếp rưới chocolate lên mặt bánh. Cuối cùng, đầu bếp đặt lát chuối lên bánh thứ nhất và lát dâu lên bánh thứ hai."

## Nguyên tắc xử lý

### 1. Xác định các scene (Level 1)
Dựa vào từ khóa chỉ thứ tự: "bắt đầu", "tiếp theo", "sau đó", "cuối cùng"...

### 2. Với mỗi scene, tách sub-detail (Level 2)
Mỗi scene có thể có nhiều atomic detail tồn tại đồng thời trong frame đó.

### 3. Chuyển động từ → mô tả tĩnh
Áp dụng cho tất cả các mô tả ở cả 2 level.

## Đầu ra JSON

```json
{
  "type": "multi_scene_detail",
  "scenes": [
    {
      "order": 1,
      "name": "scene_1",
      "description": "Mô tả scene 1 (đã tĩnh hóa)",
      "sub_details": ["detail 1 (tĩnh)", "detail 2 (tĩnh)", "detail 3 (tĩnh)"]
    },
    {
      "order": 2,
      "name": "scene_2",
      "description": "Mô tả scene 2 (đã tĩnh hóa)",
      "sub_details": ["detail 1 (tĩnh)", "detail 2 (tĩnh)"]
    }
  ]
}
```
