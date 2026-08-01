# Preprocess — KIS Detail (merged)

From a single input query, produce two categories of subqueries — **General** (coarse, stage 1) and **Specific** (fine, stage 2). Output in English. Every subquery must be a purely static visual description — convert actions, movements, and verbs into static visual states.

## Action-to-static conversion rule

Rewrite any action/verb phrase so the result describes what a single freeze-frame would show:

| Input (Vietnamese / natural) | Output (English, static visual) |
|---|---|
| người A nắm một cái cọc | a pole held by person A |
| người đang khiêng một con cá | a fish carried on a pole |
| mọi người nhảy múa trên thuyền | people standing on a boat |
| người phụ nữ đang đi qua đường | a woman crossing a street |
| xe cộ đang chạy qua lại | vehicles on the street |
| hai người mặc áo đỏ | two people in red shirts |

## Output format

```
## GENERAL
<general subquery 1>
<general subquery 2>
...

## SPECIFIC
<specific subquery 1>
<specific subquery 2>
...
```

## General rules (applies to both categories)

- Each subquery: short phrase (3-10 words), pure static visual, no named entities (festival names, place names, person names), no abstract meaning, no cultural context
- Output only the two sections — no preamble, no explanation
- All subqueries in **English**

## Critical filter — loại bỏ yếu tố không search được

SigLIP/CLIP chỉ hiểu được các **vật thể, màu sắc, bố cục không gian** cụ thể — không hiểu khái niệm trừu tượng, cảm xúc, mục đích, ý nghĩa văn hóa hay tâm linh.

**Quy tắc:** nếu một chi tiết không thể hiện rõ ràng bằng một vật thể/màu sắc/hình dạng cụ thể trong khung hình → loại bỏ nó khỏi subquery.

### Các loại KHÔNG được đưa vào subquery:

| Loại | Ví dụ trong input | Giải thích | Cách xử lý |
|---|---|---|---|
| **Hành vi trừu tượng** | cầu nguyện, cầu may, khấn vái, tưởng nhớ, tri ân | SigLIP không phân biệt được "người cầu nguyện" vs "người ngồi" | Loại bỏ hẳn, hoặc chỉ giữ phần mô tả hình thể thuần túy (vd: "người ngồi" thay vì "người cầu nguyện") |
| **Mục đích / lý do** | cầu cho chuyến đi bình an, để tỏ lòng thành kính, nhằm tạ ơn thần linh | Không có hình ảnh nào thể hiện "mục đích" | Loại bỏ hẳn |
| **Cảm xúc / nội tâm** | vui mừng, xúc động, thành kính, tha thiết | Biểu cảm khuôn mặt có thể thấy, nhưng cảm xúc trừu tượng như "thành kính" thì không | Giữ biểu cảm nếu rõ ràng (vd: "mỉm cười"), bỏ hoàn toàn cảm xúc trừu tượng |
| **Ý nghĩa biểu tượng** | tượng trưng cho sự may mắn, biểu tượng của mùa màng bội thu, mang ý nghĩa tâm linh | Ý nghĩa biểu tượng không hiển thị trong ảnh | Loại bỏ hẳn |
| **Bối cảnh văn hóa / lịch sử** | lễ hội Obon truyền thống, nghi thức cổ xưa, phong tục tập quán | Tên lễ hội, bối cảnh văn hóa không phải là đặc trưng thị giác | Loại bỏ tên riêng & bối cảnh văn hóa, chỉ giữ mô tả hình ảnh thuần túy |
| **Quan hệ xã hội / vai vế** | người dân, các cụ già, thanh niên, em bé, vua tôi | Trừ khi thể hiện qua trang phục cụ thể không thì bỏ | Chỉ giữ nếu có đặc điểm nhận dạng thị giác (vd: "người mặc áo dài" thay vì "người dân") |

### Cách áp dụng — ví dụ

| Input (có yếu tố nhiễu) | Sau khi lọc | Giải thích |
|---|---|---|
| người dân đang cầu nguyện cho chuyến đi bình an | người ngồi | "cầu nguyện" là hành vi trừu tượng, "cho chuyến đi bình an" là mục đích → bỏ hết, chỉ giữ tư thế |
| mọi người vui mừng nhảy múa trong lễ hội Obon | people on a stage, people in traditional costumes | "vui mừng" là cảm xúc → bỏ, "lễ hội Obon" là tên riêng văn hóa → bỏ, chỉ giữ mô tả thị giác |
| đoàn rước kiệu với ý nghĩa tâm linh sâu sắc | a procession with a palanquin | "ý nghĩa tâm linh sâu sắc" → bỏ, chỉ giữ "đoàn rước kiệu" chuyển thành "a procession with a palanquin" |
| cụ già đang khấn vái trước bàn thờ | an elderly person in front of an altar | "khấn vái" → bỏ, chỉ giữ "người trước bàn thờ" |

### Nguyên tắc cuối cùng

**Khi nghi ngờ, hãy tự hỏi:** "Nếu tôi chụp màn hình khung hình đó và che hết caption đi, liệu tôi có biết chi tiết này tồn tại chỉ bằng cách nhìn ảnh không?"\
→ Nếu KHÔNG → loại bỏ hoặc thay thế bằng mô tả hình thể thuần túy.\
→ Nếu CÓ → giữ lại.

## Category definitions — hiểu thế nào là GENERAL, thế nào là SPECIFIC

GENERAL và SPECIFIC khác nhau ở **mức độ chi tiết** và **góc nhìn** khi quan sát khung hình:

### GENERAL (3-5 subqueries) — cái nhìn đầu tiên, tổng quan

Những gì bạn thấy ngay khi **liếc qua khung hình trong 1 giây**. Mô tả:
- **Bối cảnh / không gian chung:** trong nhà / ngoài trời, ban ngày / ban đêm, thành phố / nông thôn / biển / sông / đường phố / sân khấu
- **Chủ thể chính, nổi bật:** đám đông, một nhóm người, một người trung tâm, xe cộ, con vật lớn, tòa nhà
- **Vật thể lớn, chiếm nhiều diện tích ảnh:** thuyền, ô tô, sân khấu, cây cối, biển báo lớn
- **Hoạt động chung của cảnh (đã chuyển sang dạng tĩnh):** người trên thuyền, người quanh bàn, xe trên đường
- **Màu sắc chủ đạo / ánh sáng tổng thể:** trời tối, nhiều đèn neon, ánh sáng vàng, trời xanh

Quy tắc: nếu bạn có thể mô tả khung hình đó cho một người bạn chỉ trong 1-2 từ — đó là GENERAL.

### SPECIFIC (2-4 subqueries) — nhìn kỹ, chi tiết riêng biệt

Những gì bạn thấy khi **nhìn kỹ khung hình trong 5-10 giây**. Mô tả:
- **Vật thể nhỏ nhưng đặc biệt:** đồ vật trên tay người, phụ kiện, trang trí, logo, chữ viết
- **Họa tiết, hoa văn, pattern:** quạt in hình cờ, áo có sọc, khăn có hoa văn, mặt nạ có hình thù
- **Màu sắc cụ thể trên một đối tượng cụ thể:** áo đỏ, quạt trắng-đỏ, mũ xanh, khăn vàng
- **Mối quan hệ không gian giữa các vật nhỏ:** cá cột vào tre, quạt trong tay, mũ trên đầu
- **Kết cấu, chất liệu (texture):** gỗ cũ, vải lụa, kim loại sáng bóng, giấy nhăn
- **Chi tiết làm nên sự khác biệt của cảnh này so với cảnh khác cùng loại**

Quy tắc: nếu bạn phải nhìn gần hoặc nhìn lâu mới thấy được chi tiết đó — đó là SPECIFIC.

### Bảng so sánh nhanh

| Góc nhìn | GENERAL | SPECIFIC |
|---|---|---|
| Thời gian quan sát | 1 giây | 5-10 giây |
| Kích thước đối tượng | Lớn, chiếm diện tích | Nhỏ, chi tiết |
| Vai trò | Mô tả bối cảnh & bố cục | Mô tả điểm nhấn riêng |
| Nếu thiếu subquery này | Không hiểu cảnh đang ở đâu | Mất chi tiết đặc trưng |
| Số lượng | 3-5 | 2-4 |

## Ví dụ phân tích chi tiết

### Ví dụ 1
Input:
Đoạn video mô tả về một lễ hội ở Nhật Bản, mọi người tập trung nhảy múa trên một chiếc thuyền. Trên tay của họ cầm những chiếc quạt giấy in hình quốc kỳ Nhật Bản. Phía trên bờ có nhiều người đứng xem, có hai người mặc áo màu đỏ đang khiêng một con cá được cột vào một cây tre.

**Phân tích:**
- Nhìn 1 giây: thấy thuyền, đám đông trên thuyền, người trên bờ → GENERAL
- Nhìn kỹ: quạt in hình cờ Nhật, cá cột vào tre, hai người áo đỏ khiêng → SPECIFIC

Output:
```
## GENERAL
people on a boat
holding paper fans
people watching on the shore
two people in red shirts

## SPECIFIC
paper fans with Japanese flag pattern
a fish tied to a bamboo pole
a pole carried by two people in red
```

### Ví dụ 2
Input:
Một cảnh đường phố Tokyo về đêm với nhiều biển hiệu neon. Có một người phụ nữ mặc kimono đang đi qua ngã tư. Khung cảnh nhộn nhịp, nhiều xe cộ qua lại.

**Phân tích:**
- Nhìn 1 giây: đường phố, neon, người phụ nữ, xe cộ → GENERAL
- Nhìn kỹ: màu sắc biển hiệu cụ thể, kiểu kimono, loại xe → SPECIFIC

Output:
```
## GENERAL
a street with neon signs at night
a woman in kimono
vehicles on the street

## SPECIFIC
multicolored neon signs
a woman in traditional kimono
vehicles at an intersection
```
