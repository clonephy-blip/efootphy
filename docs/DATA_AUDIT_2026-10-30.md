# Audit đầy đủ ngày 2026-10-30

Đã giải mã toàn bộ 2.879 ảnh, kiểm tra cặp nhãn, cấu trúc, tọa độ, lớp và trùng ảnh. Đây là QC định dạng, chưa phải xác nhận nhãn đúng hoặc chất lượng AI.

| Bộ dữ liệu | Train / Val / Test | Lỗi định dạng | Đủ điều kiện test độc lập |
|---|---|---|---|
| Minimap v2 | 229 / 27 / 0 | 0 | Chưa |
| Pitch keypoints v2 | 93 / 10 / 0 | 0 | Chưa |
| Field detection v5 gốc | 2319 / 119 / 82 | 1087 | Chưa |
| Field detection đã chuẩn hóa | 2319 / 119 / 82 | 0 | Chưa |

## Phát hiện cần xử lý

- Field gốc có 1.087 dòng polygon xen lẫn bbox. Đã chuyển min/max polygon → bbox trên bản sao, kiểm tra lại 2.520 ảnh không còn lỗi parser; giữ nguyên số instance từng lớp.
- Có hai cặp ảnh trùng pixel trong train của field, cùng các cặp gần giống giữa split cần xem thủ công. Chưa biết nhóm video gốc nên không gọi test 82 ảnh là test độc lập.
- Minimap 256 ảnh và keypoints 103 ảnh không có test; cả ba bộ thiếu nhóm video để kiểm tra rò rỉ cảnh/trận.
- Minimap chỉ có `ball`, `controlled`, `dot`; không có nhãn đội, ID tracking hoặc chứng minh `ball` là bóng thật.
- Pose là 32×3; thiếu bảng tên điểm → tọa độ sân đã được duyệt. Không suy đoán mapping để chạy AR.

## Instance được đếm từ nhãn

- minimap-dataset: ball=253, controlled=467, dot=4,972.
- landmarks-dataset: pitch=103.
- field-dataset-normalized: ball=2,348, cursor=4,735, cursor_secondary=944, goalkeeper=1,248, player=37,086, powerbar=911, referee=3,825.

## Fingerprint dùng đối chiếu

Fingerprint bao gồm split, đường dẫn và hash ảnh/nhãn; không phải chứng nhận chất lượng.

- `minimap-dataset`: `1603200b187cacc1ac49b7b5b4d5977d64a682323b05caa87aab88aacb054ae3`
- `landmarks-dataset`: `226805ad0c572240e5fcb0d3714d08538defb1cf651f1b27fec7d848607c85f0`
- `field-dataset-raw`: `ccb6d1538a46522aae32e1de9b085e3ed732af95c088eb0075ad25419c574d51`
- `field-dataset-normalized`: `c3ea464c0c7f290bc5db92f390d665214c1297d2eee54e722eeba56fab114fc0`
