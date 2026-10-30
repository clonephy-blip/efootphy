# Cổng nghiệm thu

Các con số dưới đây là **mục tiêu đề xuất**, không phải kết quả đạt được. Test logic, validation của dataset gốc và nghiệm thu trên eFootball Mobile là ba loại bằng chứng khác nhau.

## Bộ đánh giá phải đóng băng trước khi tinh chỉnh

- Ít nhất 10 trận Mobile độc lập, bao phủ ba điện thoại/độ phân giải, hai chiều tấn công, màu áo/radar, ánh sáng và đổi camera. Ba clip cắt từ một trận chỉ là một nhóm.
- Chia train/val/test theo trận/video gốc, trước augmentation; không chia ngẫu nhiên từng frame. Ghi SHA-256 tệp, nhóm, phiên bản nhãn và mã/model được đo.
- Nhãn cầu thủ trên ảnh, vị trí radar, đội, ID liên tục, bóng thật và trạng thái quan sát được. Có negative scenes: menu, game plan, replay, ăn mừng, pause và radar bị che.
- Hai người kiểm tra các ID/đội/mốc sân khó, xử lý bất đồng trước khi chấm. Không dùng chính model cần đo để tạo “ground truth”.

## Mục tiêu phải đo

| Cổng | Đề xuất ban đầu | Bằng chứng bắt buộc |
|---|---|---|
| Chấm cầu thủ radar | Precision/recall ≥97% trên phần được gán nhãn thấy rõ | TP/FP/FN, PR theo lớp, khoảng tin cậy bootstrap theo trận |
| Phân đội | ≥99% trong phần app nhận phân đội; báo riêng tỷ lệ UNKNOWN | Confusion matrix + coverage; không bỏ UNKNOWN để làm đẹp số |
| Ghép radar ↔ sân | Precision ghép ≥99% | Nhãn cùng người, cùng timestamp; báo recall và lỗi ghép nhầm |
| Tracking | IDF1 ≥90%, báo ID switches và thời gian mất dấu | Chuỗi ID ground truth liên tục, không đánh giá trên ảnh rời |
| Camera | P95 reprojection ≤3 pixel ảnh gốc trên các điểm giữ lại để chấm | Không dùng chính các inlier fit để thay cho đánh giá độc lập |
| Cảnh không thi đấu | Không phát gợi ý trong toàn bộ negative suite đã đóng băng | Đếm cảnh/frame lỗi, độ dài âm tính; báo cận trên xác suất lỗi |
| Độ trễ | P95 từ timestamp frame tới HUD ≤150 ms, không có backlog | Trace thiết bị 20 phút, ít nhất ba lượt/điện thoại |
| Ảnh hưởng game | Tăng tỷ lệ frame trễ ≤2 điểm % so với baseline cùng máy | Capture game trước/sau, nhiệt độ, pin, CPU/RAM, điều kiện giống nhau |
| Lifecycle | Không crash/rò tài nguyên khi xoay, dừng, khóa máy và cấp lại quyền | Android 8/10/12/14/15; log thao tác và soak test |

Không buộc số người luôn bằng 22: thẻ đỏ, che khuất, marker trùng và mất radar phải được mô tả đúng. Đếm đủ 22 cũng không chứng minh các ID/đội/vị trí đúng.

`tools.vision_qc.release_gate` nhận báo cáo độc lập và chặn dữ liệu thiếu, không hữu hạn, metric kém hoặc model/hash/bộ test không khớp. Nó hỗ trợ quyết định nghiên cứu; không tự bật `modelsValidated` trong APK. Cần review bằng chứng và thay đổi mã có kiểm tra khi mở tính năng.

## QC đang có

- CI test thuật toán bằng đáp án tổng hợp: hình học, outlier, ghép toàn cục, timestamp/epoch, mất dấu, không nhân đôi 22 người, cổng unknown, letterbox và NMS đa lớp.
- CI test công cụ bằng dataset nhỏ có lỗi chủ động: nhãn pose/detect sai, ảnh hỏng, thiếu export, trùng cross-split, nhóm video tràn split, bảo toàn nhãn khi normalize.
- Import model xác minh checksum và shape tensor. ONNX/PyTorch parity trên ảnh validation đo sai số số học; không thay thế kiểm tra preprocessing và runtime trên Android.
- `docs/DATA_AUDIT_2026-10-30.md` ghi QC dữ liệu thật. Báo cáo chi tiết/ảnh/video không được đưa vào repo công khai.

Chưa có video Mobile đủ nhãn ID/mốc sân để nghiệm thu các mục tiêu trên. Đang khóa gợi ý chiến thuật vì lý do này.
