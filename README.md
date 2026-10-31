# eFootPhy · eFootball vision research

Ứng dụng Android đọc ảnh màn hình do người dùng cho phép, nhận diện radar bằng mô hình ONNX nhập từ tệp, và hiển thị trạng thái theo dõi. Đây là bản nghiên cứu; chưa công bố độ chính xác Mobile, chưa cam kết giữ game 60 FPS và chưa mở gợi ý chuyền bóng.

## Phần đã triển khai

- `app`: foreground service, MediaProjection Android 8+, bubble kéo thả, lớp vẽ xuyên chạm, chọn vùng radar, nhập gói ONNX có SHA-256 và kiểm tra tensor. Capture lấy ảnh mới nhất, giới hạn tối đa 15 lần/giây; suy luận chạy trên worker CPU một thread.
- `vision-core`: tọa độ sân chuẩn hóa, homography/RANSAC, ghép minimap–sân bằng Hungarian một–một, theo dõi theo thời gian thực, hết hạn quan sát và khóa gợi ý khi thiếu bằng chứng. Các kiểm thử dùng dữ liệu tổng hợp có đáp án; chúng không đo chất lượng AI trên game.
- `tools/vision_qc`: audit toàn bộ export, phát hiện nhãn lỗi/trùng/lọt nhóm video, chuyển polygon sang bbox trên bản sao, train baseline, xuất ONNX và kiểm tra sai số với PyTorch.

**Runtime hiện kết nối mô hình minimap. Detector cầu thủ trên sân, detector 32 mốc sân, phân đội và hiệu chỉnh radar chưa được nối vào capture.** Lõi có hợp đồng nhận dữ liệu cho các nhánh này; không gọi trạng thái đó là hệ thống fusion đã kiểm chứng. Chấm radar hiện giữ đội `UNKNOWN`; nhãn `ball` của bộ minimap giữ vai trò marker, chưa được coi là quả bóng thật.

## Chạy ứng dụng

Java 17, Android SDK 35, Gradle 8.9:

```bash
gradle :vision-core:test :app:lintDebug :app:assembleDebug
```

APK không nhúng trọng số. Chỉ nhập ZIP **profile minimap** sinh bởi công cụ train (`model.onnx` và `manifest.json` ở gốc ZIP); gói field/landmark nghiên cứu chưa được APK hỗ trợ. Cấp quyền cửa sổ nổi, bật trợ lý, chuyển sang game rồi chạm **BẬT QUÉT**. Cho phép chia sẻ **toàn màn hình**; chia sẻ riêng một app/cửa sổ chưa được hiệu chỉnh tọa độ trong bản này. Chỉnh bốn thanh vùng radar theo điện thoại. Dừng từ bubble hoặc thông báo.

Các marker xuất hiện trên radar phụ; không vẽ vào vùng ảnh radar đang đọc. Lớp vẽ tự ẩn sau 250 ms khi mất kết quả. Đổi ROI/đổi phiên capture xóa trạng thái cũ. Cần kiểm tra thực tế status bar, cutout, thanh điều hướng và xoay màn hình trên từng thiết bị.

## Dữ liệu và huấn luyện

Xem [quy trình](docs/TRAINING.md), [kiến trúc và giới hạn](docs/ARCHITECTURE.md), [tiêu chuẩn QC](docs/QUALITY.md) và [audit đã thực hiện](docs/DATA_AUDIT_2026-10-30.md) và [kết quả baseline thực đo](docs/BASELINE_2026-10-30.md). Dữ liệu ảnh/video của người dùng và trọng số không đưa vào Git.

[Lượt field/pose ngày 01/11/2026](docs/PHASE2_2026-11-01.md) đã có trọng số và QC thực: AP50 cầu thủ 94,04%, bóng 13,19% trên 82 ảnh test nguồn; mốc sân đạt 0/92 điểm ở ngưỡng sai số 10 pixel trên validation. Các split chưa độc lập theo trận/video; bóng và mốc sân chưa đạt để mở trợ lý chiến thuật.

```bash
python -m pip install -r tools/vision_qc/requirements-qc.txt
python -W error::ResourceWarning -m unittest discover -s tools/vision_qc -t . -p 'test_*.py' -v
```

GitHub Actions chạy test JVM, test công cụ dữ liệu, Android lint và build APK; báo cáo nằm trong artifacts. Kiểm thử thiết bị, video mù và hiệu năng game là các cổng riêng, chưa được CI này xác nhận.

## Phạm vi phát hành

Không có chức năng tự điều khiển game. Không phát hành gợi ý chiến thuật trước khi qua bộ QC độc lập. Xem [nguồn và giấy phép](docs/THIRD_PARTY.md) trước khi phân phối mã/trọng số; giấy phép bộ dữ liệu không thay thế giấy phép của framework hay trọng số pretrained.
