# Nguồn và giấy phép

Các liên kết là nguồn gốc, không chứng nhận bộ dữ liệu phù hợp Mobile hoặc nhãn đúng.

| Thành phần | Nguồn | Thông tin cần giữ |
|---|---|---|
| Minimap v2 | https://universe.roboflow.com/enxaiet/efootball-ui-minimap/dataset/2 | Export ghi CC BY 4.0; ghi công tác giả/project/version và mô tả sửa đổi |
| Pitch landmarks v2 | Xem README.roboflow.txt trong export gốc | Export ghi Public Domain; chưa xác minh bảng nghĩa 32 keypoint |
| Field detection v5 | Xem README.roboflow.txt trong export gốc | Export ghi Public Domain; version thực tế là v5 |
| Ultralytics + YOLO11 pretrained | https://github.com/ultralytics/ultralytics ; https://docs.ultralytics.com/ | Giữ thông tin giấy phép upstream trong checkpoint/model; không mặc định weights là permissive |
| ONNX Runtime Android | https://github.com/microsoft/onnxruntime | Giữ thông báo MIT/third-party của artifact phụ thuộc |
| Kotlin, AndroidX, Gradle plugin | Maven/Google Maven theo build.gradle.kts | Giữ giấy phép/thông báo đi kèm khi phân phối |
| PnLCalib | https://github.com/mguti97/PnLCalib | Code GPL-2.0-or-later; released weights CC BY 4.0. Phase 5 chỉ gọi checkout bên ngoài cho teacher nghiên cứu; không nhúng vào APK |
| XFeat | https://github.com/verlab/accelerated_features | Giữ Apache-2.0, thông tin tác giả và thông báo đi kèm khi phân phối code/weights/ONNX dẫn xuất |
| OpenCV Python | https://github.com/opencv/opencv-python ; https://github.com/opencv/opencv | Công cụ CPU nghiên cứu, không thêm dependency NDK vào APK ở lượt này; giữ các thông báo của wheel |

Phase 5 ghim PnLCalib tại `42088864170f75ea02e76d20e9a9a1db59edb3e3`, XFeat tại `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`. SHA trọng số và phạm vi chuyển đổi nằm trong [báo cáo](PHASE5_2026-11-01.md). Không dùng `torch.hub` hoặc tự tải mã khi chạy tracker.

Repo chưa thêm giấy phép bao trùm thay cho quyết định của chủ sở hữu mã hiện có. Không tự cấp phép lại dữ liệu hoặc pretrained weights theo giấy phép riêng của app.
