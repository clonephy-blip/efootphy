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

[Chẩn đoán pose tiếp theo](docs/PHASE3_2026-11-01.md) xác nhận mạng học được tập nhỏ, bổ sung đếm optimizer update, học riêng head với đặc trưng lưu sẵn và kế hoạch chia nguồn. Sau lượt thử đủ 93 ảnh train, validation vẫn đạt 0/92 mốc ở sai số 5 pixel; trọng số thử nghiệm chưa thay baseline trong ứng dụng.

[Thử nghiệm bản đồ nhiệt và chuyển nhãn](docs/PHASE4_2026-11-01.md) học lại trên toàn bộ 93 ảnh, tăng cường ảnh bằng biến đổi có nhãn đi kèm và rà đề xuất từ 2.319 ảnh field train. Đây là nhánh nghiên cứu riêng, dùng encoder cố định và head mới; chưa tích hợp vào Android. Báo cáo giữ cả đối chứng và kết quả train chéo kém hơn.

[Module tự dựng và bám sân](docs/PHASE5_2026-11-01.md) bổ sung teacher PnLCalib, optical flow có kiểm tra hình học, tìm lại sân bằng SIFT/XFeat và runner video có trạng thái `LOST`. Teacher đạt 37/92 mốc trong 5 pixel trên cùng validation nguồn; phép chiếu tự động đạt 12/21 mốc tham chiếu trên ba ảnh Mobile. SIFT + LK đạt 212/213 khung hợp lệ trong replay giả lập từ ảnh tĩnh; đây chưa phải tracking video trận đấu. XFeat CNN đã xuất ONNX và qua parity CPU; chưa được APK nhập hoặc chạy trên điện thoại.

[Lượt tăng tốc và distillation](docs/PHASE6_2026-11-01.md) thêm cache đặc trưng FP32 có kiểm tra hash/parity, lọc teacher hai view trên train và cổng chặn hồi quy chất lượng. Hai lượt 600 update dùng đủ 93 ảnh nguồn; distillation đạt 15/92, đối chứng 14/92, nên giữ checkpoint 17/92. Chuẩn bị đặc trưng giảm 27,86 → 4,87 giây. Checkpoint giữ lại đã xuất ONNX 9,16 MB, giữ nguyên PCK trên 13 ảnh đối chiếu; bản nhỏ vẫn đạt 0/21 mốc Mobile trong 5 px nên chưa được đưa vào app.

```bash
python -m pip install -r tools/vision_qc/requirements-qc.txt
python -W error::ResourceWarning -m unittest discover -s tools/vision_qc -t . -p 'test_*.py' -v
```

GitHub Actions chạy test JVM, test công cụ dữ liệu, Android lint và build APK; báo cáo nằm trong artifacts. Kiểm thử thiết bị, video mù và hiệu năng game là các cổng riêng, chưa được CI này xác nhận.

## Phạm vi phát hành

Không có chức năng tự điều khiển game. Không phát hành gợi ý chiến thuật trước khi qua bộ QC độc lập. Xem [nguồn và giấy phép](docs/THIRD_PARTY.md) trước khi phân phối mã/trọng số; giấy phép bộ dữ liệu không thay thế giấy phép của framework hay trọng số pretrained.
