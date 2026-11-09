# eFootPhy · PC vision + Android HUD

**PC đọc màn hình game Android qua USB/scrcpy, chạy AI và tracking. Android chỉ nhận kết quả để vẽ HUD.** Đây là nhánh nghiên cứu; chất lượng bóng, hiệu chỉnh sân và hiệu năng USB thực tế còn các bước QC riêng.

AI tiếp tục dự án cần đọc [bàn giao](docs/handover/START_HERE.md), [đúng thư mục dữ liệu](docs/handover/DATASETS.md), [nhật ký](docs/handover/JOURNAL.md) và [trạng thái hiện tại](docs/handover/CURRENT_STATE.json). Nhánh làm việc: `work/pc-hud-shadow-20261109`; [PR #9](https://github.com/clonephy-blip/efootphy/pull/9). Mỗi thay đổi phải cập nhật nhật ký cùng mã.

| Thành phần | Vai trò hiện tại |
|---|---|
| `tools/pc_brain` | Windows chụp cửa sổ scrcpy, chạy model field ONNX, phân nhóm màu áo, gửi tọa độ; hỗ trợ chọn CPU/CUDA/DirectML theo provider đã cài. |
| `app` | Nhận JSON qua ADB forward, HUD xuyên chạm và bubble kéo thả. Không phụ thuộc ONNX Android, không chạy MediaProjection. |
| `tools/pc_brain/ball_shadow.py` | Đề xuất bóng đổ, mặc định tắt; hàm hình học 3D chỉ nhận camera/ánh sáng đã hiệu chỉnh. Chưa xác nhận độ chính xác trên game. |
| `tools/vision_qc` | Audit nhãn, huấn luyện/đánh giá, notebook Colab và công cụ video nghiên cứu. |
| `legacy/android-local` | Mã quét minimap trên điện thoại cũ, được giữ để tham khảo, không nằm trong APK. |
| `vision-core` | Thư viện hình học và kiểm thử JVM trước đây; hiện chưa kết nối vào PC runtime. |

**Chạy thử:** làm theo [RUN_PC.md](docs/handover/RUN_PC.md). Model field nằm trên PC; APK không nhận gói model. scrcpy chạy `--no-control`, không phát lệnh điều khiển game. Bộ minimap mới sẽ được nối vào PC sau; hiện runtime chưa hợp nhất minimap với sân.

**Kết quả hiện có:** kiểm tra cùng 38 khung hình cho ball top-1 đúng trong 10 pixel: Field 640 cũ 25/38, Field 320 mới 11/38. Model 32 mốc sân mới vẫn không đạt để triển khai hình học. Chi tiết và giới hạn phép đo nằm trong nhật ký. Không coi FPS vẽ HUD là FPS AI hoặc FPS game.

Build với Java 17, Android SDK 35, Gradle 8.9:

```bash
gradle :vision-core:test :app:lintDebug :app:assembleDebug
python -m pip install -r tools/vision_qc/requirements-qc.txt -r tools/vision_qc/requirements-colab-qc.txt
python -m unittest discover -s tools/pc_brain -t . -p 'test_*.py' -v
python -W error::ResourceWarning -m unittest discover -s tools/vision_qc -t . -p 'test_*.py' -v
```

GitHub Actions lưu APK và báo cáo. Đọc [nguồn/giấy phép](docs/THIRD_PARTY.md) trước khi phân phối. Các tài liệu `PHASE*`, `ARCHITECTURE.md` và báo cáo cũ là lịch sử nghiên cứu; kiến trúc mới nhất nằm trong `docs/handover`.
