# Huấn luyện baseline có thể tái lập

Python 3.12. Cài PyTorch theo hướng dẫn chính thức phù hợp CPU/CUDA của máy, sau đó:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/vision_qc/requirements-train.txt
python -m tools.vision_qc.dataset_qc --help
python -m tools.vision_qc.normalize --help
```

Xuất toàn bộ Roboflow YOLOv8 labels + ảnh, giữ `data.yaml` và README. `--expected-images` dùng số ảnh của đúng version để phát hiện tải thiếu. Công cụ audit hỗ trợ detect bbox, pose 32×3; polygon phải normalize trên bản sao. Kết quả geometry của polygon được thu về min/max bbox; không giữ segmentation mask cho task detection.

```bash
python -m tools.vision_qc.normalize datasets/field-raw datasets/field-detect
python -m tools.vision_qc.train \
  --data datasets/minimap/data.yaml --profile minimap --expected-images 256 \
  --research-only --epochs 60 --patience 15 --batch 16 --lr 0.001 \
  --device cpu --threads 4 --out training-runs/minimap-v1
```

Kiểm tra `--help` của normalize để dùng đúng tham số nếu lấy script ở phiên bản khác. Thư mục output train phải mới/rỗng. Với máy CPU bị hạn chế kernel FP16, thêm `--portable-cpu`: chuyển checkpoint pretrained sang FP32 bằng NumPy và lưu EMA/optimizer FP32. Không thay đổi loss/optimizer upstream; chậm hơn backend CPU tối ưu. Không dùng chế độ này để giả số đo Android.

Runner dùng seed 20261030, không mosaic/mixup; pose không flip khi chưa xác minh nghĩa 32 điểm. AdamW có warmup và LR rõ ràng; không suy diễn chất lượng từ việc epoch đã chạy. Gói xuất lưu dataset fingerprint, phiên bản runtime, metrics nguồn, SHA-256 và kết quả parity. `models_validated` và `release_eligible` luôn false trong baseline.

Profile `field` cần dữ liệu detect đã chuẩn hóa, bảy lớp đúng thứ tự. Profile `landmarks` cần pose `kpt_shape:[32,3]`, lớp `pitch`; trước khi dùng để chiếu sân phải duyệt mapping ID → tọa độ sân trên nhiều camera. Không có file mapping đã duyệt thì không nối pose vào AR.

Cần bổ sung `groups.json` dạng `{ "train/images/file.jpg": "match_001", ... }` từ nguồn thật; audit sẽ chặn nhóm xuất hiện ở nhiều split. Không tự đoán video group từ tên ảnh rồi coi là đã kiểm chứng. `--research-only` cho phép nghiên cứu khi thiếu test độc lập, không bỏ qua dữ liệu lỗi.

## Tệp đầu ra

- `dataset-qc.json`: toàn bộ ảnh/nhãn và fingerprint.
- `run/results.csv`, `run/weights/{best,last}.pt`: quá trình học và checkpoint.
- `validation/`: kết quả validation của nguồn, có thể lạc quan khi video group chưa tách.
- `model-bundle.zip`: `model.onnx` + `manifest.json`; app hiện chỉ nhập profile minimap 320, FP32 raw `[1,7,2100]`.

Không đưa output training, dữ liệu người dùng, token Roboflow hay video vào Git. Kiểm tra giấy phép của dữ liệu, pretrained weights và Ultralytics trước phân phối.

Môi trường CPU đã chạy thành công được ghi ở `tools/vision_qc/requirements-train-cpu.lock.txt`; GPU dùng phiên bản torch/torchvision phù hợp máy và ghi lại phiên bản thực tế trong manifest. Các số thực đo của baseline xem `BASELINE_2026-10-30.md`.

`tools/runtime_qc/OnnxParity.kt` kiểm tra cùng buffer FP32 little-endian qua Java ORT; dùng đúng phiên bản ORT trong app. So sánh này không đo Android Bitmap preprocessing hoặc tốc độ thiết bị.
