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
# Bounded field and landmark experiments

`train.py` accepts `--max-minutes` and stops after a saved epoch when the next
epoch would exceed the estimated budget. Setup, final validation and ONNX export
are outside that estimate. `progress.json` records actual completed epochs,
elapsed time, metrics and checkpoint existence; requested epochs are not completed
epochs. Create `OUT/STOP_AFTER_EPOCH` for a graceful stop.

`--no-plots` skips optional upstream training/final-validation charts without
changing loss, numerical validation or checkpoint selection. Use this when label
plotting stalls on a restricted host. Polars threads are capped before import.
Field ONNX evaluation is numerical by default; `evaluate_field --plots` enables
its optional charts. Source annotation totals include exact duplicate rows; QC
warns about these because the training loader removes duplicate annotations.

`--weights /path/to/last.pt` starts from saved weights with a **new optimizer**;
it does not resume optimizer/scheduler state. Use a new output directory. Portable
CPU checkpoints use atomic replacement so interrupted writes do not overwrite the
previous complete checkpoint. Save the resulting model package persistently at
each completed stage, before starting another long experiment.

On restricted CPU hosts, `--portable-cpu --cpu-backend onednn` enables FP32
oneDNN kernels while retaining the FP32 checkpoint conversion workaround.
`generic` remains available. Threads are reapplied after Ultralytics device
selection. Run heavy jobs sequentially on a memory-limited host; benchmark values
from this machine do not establish Android latency or game FPS.

For 32-point pitch training, `--pose-train-sigma 0.1` explicitly changes the
**training loss** only. Validation OKS sigmas remain upstream defaults. This is an
experiment setting, not an acceptance-metric adjustment. Translation, scale,
flips, rotation, perspective and mosaic are disabled for pitch landmarks until
their coordinate semantics are validated. Detection augmentation remains separate.

Reviewed label edits use `python -m tools.vision_qc.reviewed_keypoints --source
SOURCE --out NEW_COPY --review REVIEW.json`. Each correction names a source label,
its SHA-256, an explicit swap/missing operation and a human review reason. All edits
are validated before copying. Original data stays unchanged. Model predictions
must not be used as ground truth. One review is not independent label certification.

The Android model importer currently accepts the minimap detector only. Field
and landmark bundles are research exports; training or exporting them does not
connect those branches to Android capture or enable tactical advice.

Use `python -m tools.vision_qc.landmark_diagnostics --data DATA.yaml --out NEW_DIR`
to inspect per-ID visible support and centre-label consistency. ID 15 in this
source export is the centre spot. This must not be assumed from a generic
32-point pitch template. IDs 13/16 and 30/31 define two intersecting pitch lines;
their intersection gives a source-label consistency check, not model accuracy.
No metre coordinates or calibrated camera are certified by this check.

Use `python -m tools.vision_qc.evaluate_landmarks --bundle RUN/bundle --data DATA.yaml
--split val --out NEW_DIR` to measure PCK at 3/5/10 **original-image pixels**.
Missing object detections and low-confidence points remain misses in the PCK
denominator. Unsupported IDs return null rather than an invented score.
Object-matched coordinate errors and unfiltered PCK are diagnostic supplements.
The contact sheet uses yellow ground truth and cyan predictions.

If upstream `best.pt` and `last.pt` differ, use `export_checkpoint` to validate
and export the second checkpoint, then evaluate both on validation only. For the
bounded landmark recovery experiment, select by validation PCK@5, then PCK@10,
then median pixel error; use later epoch as a final tie-breaker. Record both
results and the chosen checkpoint. This is model selection on a tiny source
validation set, not independent acceptance. Do not select checkpoints using test
images. Detection continues to use the upstream source-validation fitness.

