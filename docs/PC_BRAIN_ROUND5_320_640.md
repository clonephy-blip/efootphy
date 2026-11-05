# eFootPhy — Lượt 5: So sánh field model 320 ↔ 640 (05/11/2026)

## Trạng thái kiểm tra

**Mục tiêu:** so sánh hai file ONNX thực tế (không đổi kích thước ảnh rồi tự gọi là model 320). Phân biệt chất lượng nhận diện bóng, thời gian chạy và các trường hợp khác nhau theo từng frame.

| Hạng mục | Trạng thái |
|---|---|
| Đã xác minh model field 640 | Có — efootphy-field-research-2026-11-01.zip |
| Input/output model 640 | [1,3,640,640] / [1,11,8400] |
| SHA256 model 640 | 3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6 |
| Checkpoint field 320 đang dùng ở Windows | **Chưa có trong GitHub hoặc các bản nghiên cứu đã kiểm tra** |
| Dữ liệu validation video PART_010 | Có 38 frame thấy bóng được rà (nhưng chưa được gán nhãn độc lập) |
| Đối chứng 320 ↔ 640 thực | **Chưa thực hiện — thiếu file model 320** |
| Thay model / train thêm | **Không thực hiện** |

Không sử dụng model 640 sửa đầu vào giả lập thành 320 và gọi là model 320 thật. Không dùng model minimap/ball-only (cũng có thể 320) như một model field 7-class.

## Phát hiện quan trọng từ đối chứng thăm dò

Với checkpoint field 640, nghiên cứu lượt 4 ghi được top-1 ball hit ≤10px: **25/38** ở cả hai cách hậu xử lý; 13/38 frame chưa chọn được bóng đúng trong 10px. Theo clip, riêng val-00 trượt **4/4**. Những số này là từ phép thăm dò trên video PART_010 cùng họ trận đấu và chưa phải benchmark độc lập. Một kết quả phụ về nhiều ứng viên bóng cũ (**26/38**) đã được **rút lại** do không tương thích đường xử lý cũ, vốn chỉ trả về một bóng top-1. Xem bản đính chính trong docs/PC_BRAIN_ROUND4_REAL_VIDEO.md.

Nguồn test Roboflow của checkpoint 640 ghi ball AP50 13.19%, player AP50 94.04%, cursor AP50 76.83%. Không sử dụng AP nguồn để suy ra AP trên video Mobile.

## Hướng dẫn Windows — chạy ở máy có model 320 thật

Thực thi tại thư mục gốc clone efootphy đã checkout nhánh này. Giữ các ZIP từ Library và hai ONNX **trên máy cá nhân**; KHÔNG commit dữ liệu game, model hay output benchmark lên repo GitHub công khai.

Cài các thư viện benchmark:

\`\`\`powershell
py -m pip install -r tools/vision_qc/requirements-qc.txt
\`\`\`

Kiểm tra riêng model 320 (không cần model 640 để bắt đầu):

\`\`\`powershell
py -m tools.pc_brain.benchmark_field_video \`
  --model "C:\efootphy-private\field_yolo11n_320.onnx" \`
  --input-size 320 \`
  --source-clips "C:\efootphy-private\efootphy-round2-source-clips.zip" \`
  --qc "C:\efootphy-private\efootphy-round3-qc-2026-11-03.zip" \`
  --out "C:\efootphy-private\round5-field-320.json"
\`\`\`

Khi có model 640 từ archive nghiên cứu, kiểm định cùng 38 frame rồi tạo báo cáo ghép:

\`\`\`powershell
py -m tools.pc_brain.compare_field_models \`
  --model-320 "C:\efootphy-private\field_yolo11n_320.onnx" \`
  --model-640 "C:\efootphy-private\field640.onnx" \`
  --source-clips "C:\efootphy-private\efootphy-round2-source-clips.zip" \`
  --qc "C:\efootphy-private\efootphy-round3-qc-2026-11-03.zip" \`
  --sha-640 3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6 \`
  --out "C:\efootphy-private\round5-320-vs-640.json"
\`\`\`

Gói model 640 nghiên cứu: giải nén **bundle/model.onnx** thành field640.onnx. Tuỳ chọn thêm \`--sha-320\` sau khi tính được SHA256 model riêng của người dùng. SHA bản 320 chưa biết; **không được đoán**.

Lưu ý: Nếu chạy Python trong cmd.exe thay vì PowerShell, thay dấu nối dòng backtick bằng dấu ^ hoặc dùng một dòng duy nhất.

## Bộ tiêu chuẩn so sánh

Script \`compare_field_models.py\` dùng chung runner \`benchmark_field_video.py\` để đọc đúng cùng clip/PTS, threshold, hậu xử lý và phép đo. Nó:
- Từ chối chạy nếu thiếu 320 hoặc 640, hình dạng dự đoán không khớp, hash không khớp tuỳ chọn hoặc dữ liệu tham chiếu thay đổi.
- Đếm các frame **640 bắt được mà 320 bỏ sót**, ngược lại và cả hai cùng đúng/sai.
- Báo riêng top-1 và "có ứng viên đúng", tránh nhầm lẫn trong cách đếm.
- Tách kết quả theo clip, liệt kê frame bất đồng.
- Ghi p50/p95 inference CPU; KHÔNG coi đây là độ trễ tổng scrcpy → HUD.
- Luôn đánh dấu \`eligible_for_auto_promotion=false\`.

**Không chọn model chỉ vì mAP cao hơn trong số ít khung đã được xem khi nghiên cứu.** Bộ PART_010 từng dùng chọn checkpoint, các clip cùng MATCH_01 nên test này chỉ có ý nghĩa đối chứng kỹ thuật, không phải nghiệm thu độc lập.

## Cổng cần vượt qua trước khi phát hành

1. Chạy cả hai ONNX thật trên PC Windows của người dùng, ghi SHA256 và cùng thiết lập phần cứng.
2. Chấm đủ ball-positive **và ball-negative trong gameplay**; nhãn người/referee/cursor được review, không đếm ứng viên như true positives.
3. Đánh giá chuỗi video theo PTS, ID và bóng bổng của PART_005. Đảm bảo không dùng PART_010 làm train và không sửa ngưỡng sau khi nhìn test.
4. Đo FPS, P95 latency, USB RTT và tác động vào frame-time game trên thiết bị Android.
5. Chỉ promote model có cải thiện bền vững, không làm tăng false positive và đạt yêu cầu hiệu năng; phiên bản chính vẫn giữ nguyên cho tới lúc đó.

## Kết luận lượt 5

Hoàn thiện **hạ tầng đo và cổng QC**, chưa thể xác nhận độ chính xác của **model 320 đang dùng thật** khi chưa có file/hardware tương ứng. Không tạo thêm checkpoint, không claim 60 FPS, không merge nhánh nghiên cứu tự động.
