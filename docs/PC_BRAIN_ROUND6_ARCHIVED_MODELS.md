# efootphy — Lượt 6: benchmark checkpoint thực lưu trong Library (05/11/2026)

**Trạng thái: đã chạy benchmark offline trên 3 file ONNX thật. KHÔNG promote model; KHÔNG xác nhận 60 FPS hay độ chính xác trên máy PC của người dùng.**

## Nguồn và hợp đồng checkpoint

Các tệp dưới đây đã được giải nén trực tiếp từ Library và mở bằng OpenCV DNN, không suy diễn từ tên tệp.

| Mục | ZIP gốc / ONNX bên trong | SHA256 của ONNX | Input → output | Nhiệm vụ |
|---|---|---|---|---|
| Round2 ball 320 | efootphy-round2-trained-checkpoint.zip / run/weights/best.onnx | a05e725de2bc6e043733f6a2c84709b384f661b00d2d7aabadda4ed5029afbd8 | [1,3,320,320] → [1,11,2100] | Fine-tune **chỉ class 0 (ball)** |
| Round3 ball 320 | efootphy-round3-trained-checkpoint.zip / model/best.onnx | ba108ac54736c41e6cdbef1db25e8e06ba05d97e5195566d2b67349bccdc2fd6 | [1,3,320,320] → [1,11,2100] | Fine-tune **chỉ class 0 (ball)** |
| Field source 640 | efootphy-field-research-2026-11-01.zip / bundle/model.onnx | 3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6 | [1,3,640,640] → [1,11,8400] | Field **7 class**, 1 epoch nghiên cứu |

**Quan trọng:** Hai ONNX 320 kể trên **chưa được xác minh là file** \`tools/models/field_yolo11n_320.onnx\` của PC Brain Windows hiện tại. Chúng có 7 kênh đầu ra nhưng quá trình fine-tune ưu tiên class bóng, không phải detector field hoàn chỉnh.

## Phép đo thực tế

Nguồn video: 38 frame nhìn rõ bóng, đã được trợ lý rà điểm tham chiếu, từ các clip \`MATCH_01_PART_010\` trong \`efootphy-round2-source-clips.zip\`; tọa độ tham chiếu nằm trong \`efootphy-round3-qc-2026-11-03.zip/r2-coarse-fullframe.json\`. Các frame từ file MP4 đã nén lại, độ phân giải nguồn giữ 2340×1080. Các mẫu này đã được sử dụng khi nghiên cứu trước, **không phải independent-match blind test**.

Mỗi model được đưa **toàn bộ màn hình game**, resize **stretch** đúng kích thước input, RGB/255. Sử dụng OpenCV DNN CPU trên cùng môi trường. Ngưỡng đề xuất bóng \`0.08\`, NMS IoU \`0.50\`, top-1 được tính là đúng khi sai số tâm bóng ≤10 px.

| Checkpoint | Bóng top-1 đúng / 38 | Frame có ứng viên bóng | CPU inference p50 | CPU inference p95 | Ảnh menu/chuyển cảnh có bóng giả |
|---|---:|---:|---:|---:|---:|
| Round2 ball 320 | **4/38** | 4/38 | **21.2 ms** | 23.7 ms | 0/10 |
| Round3 ball 320 | **2/38** | 2/38 | **21.5 ms** | 26.8 ms | 0/10 |
| Field source 640 | **25/38** | 33/38 | **79.4 ms** | 97.8 ms | 0/10 |

Tất cả kết quả trên là **suy luận full-frame**, không có crop/reacquisition. Tại cách giải mã mới (ball channel độc lập và classwise NMS), số khung **top-1** vẫn như bảng trên. Với field 640, có ứng viên bóng đúng ở **27/38** frame, nhưng top-1 chỉ đúng **25/38**; không được tính 27 là tỉ lệ tracking. Cách xử lý cũ thật chỉ trả về một bóng nên bất kỳ ứng viên đúng = top-1 đúng (25/38 ở checkpoint 640).

Sai số lớn vẫn tồn tại: **13/38** khung checkpoint field 640 chọn sai hoặc không tìm thấy bóng. \`val-00\` bỏ sót **4/4** frame. Không có ground-truth âm tính trong gameplay; 10 ảnh menu/chuyển cảnh không đủ chứng minh false-positive rate.

## Ý nghĩa đối với dự án

1. Không thể kết luận cứ 320 là kém 640: ba checkpoint có **weights và quy trình train khác nhau**. Đây không phải thí nghiệm ablation chỉ thay resolution.
2. Round2/3 ball 320 trước kia dùng crop, native reacquisition và cơ chế chọn bóng theo thời gian. Đưa thẳng toàn bộ 2340×1080 vào input 320 làm bóng rất nhỏ, nên kết quả 4/38 hoặc 2/38 không phủ nhận các kết quả crop trước đây.
3. Field 640 có số khung thấy bóng cao hơn trên nhóm mẫu này nhưng suy luận CPU chậm khoảng **3.7×** theo median. Đây là số liệu của môi trường thử hiện tại, **không phải tốc độ trên PC của người dùng**.
4. Không thay \`field_yolo11n_320.onnx\` của PC Brain bằng bất kỳ checkpoint ball-only nào chỉ vì file cũng có 7 output channels. Shape đúng không bảo đảm class khác còn nhận diện tốt.
5. Muốn có cơ sở quyết định nên giữ 320 hay 640, cần **chính file ONNX 320 được PC Brain load**, rồi chạy \`compare_field_models.py\` trên cùng clip và chạy PC perf thật. Sau đó kiểm tra chuỗi liên tục, bóng bổng \`PART_005\`, cursor và hai đội.

## Chạy lại phép thử mà không train lại

Cài Python + thư viện trong \`tools/vision_qc/requirements-qc.txt\`, rồi đặt bốn ZIP nghiên cứu và hai ZIP validation trong thư mục riêng. Chạy từ gốc repo:

\`\`\`powershell
py -m tools.pc_brain.compare_saved_checkpoints \`
  --round2-zip "C:\Private\efootphy-round2-trained-checkpoint.zip" \`
  --round3-zip "C:\Private\efootphy-round3-trained-checkpoint.zip" \`
  --field-zip "C:\Private\efootphy-field-research-2026-11-01.zip" \`
  --source-clips "C:\Private\efootphy-round2-source-clips.zip" \`
  --qc "C:\Private\efootphy-round3-qc-2026-11-03.zip" \`
  --out "C:\Private\round6-research.json"
\`\`\`

Công cụ \`compare_saved_checkpoints.py\` xác minh SHA256, đối chiếu manifest của Round2/3 chỉ huấn luyện class 0, chạy lại từng model bằng benchmark của lượt 4 và từ chối tự promote. **Không commit các ZIP/video/checkpoint vào repo GitHub công khai.**

## Gate để chuyển sang triển khai

- **Blocker 1:** Model thực tế đang nằm trên Windows PC chưa được cung cấp. Tên dự kiến: \`field_yolo11n_320.onnx\`; phải đối chiếu SHA với danh sách model đã biết, không đoán theo tên.
- **Blocker 2:** Thiếu validation gồm ground-truth player/referee/cursor/team/controlled-player và bóng vắng mặt khi gameplay thực, đồng thời thiếu trận độc lập.
- **Blocker 3:** Chưa có p50/p95 tổng đường truyền scrcpy → PC inference → Android HUD hoặc tiêu chí 60FPS đo trên thiết bị.

**Quyết định:** giữ nguyên baseline PC Brain, không train thêm, không đổi model chính và không merge tự động; nghiên cứu đã xác nhận phải tách \`field global detector\` và \`ball crop refinement\`, không thay vai trò của chúng bằng các ONNX 320 cùng kích thước.
