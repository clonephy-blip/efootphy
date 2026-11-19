# Efootphy chạy train và QC qua Colab CLI

Bộ điều phối gửi công việc từ Antigravity lên Colab, lấy bộ kết quả có checksum rồi giải phóng máy. Mỗi lượt có cấu hình, thư mục kết quả và log riêng. Mã điều phối nằm ở `tools/colab_pipeline`. Workflow cho Antigravity nằm ở `.agent/workflows/efootphy-colab.md`.

**Chuẩn bị trên Windows.** Chạy CLI và bản clone của repo trong Linux qua WSL2. Máy cần có WSL và Python 3.12 trở lên. WSL là môi trường triển khai đề xuất. README upstream công bố Linux/macOS. Xác nhận đăng nhập và cấp GPU trên tài khoản thật là bước còn phải thực hiện.

Sau khi checkout nhánh có bộ điều phối, mở terminal Linux trong thư mục repo và chạy một lần.

```bash
bash tools/colab_pipeline/setup.sh
.venv-colab/bin/colab sessions
cp tools/colab_pipeline/pilot.example.json .colab.private.json
```

Lệnh `sessions` dùng để đăng nhập Google. Chọn đúng tài khoản chứa dự án. Xem hướng dẫn CLI bằng `.venv-colab/bin/colab skill`. Bản CLI 0.8.0 đã được cài và kiểm tra giao diện lệnh trong phiên phát triển này.

**Kiểm tra cấu hình.** Mở `.colab.private.json` và đối chiếu `project_root`, video cùng ba trọng số. Cấu hình mẫu dùng các tên tệp đã thấy trong workspace V2/V5. Nội dung Drive có thể tiếp tục được cập nhật, nên inventory sẽ lưu hash thực tế cho mỗi lượt. `max_minutes` giới hạn phần thực thi các tác vụ trên VM. Thời gian cài thư viện, tạo máy và đóng gói kết quả tính thêm vào thời gian sử dụng Colab.

```bash
.venv-colab/bin/python -m tools.colab_pipeline plan --config .colab.private.json --run-id baseline-001
```

Lệnh `plan` đọc cấu hình cục bộ và có số gọi cloud bằng 0. Mặc định chọn inventory và chạy 300 khung qua V5 với model hiện có.

```bash
.venv-colab/bin/python -m tools.colab_pipeline run --config .colab.private.json --run-id baseline-001 --colab .venv-colab/bin/colab
```

Khi Colab yêu cầu quyền Drive, người dùng xác nhận ở terminal/trình duyệt. Nếu Antigravity chạy trong terminal không tương tác, chương trình trả mã 3, ghi trạng thái `awaiting_drive` và in đúng lệnh `drivemount`. Chạy lệnh được in ở terminal của bạn, rồi chạy lại lệnh `run` với cùng mã lượt. Phiên đó còn được cấp tài nguyên trong lúc chờ.

**Chạy thử train trước khi tăng số epoch.** Lượt mẫu huấn luyện bóng và người một epoch, dùng các hàm chuẩn bị dữ liệu V2 đang có trên Drive. Ảnh được chuẩn bị trong `/content/efootphy-colab/<run-id>/`. Trọng số và báo cáo của lượt nằm trong thư mục riêng dưới `output_root`.

```bash
.venv-colab/bin/python -m tools.colab_pipeline run \
  --config .colab.private.json --run-id pilot-001 \
  --tasks inventory,baseline,ball,actors,candidate,export-ball,export-actors \
  --colab .venv-colab/bin/colab
```

`@ball` và `@actors` tham chiếu checkpoint sinh ra từ tác vụ train đứng trước trong cùng lượt. Các bước bao gồm kiểm tra ảnh/nhãn, kiểm tra thứ tự lớp, train, validation nguồn, replay cùng video, xuất ONNX FP32 và kiểm tra graph/tensor. ONNX parity số học và độ chính xác sau export còn phải đo trước khi đưa vào runtime.

**Bốn khung trong cấu hình.**

| Khung | Tác vụ | Đầu vào |
|---|---|---|
| Bóng | `ball` | Gói BALL_ROI_V1, nguồn field và checkpoint bóng trong workspace V2 |
| Sân | `pitch` | YAML pose đã kiểm tra semantic, shape 18×3 trong ví dụ và checkpoint tương ứng |
| Minimap | `minimap` | YAML ba lớp `ball, controlled, dot` và checkpoint train được |
| Cầu thủ, thủ môn, trọng tài | `actors` | Gói field bốn lớp, thứ tự `ball, goal_keeper, player, referee` |

Điền các giá trị `EDIT_ME` ở pitch/minimap theo dataset thực rồi mới chọn hai tác vụ đó. YAML phải nằm ở gốc của thư mục chứa ảnh và labels của các split. Phép lật được tắt với pose khi semantic còn cần kiểm chứng. Bản audit dùng lại `tools.vision_qc.dataset_qc` để phát hiện nhãn sai, ảnh trùng giữa các split và dữ liệu thiếu. Nhóm trận độc lập phải được ghi từ nguồn thật khi lập bộ nghiệm thu.

**Kết quả và xử lý lỗi.** Mỗi tác vụ chạy trong tiến trình riêng. Worker ghi nhịp tiến độ, dừng ở bước thất bại, và kiểm tra kết quả riêng vì `colab exec` có thể trả mã 0 dù Python trên kernel báo lỗi. Bản V3 master trên Drive có nhánh bắt exception rồi tiếp tục in hoàn tất. Bộ điều phối gọi chuẩn bị dữ liệu, train và benchmark theo từng bước có biên nhận riêng.

Trong `.efootphy-colab/<run-id>/` có `summary.json`, `controller.log`, `state.json` và `artifacts.zip`. Archive được kiểm tra hash từng tệp. Trên Drive có checkpoint, `execution.log`, báo cáo, cấu hình và video. Archive tải về tối đa 200 MiB dữ liệu trước nén. Các tệp lớn còn trên Drive được liệt kê trong `remote_only_files`.

```bash
# Xem trạng thái và compute units của phiên thuộc lượt này.
.venv-colab/bin/python -m tools.colab_pipeline status --config .colab.private.json --run-id pilot-001 --tasks inventory,baseline,ball,actors,candidate,export-ball,export-actors --colab .venv-colab/bin/colab

# Tiếp tục lượt lỗi với đúng cấu hình và thứ tự tác vụ ban đầu.
.venv-colab/bin/python -m tools.colab_pipeline run --resume --config .colab.private.json --run-id pilot-001 --tasks inventory,baseline,ball,actors,candidate,export-ball,export-actors --colab .venv-colab/bin/colab

# Thu lại kết quả đã lưu, dùng một phiên CPU nếu phiên cũ đã dừng.
.venv-colab/bin/python -m tools.colab_pipeline collect --config .colab.private.json --run-id baseline-001 --colab .venv-colab/bin/colab

# Dừng phiên do bộ điều phối đã tạo, kể cả khi đang chờ quyền Drive.
.venv-colab/bin/python -m tools.colab_pipeline stop --config .colab.private.json --run-id baseline-001 --colab .venv-colab/bin/colab
```

Tác vụ hoàn thành được kiểm tra checksum trước khi tái sử dụng. Checkpoint `last.pt` cho phép tiếp tục train sau lỗi khi checkpoint còn trạng thái optimizer. Nếu thiếu ảnh local sau khi đổi VM, bước chuẩn bị dựng lại dữ liệu từ nguồn. Giữ nguyên dữ liệu nguồn và cấu hình khi resume. Khi cần thay dữ liệu hoặc tham số, chọn mã lượt mới.

Mỗi mã lượt có khóa thao tác cục bộ để ngăn hai lệnh chạy đè. Khi terminal bị tắt cưỡng bức, kiểm tra tiến trình và phiên Colab trước khi xóa `operation.lock` đã cũ. Dùng một máy điều phối cho một mã lượt. Chương trình quản lý phiên có tên riêng và tệp metadata riêng.

**Đọc các con số.** Tỷ lệ có quan sát bóng trong CSV và độ dài đoạn `NO_DETECTION` giúp chọn pha cần duyệt. Chúng còn chứa cảnh dừng và các dự đoán có thể sai. Video replay, tốc độ xử lý tệp và giới hạn 300 khung là phép thử offline. Độ bám đúng qua trận độc lập, trễ USB, FPS game và HUD cần log/đo riêng trên thiết bị. `release_eligible` được ghi false cho các tác vụ này.

**Tự động hóa có ích cho Antigravity.** Giao tác nhân đọc `summary.json`, mở đúng log của bước lỗi, tạo một thay đổi có giả thuyết rồi chạy lại cùng bài kiểm tra. Dùng script cho vòng lặp train và thống kê. Hạn mức agent dùng cho phân tích và sửa mã. Compute units Colab dùng cho phiên GPU/CPU. Sau mỗi đợt thử, cập nhật handover bằng kết quả đã đo.

Nguồn giao diện CLI đã đối chiếu ở commit `aa01a39019ed6570c7496ec8ec5330daf1730b33`. Tài liệu upstream có tại [google-colab-cli](https://github.com/googlecolab/google-colab-cli). Các tên file có `60fps` được giữ theo workspace nguồn. Chứng nhận tốc độ cần dựa trên phép đo thực tế.
