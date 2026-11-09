# Chạy bản PC + USB + Android HUD

Đây là bản thử nghiệm của nhánh `work/pc-hud-shadow-20261109`, không phải APK quét màn hình trên điện thoại trước đây.

## Chuẩn bị một lần

1. Lấy APK từ lần GitHub Actions **thành công đúng commit của PR #9**. Cài lên Android, mở ứng dụng, bấm **BẬT HUD**, cấp quyền hiển thị trên ứng dụng khác rồi quay lại bấm **BẬT HUD**. Dừng được từ ứng dụng hoặc thông báo.
2. PC Windows cần Python 3.12, scrcpy + adb, và model **field** ONNX riêng. Giữ model minimap/pitch ở thư mục riêng. Hash model tham chiếu nằm trong `CURRENT_STATE.json`.
3. Trong PowerShell tại thư mục mã nguồn, tạo môi trường Python. Đây là cài đặt PC, không chạy trên Android:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy opencv-python-headless mss Pillow onnxruntime
```

Đây là cấu hình CPU ban đầu, chưa phải bộ phiên bản runtime Windows đã chứng nhận. GPU cần package/provider phù hợp từ [hướng dẫn ONNX Runtime](https://onnxruntime.ai/docs/install/). Chỉ cài một biến thể ORT trong một môi trường. Chọn `cuda` hoặc `dml` chỉ khi môi trường đã có provider tương ứng; yêu cầu GPU thiếu provider sẽ báo lỗi. `auto` dùng provider khả dụng và in tên thật ra terminal; có tên GPU chưa chứng minh toàn bộ graph chạy trên GPU.

## Mỗi lần chạy

1. Cắm **một điện thoại USB**, bật USB debugging, chấp nhận hộp thoại trên điện thoại. Mở game nằm ngang, bật HUD.
2. Đóng cửa sổ `eFootball-Screen` cũ để chỉ có một cửa sổ. Giữ cửa sổ mới không bị che hay thu nhỏ. Bản hiện tại chụp vùng client của scrcpy bằng MSS; cửa sổ nhỏ hoặc có viền đen sẽ làm mất chi tiết/sai tọa độ. Chưa có nhận luồng giải mã trực tiếp.
3. Thay hai đường dẫn dưới đây bằng đường dẫn có thật của bạn:

```powershell
$env:EFOOTPHY_SCRCPY_DIR = 'C:\scrcpy-win64-v4.1'
$env:EFOOTPHY_FIELD_MODEL = 'D:\efootphy-models\field640\model.onnx'
$env:EFOOTPHY_EXECUTION_PROVIDER = 'cpu'
$env:EFOOTPHY_SHADOW_EVIDENCE = '0'
.\start_pc_brain.bat
```

Launcher mặc định dùng `.venv\Scripts\python.exe`. Nếu dùng Python đã cài riêng, đặt `$env:EFOOTPHY_PYTHON` thành đường dẫn đầy đủ tới `python.exe` trước khi chạy. Cửa sổ scrcpy được đặt tên chính xác `eFootball-Screen`, sử dụng USB, `--no-control`, chiều dài tối đa 1920 và yêu cầu tối đa 60 FPS. Đây là cấu hình yêu cầu, không phải kết quả đo đạt 60 FPS.

## Đọc HUD đúng ý nghĩa

- Vòng vàng: phát hiện bóng ở khung vừa xử lý; vẫn có thể là false positive của detector.
- Vòng xám nét đứt: ước lượng ngắn khi mất bóng, tối đa 150 ms từ lần phát hiện cuối; sau đó ẩn.
- Đường ngắn: hướng vận tốc **trên ảnh 2D**, chưa phải quỹ đạo rơi theo mét.
- Dấu cộng tím khi bật `EFOOTPHY_SHADOW_EVIDENCE=1`: điểm **nghi là bóng đổ**, chưa được ghép chắc chắn với bóng. Mặc định tắt.
- AI Hz / xử lý ms: tốc độ vòng xử lý PC. USB RTT ở terminal đo gửi/nhận ACK, không bao gồm toàn bộ độ trễ scrcpy và thời gian vẽ.
- Khi kết quả cũ hơn 350 ms theo thời gian nhận + xử lý PC hoặc sai tỉ lệ ảnh, HUD ẩn hình học. Chưa đồng bộ đồng hồ/độ trễ một chiều qua USB.

## Cửa kiểm tra trước khi đánh giá chất lượng

Ghi rõ CPU/GPU, RAM, driver, model SHA-256, provider, độ phân giải cửa sổ và điện thoại. So sánh cùng đoạn chơi với HUD tắt/bật: hình HUD có thể xuất hiện trở lại trong scrcpy và bị AI đọc nhầm. Cần kiểm tra vùng cutout/thanh điều hướng, xoay màn hình, che cửa sổ, rút USB, dừng app và mất bóng.

Chưa có kiểm thử phần cứng Windows/Android trong phiên này. Minimap chưa được nối vào PC runtime; camera/pitch/3D và chiến thuật chưa đủ QC để bật. Bước đo tiếp theo là lấy log PC + ghi hình cùng phiên có/không HUD, sau đó mới tối ưu luồng giải mã và GPU. Không dùng số unit test làm tỷ lệ nhận diện.
