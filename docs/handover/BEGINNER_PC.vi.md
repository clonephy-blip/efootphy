# Hướng dẫn chạy eFootPhy PC cho người mới

Điện thoại chạy game và vẽ HUD. Máy tính xem hình game qua USB, nhận diện rồi gửi kết quả về điện thoại. **Cài APK trên điện thoại xong mới chỉ có phần hiển thị; PC phải mở chương trình AI thì HUD mới có kết quả.**

Bạn chơi và điều khiển game trên điện thoại. Cửa sổ scrcpy trên PC dùng để lấy hình.

## 1. Tải và giải nén bộ PC

Dùng ZIP riêng **efootphy_PC_START_v0.2.1.zip** được gửi trong cuộc trò chuyện. Model đã nằm trong ZIP, không cần tự tìm model hoặc đưa dữ liệu train vào máy.

Bấm chuột phải ZIP → **Extract All / Giải nén tất cả** → chọn thư mục:

`C:\eFootPhyPC`

Mở thư mục đó. Ngay bên trong phải thấy:

| Tệp/thư mục | Dùng để làm gì |
|---|---|
| `01_CAI_DAT_PC.bat` | Cài các thư viện cho PC lần đầu; cần Internet. |
| `02_CHAY_TRO_LY.bat` | Mở trợ lý mỗi lần chơi. |
| `models\field640.onnx` | Model nhận diện sân chính đã đặt sẵn. |
| `efootphy-pc-hud-v0.2.1.apk` | APK HUD cho điện thoại nếu chưa cài. |
| `HUONG_DAN.txt` | Bản hướng dẫn trong ZIP. |

Phải giải nén trước, không mở file BAT trực tiếp bên trong ZIP. Giữ các tệp trong cùng thư mục.

## 2. Cài phần PC một lần

Nếu đã có Python 3.12 **bản đầy đủ 64-bit**, có thể thử luôn file `01_CAI_DAT_PC.bat`.

Nếu file đó báo chưa tìm thấy Python: mở [trang Python 3.12.10](https://www.python.org/downloads/release/python-31210/), kéo tới mục **Files**, chọn **Windows installer (64-bit)**. Chạy file tải về, chọn **Add python.exe to PATH**, giữ **Python Launcher** được cài, rồi **Install Now**. Sau khi cài, đóng cửa sổ cũ.

Python 3.12.10 là bản dùng cho hướng dẫn cài Windows này; các bản security-only mới hơn trong nhánh 3.12 không có cùng bộ installer. ZIP không tự cài hay thay Python của bạn.

Trong `C:\eFootPhyPC`, nhấp đúp `01_CAI_DAT_PC.bat`. Chờ tải thư viện. Khi thấy:

`DA CAI XONG - CPU runtime san sang`

là phần cài thư viện đã xong. Bấm phím bất kỳ để đóng. Những lần chơi sau không phải chạy lại bước này.

Nếu có lỗi và không xuất hiện dòng trên, giữ cửa sổ, gửi nguyên phần lỗi để xử lý. Chưa tiếp tục sang phần chạy AI.

## 3. Chuẩn bị điện thoại

1. Cắm một điện thoại vào PC bằng USB, bật **USB debugging / Gỡ lỗi USB**. Chấp nhận **Allow USB debugging / Cho phép gỡ lỗi USB** nếu được hỏi.
2. Dùng app **eFootPhy PC HUD** bản mới; bấm **BẬT HUD**, cấp quyền hiển thị trên ứng dụng khác, quay lại app rồi bấm **BẬT HUD** lần nữa.
3. Nếu app cũ còn vẽ hoặc đang chạy trợ lý, dừng nó. App cũ là `com.example.minimap`; app mới là `com.example.minimap.pchud`. Không cần gỡ app cũ.
4. Mở eFootball trên điện thoại và vào trận. Đặt điện thoại nằm ngang.

Nếu APK mới chưa cài, mở PowerShell và chạy:

```powershell
cd C:\scrcpy-win64-v4.1
.\adb.exe -d install -r --no-incremental C:\eFootPhyPC\efootphy-pc-hud-v0.2.1.apk
.\adb.exe -d shell am force-stop com.example.minimap
.\adb.exe -d shell am start -n com.example.minimap.pchud/com.example.minimap.ui.PcHudActivity
```

Khi cài, cần thấy **Success**. Nếu có `Failure [...]` thì gửi nguyên dòng đó trước khi làm tiếp.

## 4. Mở AI trên PC

Bạn đã có thư mục scrcpy `C:\scrcpy-win64-v4.1` chứa `adb.exe` và `scrcpy.exe`.

1. Đóng các cửa sổ scrcpy cũ.
2. Mở `C:\eFootPhyPC`, nhấp đúp **`02_CHAY_TRO_LY.bat`**.
3. Một cửa sổ hình điện thoại tên **eFootball-Screen** sẽ mở; một cửa sổ chữ chạy AI cũng mở.
4. Giữ cả hai cửa sổ mở. Cửa sổ hình điện thoại phải nhìn thấy được, không thu nhỏ và không bị cửa sổ khác che.

Launcher đã chọn sẵn model và CPU; bạn không cần nhập đường dẫn model hoặc đặt biến môi trường. Để đổi thư mục scrcpy, sửa riêng dòng `EFOOTPHY_SCRCPY_DIR` trong file chạy, hoặc gửi đường dẫn thực tế để được sửa giúp.

## 5. Nhìn dấu hiệu đã kết nối

| Thấy gì | Nghĩa là gì |
|---|---|
| `DA CAI XONG - CPU runtime san sang` | Đã cài thư viện PC; chưa chứng minh đã kết nối game. |
| Hình game hiện trong **eFootball-Screen** | Điện thoại đã truyền hình lên PC. |
| `CONNECTED TO PHONE HUD` trong cửa sổ AI | PC đã nối tới app HUD. |
| Dòng `Players:`, `Ball:`, `Infer:` thay đổi khi chơi | Vòng xử lý PC đang chạy. |
| Vòng vàng ở bóng / dấu ở chân cầu thủ trên điện thoại | HUD đang hiển thị kết quả PC. |
| `waiting phone...` hoặc nút **HUD …** | Chưa nối được HUD; kiểm tra app mới đã bật HUD và quyền cửa sổ nổi. |
| `SCRCPY_UNAVAILABLE` | Không lấy được cửa sổ game; kiểm tra cửa sổ scrcpy đang mở và không thu nhỏ. |

Vòng vàng là detector vừa tìm thấy bóng, vẫn có thể nhận nhầm. Vòng xám nét đứt là dự đoán ngắn khi mất bóng. Bản thử hiện còn bỏ sót bóng; chưa hợp nhất minimap, chưa dựng 3D hoặc xác nhận đạt 60 FPS. Lần đầu dùng CPU để kiểm tra kết nối; việc chọn GPU phù hợp sẽ cần tên card và số liệu thực tế.

## 6. Tắt và lần sau dùng lại

Để tắt: chọn cửa sổ AI, bấm **Ctrl+C**, đóng cửa sổ hình điện thoại; trên điện thoại bấm **DỪNG HUD** hoặc dừng trong thông báo.

Lần sau: **cắm USB → mở app mới và bật HUD → vào trận → nhấp đúp `02_CHAY_TRO_LY.bat`**.

Nếu không chạy được, gửi nguyên phần lỗi trong cửa sổ AI hoặc ảnh phần lỗi. Không cần tải lại dữ liệu train hay tự đổi model.
