# eFootball Minimap Assistant

Bộ khung Kotlin: Android 8.0+ (minSdk 26), compileSdk/targetSdk 35, JDK 17.
Chỉ chụp màn hình và tạo dữ liệu RGBA; chưa nhận diện cầu thủ, HSV hoặc điều khiển game.

## Mở và chạy

1. Mở thư mục này trong Android Studio, chọn Gradle 8.9 và JDK 17, cài SDK 35 rồi Sync.
   Bộ mã nguồn không kèm Gradle Wrapper JAR/scripts. Nếu IDE yêu cầu Wrapper, dùng
   Gradle 8.9 đã cài: `gradle wrapper --gradle-version 8.9` rồi `./gradlew assembleDebug`
   (Windows: `gradlew.bat assembleDebug`). Các phiên bản dependency đã ghim, không khẳng định là mới nhất.
2. Run app, bấm **Bật bubble**, cho phép hiển thị trên ứng dụng khác.
   Quyền thông báo có thể từ chối; foreground service vẫn cần notification.
3. Mở eFootball, kéo bubble ra ngoài vùng minimap và các nút game.
4. Chạm bubble, cấp quyền chụp trong hộp thoại hệ thống. Android 14+ ưu tiên chụp
   toàn màn hình để tọa độ nhất quán; resize callback vẫn xử lý kích thước thực tế.
5. Bubble hiện ON; Logcat tag `Minimap` báo kích thước ROI mỗi 5 giây ở debug build.
   Chạm lại để dừng projection. Chạm lần kế tiếp cần cấp quyền phiên mới.
   Bấm **Dừng** trong thông báo hoặc MainActivity để gỡ cả bubble.

MainActivity khởi chạy service bằng:

```kotlin
startForegroundService(Intent(this, FloatingOverlayService::class.java).apply {
    action = FloatingOverlayService.ACTION_SHOW
})
```

## Bố cục

| File / package | Vai trò |
| --- | --- |
| `ui/MainActivity.kt` | Xin quyền overlay, thông báo; bật/tắt bubble |
| `ui/CaptureConsentActivity.kt` | Activity Result API cho consent MediaProjection |
| `service/FloatingOverlayService.kt` | Hai cửa sổ WindowManager, foreground notification, bật/tắt capture |
| `service/ScreenCaptureManager.kt` | Projection, ImageReader, resize và cleanup trên HandlerThread |
| `processor/MinimapCropper.kt` | Crop ROI từ Image hoặc Bitmap, xuất ByteArray RGBA liên tục |

## Chỉnh vùng minimap

Trong `FloatingOverlayService`, đổi `roiProvider` sang `Rect(x, y, x + width, y + height)`.
Các số là pixel **của ảnh ImageReader sau khi scale**, gốc (0,0) ở góc trên trái,
right/bottom không bao gồm trong vùng crop. Vùng mặc định góc dưới trái chỉ là ví dụ,
không phải tọa độ eFootball đã đo. Cần hiệu chỉnh theo HUD, tỷ lệ màn hình và hướng máy.

Nếu đo trên màn hình gốc, với `s = min(1, 1280 / max(screenWidth, screenHeight))`,
nhân tọa độ và kích thước với s (hoặc dùng tỷ lệ như mẫu).
Khi xoay màn hình phải tính lại vùng ROI. Code dùng kích thước ảnh hiện tại để làm việc đó.

Đường Bitmap dùng được độc lập:

```kotlin
val frame = MinimapCropper().crop(bitmap, x = 20, y = 600, width = 200, height = 120)
```

Trong vòng lặp, tái sử dụng một MinimapCropper. Bitmap phải là bitmap phần mềm đọc được,
không phải `Bitmap.Config.HARDWARE`. Pipeline mặc định đọc Image trực tiếp để tránh
tạo/copy bitmap toàn màn hình. Nó xử lý cả rowStride và pixelStride.

## Điểm nối OpenCV / NDK

`onFrame` chạy đồng bộ trên capture thread. `frame.pixels` là RGBA 8 bit, thứ tự
R,G,B,A, kích thước `width * height * 4`, không có padding. Buffer được dùng lại
ở lần crop kế tiếp: xử lý ngay tại callback; nếu chuyển cho worker khác, cần
buffer pool/ownership riêng (hoặc copy và bounded queue có chính sách drop).

Sau khi thêm và khởi tạo OpenCV, có thể dùng Java API hoặc JNI. Ví dụ bước HSV
(minh họa, chưa đưa OpenCV vào dependency):

```kotlin
// Reuse Mat objects between frames; release them at session teardown.
rgbaMat.create(frame.height, frame.width, CvType.CV_8UC4)
rgbaMat.put(0, 0, frame.pixels)
Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
Imgproc.cvtColor(rgbMat, hsvMat, Imgproc.COLOR_RGB2HSV)
```

AHardwareBuffer/ImageFormat.PRIVATE không dùng cho đường CPU crop này: PRIVATE không
cung cấp pixel planes để đọc như RGBA. Muốn zero-copy GPU/NDK cần pipeline riêng với
HardwareBuffer/EGL, đồng bộ fence và quản lý lifetime; bộ khung dùng RGBA để đơn giản.

## Các điểm thiết kế cần giữ

- Bubble **không** có FLAG_NOT_TOUCHABLE: nó cần nhận kéo/thả và click. Cửa sổ
  trạng thái riêng có TYPE_APPLICATION_OVERLAY, FLAG_NOT_FOCUSABLE và FLAG_NOT_TOUCHABLE.
  Window alpha = 0.7 cho cửa sổ trạng thái đơn lẻ để tuân thủ giới hạn che chạm Android 12+.
  Phần trong bubble vẫn nhận chạm; phần ngoài bubble chuyển cho game.
- Không có quyền runtime công khai tên `android.permission.MEDIA_PROJECTION`.
  Consent dùng MediaProjectionManager; Android 14+ cần thêm FOREGROUND_SERVICE_MEDIA_PROJECTION.
- Loại foreground `specialUse` giữ bubble khi chưa capture; `mediaProjection` chỉ bật
  sau consent. Manifest khai báo subtype của specialUse. Nếu phát hành Google Play,
  mục đích specialUse cần được duyệt; boilerplate không bảo đảm phê duyệt.
- Mỗi token chỉ gọi getMediaProjection/createVirtualDisplay một lần. Callback đăng ký
  trước khi tạo display. Khi xoay, resize display hiện có và đổi ImageReader surface.
- Android 14+ dùng onCapturedContentResize; Android 8–13 dùng DisplayListener.
- acquireLatestImage + maxImages=3 + finally image.close; không xếp hàng xử lý mọi frame.
  Mọi xử lý ảnh chạy trên HandlerThread, không chạy trên main thread.
- START_NOT_STICKY: process bị dừng thì người dùng mở lại app và cấp quyền mới.
  onStop từ hệ thống (khóa màn hình, dừng chia sẻ...) giải phóng tài nguyên.

## Hiệu năng và kiểm tra thực tế

Mặc định cạnh dài capture <=1280 px và phân tích tối đa 15 FPS. `scanFps` là tốc độ
phân tích, không phải tốc độ producer VirtualDisplay. Các frame bị bỏ vẫn được acquire/close.
MediaProjection vẫn chụp khung màn hình/app đầy đủ; crop ROI giảm sao chép CPU và xử lý,
không loại bỏ toàn bộ chi phí capture/compositor. Không cam kết game giữ 60 FPS trên mọi máy.
Thử 10/15/20 FPS, 960/1280 px và đo frame time của game bằng Perfetto trên thiết bị đích.
Nếu marker nhỏ bị mất sau scale, tăng maxCaptureSide và đo lại tải/nhiệt.

Kiểm tra trên máy thật: bật/tắt bubble; từ chối consent; nhiều lần bật/dừng; kéo và xoay
máy; bấm xuyên badge; khóa màn hình; dừng chia sẻ bằng UI hệ thống; thu hồi quyền overlay.
Nội dung được bảo vệ bằng FLAG_SECURE có thể đen trong capture.

Bộ nguồn đã được rà soát cấu trúc và lifecycle; môi trường tạo file không có Android SDK/
Gradle nên chưa build APK, chạy instrumented test hay đo FPS trên thiết bị.

## Tài liệu Android chính thức

- https://developer.android.com/media/grow/media-projection
- https://developer.android.com/develop/background-work/services/fgs/service-types
- https://developer.android.com/reference/android/view/WindowManager.LayoutParams#FLAG_NOT_TOUCHABLE
- https://developer.android.com/reference/android/media/ImageReader#acquireLatestImage()
