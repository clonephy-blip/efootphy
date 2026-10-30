package com.example.minimap.processor

import android.graphics.Rect
import android.media.Image

// Khung dữ liệu ảnh RGBA chuẩn mà ScreenCaptureManager, FloatingOverlayService và TacticalAnalyzer đang gọi
class RgbaFrame(
    val width: Int,
    val height: Int,
    val pixels: ByteArray
)

class MinimapCropper {
    // Tái sử dụng mảng byte giữa các khung hình để không làm đầy RAM điện thoại
    private var reusableBuffer = ByteArray(0)

    fun crop(image: Image, roi: Rect): RgbaFrame? {
        val planes = image.planes
        if (planes.isEmpty()) return null

        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        // Giới hạn tọa độ cắt nằm an toàn trong kích thước ảnh chụp màn hình
        val left = roi.left.coerceIn(0, (image.width - 1).coerceAtLeast(0))
        val top = roi.top.coerceIn(0, (image.height - 1).coerceAtLeast(0))
        val right = roi.right.coerceIn(left + 1, image.width.coerceAtLeast(1))
        val bottom = roi.bottom.coerceIn(top + 1, image.height.coerceAtLeast(1))

        val cropW = right - left
        val cropH = bottom - top
        if (cropW <= 2 || cropH <= 2) return null

        val requiredBytes = cropW * cropH * 4
        if (reusableBuffer.size < requiredBytes) {
            reusableBuffer = ByteArray(requiredBytes)
        }

        // Copy siêu tốc vùng pixel của Minimap từ bộ đệm ImageReader sang mảng RGBA
        if (pixelStride == 4) {
            val rowBytes = cropW * 4
            for (y in 0 until cropH) {
                val srcPos = (top + y) * rowStride + left * 4
                buffer.position(srcPos)
                buffer.get(reusableBuffer, y * rowBytes, rowBytes)
            }
        } else {
            var dstIdx = 0
            for (y in 0 until cropH) {
                var srcPos = (top + y) * rowStride + left * pixelStride
                for (x in 0 until cropW) {
                    reusableBuffer[dstIdx] = buffer.get(srcPos)
                    reusableBuffer[dstIdx + 1] = buffer.get(srcPos + 1)
                    reusableBuffer[dstIdx + 2] = buffer.get(srcPos + 2)
                    reusableBuffer[dstIdx + 3] = buffer.get(srcPos + 3)
                    dstIdx += 4
                    srcPos += pixelStride
                }
            }
        }

        return RgbaFrame(
            width = cropW,
            height = cropH,
            pixels = reusableBuffer
        )
    }
}
