package com.example.minimap.processor

import android.graphics.Rect
import android.media.Image
import android.os.SystemClock

object ScreenMode {
    const val MODE_UNKNOWN_OR_MENU = 0
    const val MODE_LIVE_MATCH = 1
    const val MODE_GAMEPLAN_FORMATION = 2
}

class RgbaFrame(
    val width: Int,
    val height: Int,
    val pixels: ByteArray,
    val timestampNs: Long = 0L,
    val screenMode: Int = ScreenMode.MODE_UNKNOWN_OR_MENU,
    val roiLeft: Int = 0,
    val roiTop: Int = 0,
    val captureWidth: Int = 0,
    val captureHeight: Int = 0
)

class MinimapCropper {
    private var minimapBuffer = ByteArray(0)

    // Giữ tương thích ngược nếu nơi nào gọi hàm 2 tham số
    fun crop(image: Image, roi: Rect): RgbaFrame? = crop(image, roi, SystemClock.elapsedRealtimeNanos())

    fun crop(image: Image, roi: Rect, timestampNs: Long): RgbaFrame? {
        val planes = image.planes
        if (planes.isEmpty()) return null

        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val imgW = image.width
        val imgH = image.height
        if (imgW <= 32 || imgH <= 32 || pixelStride < 4) return null

        // 1. Cắt vùng Minimap độ nét gốc
        val left = roi.left.coerceIn(0, imgW - 2)
        val top = roi.top.coerceIn(0, imgH - 2)
        val right = roi.right.coerceIn(left + 2, imgW)
        val bottom = roi.bottom.coerceIn(top + 2, imgH)

        val cropW = right - left
        val cropH = bottom - top
        val neededBytes = cropW * cropH * 4
        if (minimapBuffer.size < neededBytes) {
            minimapBuffer = ByteArray(neededBytes)
        }

        if (pixelStride == 4) {
            val rowBytes = cropW * 4
            for (y in 0 until cropH) {
                val srcPos = (top + y) * rowStride + left * 4
                buffer.position(srcPos)
                buffer.get(minimapBuffer, y * rowBytes, rowBytes)
            }
        } else {
            var dst = 0
            for (y in 0 until cropH) {
                var src = (top + y) * rowStride + left * pixelStride
                for (x in 0 until cropW) {
                    minimapBuffer[dst] = buffer.get(src)
                    minimapBuffer[dst + 1] = buffer.get(src + 1)
                    minimapBuffer[dst + 2] = buffer.get(src + 2)
                    minimapBuffer[dst + 3] = buffer.get(src + 3)
                    dst += 4
                    src += pixelStride
                }
            }
        }

        // 2. Coarse scene sampling; runtime must be measured on the target device.
        // This heuristic is only a research scene gate, not a validated classifier.
        val step = 4
        val gW = imgW / step
        val gH = imgH / step
        var pitchGrassPixels = 0
        var leftSidebarDarkBluePixels = 0
        var formationPitchGreenPixels = 0

        for (gy in 0 until gH) {
            val rowStart = (gy * step) * rowStride

            for (gx in 0 until gW) {
                val pPos = rowStart + (gx * step) * pixelStride
                val r = buffer.get(pPos).toInt() and 0xFF
                val g = buffer.get(pPos + 1).toInt() and 0xFF
                val b = buffer.get(pPos + 2).toInt() and 0xFF

                // Kiểm tra màu cỏ trên sân
                if (g > r + 14 && g > b + 18 && g in 55..190) {
                    pitchGrassPixels++
                    if (gx > gW * 0.25f && gx < gW * 0.92f && gy in (gH * 0.10f).toInt()..(gH * 0.88f).toInt()) {
                        formationPitchGreenPixels++
                    }
                }

                // Kiểm tra cột menu bên trái đặc trưng của màn hình Sơ đồ Game Plan eFootball (Màu Xanh Đen / Xám Tối)
                if (gx < gW * 0.22f && r < 45 && g < 55 && b in 35..110) {
                    leftSidebarDarkBluePixels++
                }

            }
        }

        // Nhận diện tự động Trạng thái Màn hình
        val totalGridCells = (gW * gH).coerceAtLeast(1)
        val grassRatio = pitchGrassPixels.toFloat() / totalGridCells
        val leftSidebarRatio = leftSidebarDarkBluePixels.toFloat() / (totalGridCells * 0.22f)

        val detectedMode = when {
            // Màn hình Sơ đồ đội hình (Game Plan): Có cột menu tối bên trái (> 45%) và sa bàn sân cỏ bên phải
            leftSidebarRatio > 0.45f && formationPitchGreenPixels > totalGridCells * 0.18f -> {
                ScreenMode.MODE_GAMEPLAN_FORMATION
            }
            // Màn hình Đang đá trong trận: Mặt cỏ chiếm đa số màn hình (> 38%)
            grassRatio > 0.38f -> {
                ScreenMode.MODE_LIVE_MATCH
            }
            else -> ScreenMode.MODE_UNKNOWN_OR_MENU
        }

        return RgbaFrame(
            width = cropW,
            height = cropH,
            pixels = minimapBuffer,
            timestampNs = timestampNs,
            screenMode = detectedMode,
            roiLeft = left,
            roiTop = top,
            captureWidth = imgW,
            captureHeight = imgH
        )
    }
}
