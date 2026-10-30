package com.example.minimap.processor

import android.graphics.Rect
import android.media.Image

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
    val screenMode: Int = ScreenMode.MODE_LIVE_MATCH,
    val controlled3DX: Float = -1f, // Tọa độ X (0..1) bàn chân cầu thủ đang cầm bóng trên Sân 3D
    val controlled3DY: Float = -1f, // Tọa độ Y (0..1) bàn chân cầu thủ đang cầm bóng trên Sân 3D
    val gridW: Int = 0,
    val gridH: Int = 0,
    val gridPixels: ByteArray = ByteArray(0)
)

class MinimapCropper {
    private var minimapBuffer = ByteArray(0)
    private var gridBuffer = ByteArray(0)

    // Giữ tương thích ngược nếu nơi nào gọi hàm 2 tham số
    fun crop(image: Image, roi: Rect): RgbaFrame? = crop(image, roi, System.nanoTime())

    fun crop(image: Image, roi: Rect, timestampNs: Long): RgbaFrame? {
        val planes = image.planes
        if (planes.isEmpty()) return null

        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val imgW = image.width
        val imgH = image.height
        if (imgW <= 32 || imgH <= 32) return null

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

        // 2. Quét lưới toàn màn hình 3D (Step = 4) trong 0.3ms để:
        //    (a) Bắt thanh tên Xanh Ngọc (Neon Cyan) trên đầu cầu thủ đang điều khiển trên sân 3D
        //    (b) Tự động phân biệt đang Đá trong trận (LIVE_MATCH) hay đang mở Sơ đồ chiến thuật (GAMEPLAN)
        val step = 4
        val gW = imgW / step
        val gH = imgH / step
        val neededGrid = gW * gH * 4
        if (gridBuffer.size < neededGrid) {
            gridBuffer = ByteArray(neededGrid)
        }

        var sumCyanX = 0f
        var sumCyanY = 0f
        var cyanCount = 0

        var pitchGrassPixels = 0
        var leftSidebarDarkBluePixels = 0
        var formationPitchGreenPixels = 0

        val pitchMinY = (gH * 0.12f).toInt()
        val pitchMaxY = (gH * 0.76f).toInt()
        val pitchMinX = (gW * 0.08f).toInt()
        val pitchMaxX = (gW * 0.92f).toInt()

        var gIdx = 0
        for (gy in 0 until gH) {
            val rowStart = (gy * step) * rowStride
            val in3DPlayAreaY = gy in pitchMinY..pitchMaxY

            for (gx in 0 until gW) {
                val pPos = rowStart + (gx * step) * pixelStride
                val r = buffer.get(pPos).toInt() and 0xFF
                val g = buffer.get(pPos + 1).toInt() and 0xFF
                val b = buffer.get(pPos + 2).toInt() and 0xFF

                gridBuffer[gIdx] = r.toByte()
                gridBuffer[gIdx + 1] = g.toByte()
                gridBuffer[gIdx + 2] = b.toByte()
                gridBuffer[gIdx + 3] = -1
                gIdx += 4

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

                // Bắt thanh thể lực & con trỏ Xanh Ngọc (Neon Cyan) trên đầu cầu thủ đang điều khiển trên sân 3D
                if (in3DPlayAreaY && gx in pitchMinX..pitchMaxX) {
                    if (g > 212 && b > 202 && r < 72) {
                        sumCyanX += gx.toFloat() / gW
                        sumCyanY += gy.toFloat() / gH
                        cyanCount++
                    }
                }
            }
        }

        // Xác định tọa độ bàn chân cầu thủ đang cầm bóng trên mặt cỏ 3D (nằm dưới thanh tên khoảng 8.2% chiều cao màn hình)
        var foot3DX = -1f
        var foot3DY = -1f
        if (cyanCount >= 2) {
            foot3DX = (sumCyanX / cyanCount).coerceIn(0.05f, 0.95f)
            foot3DY = ((sumCyanY / cyanCount) + 0.082f).coerceIn(0.15f, 0.88f)
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
            controlled3DX = foot3DX,
            controlled3DY = foot3DY,
            gridW = gW,
            gridH = gH,
            gridPixels = gridBuffer
        )
    }
}
