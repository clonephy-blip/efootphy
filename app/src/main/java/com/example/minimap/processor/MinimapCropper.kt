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
    val controlled3DX: Float = -1f,
    val controlled3DY: Float = -1f,
    val gridW: Int = 0,
    val gridH: Int = 0,
    val gridPixels: ByteArray = ByteArray(0)
)

class MinimapCropper {
    private var minimapBuffer = ByteArray(0)
    private var gridBuffer = ByteArray(0)

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

        val step = 4
        val gW = imgW / step
        val gH = imgH / step
        val neededGrid = gW * gH * 4
        if (gridBuffer.size < neededGrid) {
            gridBuffer = ByteArray(neededGrid)
        }

        var gIdx = 0
        for (gy in 0 until gH) {
            val rowStart = (gy * step) * rowStride
            for (gx in 0 until gW) {
                val pPos = rowStart + (gx * step) * pixelStride
                gridBuffer[gIdx] = buffer.get(pPos)
                gridBuffer[gIdx + 1] = buffer.get(pPos + 1)
                gridBuffer[gIdx + 2] = buffer.get(pPos + 2)
                gridBuffer[gIdx + 3] = -1
                gIdx += 4
            }
        }

        // Quét tìm Tam giác Xanh Ngọc dưới Thanh Thể Lực Đen của cầu thủ đang điều khiển (Bỏ qua bảng LED khán đài y < 0.23)
        val minPlayY = (gH * 0.23f).toInt().coerceAtLeast(2)
        val maxPlayY = (gH * 0.77f).toInt().coerceAtMost(gH - 2)
        val minPlayX = (gW * 0.12f).toInt().coerceAtLeast(2)
        val maxPlayX = (gW * 0.88f).toInt().coerceAtMost(gW - 2)

        var sumCursorX = 0f
        var sumCursorY = 0f
        var cursorHits = 0

        for (gy in minPlayY..maxPlayY) {
            val rowOff = gy * gW
            for (gx in minPlayX..maxPlayX) {
                val idx = (rowOff + gx) * 4
                val r = gridBuffer[idx].toInt() and 0xFF
                val g = gridBuffer[idx + 1].toInt() and 0xFF
                val b = gridBuffer[idx + 2].toInt() and 0xFF

                // Tam giác con trỏ eFootball: Xanh Ngọc sáng (R < 55, G > 210, B > 210)
                if (r < 55 && g > 210 && b > 210) {
                    // Kiểm tra ngay phía trên (gy - 1 hoặc gy - 2) phải có nền Đen của hộp thanh thể lực!
                    val up1 = ((gy - 1) * gW + gx) * 4
                    val up2 = ((gy - 2) * gW + gx) * 4
                    val r1 = gridBuffer[up1].toInt() and 0xFF
                    val g1 = gridBuffer[up1 + 1].toInt() and 0xFF
                    val b1 = gridBuffer[up1 + 2].toInt() and 0xFF
                    val r2 = gridBuffer[up2].toInt() and 0xFF
                    val g2 = gridBuffer[up2 + 1].toInt() and 0xFF
                    val b2 = gridBuffer[up2 + 2].toInt() and 0xFF

                    val hasDarkStaminaBox = (r1 < 48 && g1 < 48 && b1 < 48) || (r2 < 48 && g2 < 48 && b2 < 48)
                    if (hasDarkStaminaBox) {
                        sumCursorX += gx.toFloat() / gW
                        sumCursorY += gy.toFloat() / gH
                        cursorHits++
                    }
                }
            }
        }

        var foot3DX = -1f
        var foot3DY = -1f
        if (cursorHits >= 1) {
            foot3DX = (sumCursorX / cursorHits).coerceIn(0.08f, 0.92f)
            // Vòng tròn trắng dưới bàn chân cầu thủ nằm dưới mũi tam giác Xanh Ngọc đúng 6.8% chiều cao màn hình
            foot3DY = ((sumCursorY / cursorHits) + 0.068f).coerceIn(0.24f, 0.85f)
        }

        return RgbaFrame(
            width = cropW,
            height = cropH,
            pixels = minimapBuffer,
            timestampNs = timestampNs,
            screenMode = ScreenMode.MODE_LIVE_MATCH,
            controlled3DX = foot3DX,
            controlled3DY = foot3DY,
            gridW = gW,
            gridH = gH,
            gridPixels = gridBuffer
        )
    }
}
