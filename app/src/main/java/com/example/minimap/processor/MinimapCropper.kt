package com.example.minimap.processor

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.Rect
import kotlin.math.hypot

data class TacticalPassLine(
    val startX: Float,
    val startY: Float,
    val endX: Float,
    val endY: Float
)

data class MinimapAnalysisResult(
    val homePlayers: List<PointF>,
    val awayPlayers: List<PointF>,
    val passSuggestions: List<TacticalPassLine>,
    val defensiveGapWarning: Boolean
)

class MinimapCropper(
    // Tọa độ mặc định của khung Minimap trên màn hình ngang (có thể chỉnh theo độ phân giải máy)
    var minimapRect: Rect = Rect(760, 780, 1160, 1030)
) {
    private var previousHomePlayers: List<PointF> = emptyList()

    // Dải màu HSV mặc định cho chấm cầu thủ đội nhà (Xanh sáng/Trắng) và đội khách (Đỏ/Cam)
    var homeHueRange = 180f..250f
    var awayHueRange = 0f..30f

    fun processFrame(fullScreenBitmap: Bitmap): MinimapAnalysisResult {
        val safeRect = Rect(
            minimapRect.left.coerceIn(0, fullScreenBitmap.width - 1),
            minimapRect.top.coerceIn(0, fullScreenBitmap.height - 1),
            minimapRect.right.coerceIn(1, fullScreenBitmap.width),
            minimapRect.bottom.coerceIn(1, fullScreenBitmap.height)
        )

        val width = safeRect.width()
        val height = safeRect.height()
        if (width <= 10 || height <= 10) {
            return MinimapAnalysisResult(emptyList(), emptyList(), emptyList(), false)
        }

        val pixels = IntArray(width * height)
        fullScreenBitmap.getPixels(pixels, 0, width, safeRect.left, safeRect.top, width, height)

        val rawHomePoints = mutableListOf<PointF>()
        val rawAwayPoints = mutableListOf<PointF>()
        val hsv = FloatArray(3)

        // Quét bước nhảy 2px để tối ưu tốc độ tuyệt đối trên điện thoại
        for (y in 0 until height step 2) {
            for (x in 0 until width step 2) {
                val color = pixels[y * width + x]
                Color.colorToHSV(color, hsv)
                val h = hsv[0]
                val s = hsv[1]
                val v = hsv[2]

                if (s > 0.45f && v > 0.65f) {
                    if (h in homeHueRange) {
                        rawHomePoints.add(PointF(x.toFloat(), y.toFloat()))
                    } else if (h in awayHueRange || h >= 340f) {
                        rawAwayPoints.add(PointF(x.toFloat(), y.toFloat()))
                    }
                }
            }
        }

        val homeClusters = clusterDots(rawHomePoints, radius = 8f)
        val awayClusters = clusterDots(rawAwayPoints, radius = 8f)

        val passLines = detectForwardRuns(homeClusters, awayClusters)
        val gapWarning = detectDefensiveGap(homeClusters, width)

        previousHomePlayers = homeClusters

        return MinimapAnalysisResult(
            homePlayers = homeClusters,
            awayPlayers = awayClusters,
            passSuggestions = passLines,
            defensiveGapWarning = gapWarning
        )
    }

    // Gom các điểm ảnh gần nhau thành 1 tâm chấm cầu thủ duy nhất
    private fun clusterDots(points: List<PointF>, radius: Float): List<PointF> {
        val clusters = mutableListOf<PointF>()
        val visited = BooleanArray(points.size)

        for (i in points.indices) {
            if (visited[i]) continue
            var sumX = points[i].x
            var sumY = points[i].y
            var count = 1
            visited[i] = true

            for (j in i + 1 until points.size) {
                if (!visited[j] && hypot(points[i].x - points[j].x, points[i].y - points[j].y) <= radius) {
                    sumX += points[j].x
                    sumY += points[j].y
                    count++
                    visited[j] = true
                }
            }
            clusters.add(PointF(sumX / count, sumY / count))
        }
        return clusters
    }

    // Phát hiện tiền đạo bứt tốc áp sát đường việt vị (tấn công từ trái sang phải)
    private fun detectForwardRuns(home: List<PointF>, away: List<PointF>): List<TacticalPassLine> {
        if (previousHomePlayers.isEmpty() || away.isEmpty()) return emptyList()
        val suggestions = mutableListOf<TacticalPassLine>()

        // Tìm hậu vệ cuối cùng của đối phương (bỏ qua thủ môn ở sát biên phải)
        val sortedAwayX = away.map { it.x }.sortedDescending()
        val offsideLineX = if (sortedAwayX.size >= 2) sortedAwayX[1] else sortedAwayX.firstOrNull() ?: return emptyList()

        for (curr in home) {
            val prev = previousHomePlayers.minByOrNull { hypot(curr.x - it.x, curr.y - it.y) } ?: continue
            val vx = curr.x - prev.x
            val vy = curr.y - prev.y

            // Nếu cầu thủ đang bứt tốc về phía khung thành (vx > 2.5px/frame) và nằm gần vạch việt vị
            if (vx > 2.5f && curr.x in (offsideLineX - 35f)..(offsideLineX + 5f)) {
                suggestions.add(
                    TacticalPassLine(
                        startX = minimapRect.left + curr.x,
                        startY = minimapRect.top + curr.y,
                        endX = minimapRect.left + curr.x + vx * 8f,
                        endY = minimapRect.top + curr.y + vy * 8f
                    )
                )
            }
        }
        return suggestions
    }

    // Cảnh báo khi khoảng cách giữa các hậu vệ đội nhà bị kéo giãn quá rộng
    private fun detectDefensiveGap(home: List<PointF>, minimapWidth: Int): Boolean {
        val defenders = home.filter { it.x < minimapWidth * 0.45f }.sortedBy { it.y }
        if (defenders.size < 2) return false
        for (i in 0 until defenders.size - 1) {
            if (defenders[i + 1].y - defenders[i].y > 42f) {
                return true
            }
        }
        return false
    }
}