package com.example.minimap.processor

import android.graphics.PointF
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class PassSuggestion(
    val startX: Float, // Tọa độ tương đối (0f..1f) trong khung Minimap
    val startY: Float,
    val endX: Float,
    val endY: Float
)

data class TacticalResult(
    val homeDots: List<PointF>, // Tọa độ chuẩn hóa (0f..1f)
    val awayDots: List<PointF>,
    val passLines: List<PassSuggestion>,
    val defenseGapAlert: Boolean
)

class TacticalAnalyzer {
    private var prevHomeDots: List<PointF> = emptyList()

    // Mặc định hiệp 1 tấn công từ Trái sang Phải (true). Nhấn giữ Bubble để đảo chiều sang hiệp 2.
    @Volatile
    var attackLeftToRight: Boolean = true

    fun analyze(frame: RgbaFrame): TacticalResult {
        val w = frame.width
        val h = frame.height
        val bytes = frame.pixels
        if (w <= 10 || h <= 10 || bytes.size < w * h * 4) {
            return TacticalResult(emptyList(), emptyList(), emptyList(), false)
        }

        val rawHome = mutableListOf<PointF>()
        val rawAway = mutableListOf<PointF>()

        // Quét bước nhảy 2 pixel để giảm 75% tải CPU
        var y = 2
        while (y < h - 2) {
            var x = 2
            while (x < w - 2) {
                val idx = (y * w + x) * 4
                val r = bytes[idx].toInt() and 0xFF
                val g = bytes[idx + 1].toInt() and 0xFF
                val b = bytes[idx + 2].toInt() and 0xFF

                val maxC = max(r, max(g, b))
                val minC = min(r, min(g, b))
                val delta = maxC - minC

                // Chỉ xét các điểm sáng nổi bật trên nền cỏ tối của Minimap
                if (maxC > 165 && delta > 70) {
                    val hue = computeHue(r, g, b, maxC, delta)
                    // Đội nhà: Xanh dương / Xanh lơ (Hue 175..250)
                    if (hue in 175f..250f) {
                        rawHome.add(PointF(x.toFloat() / w, y.toFloat() / h))
                    }
                    // Đội khách: Đỏ / Cam / Vàng đậm (Hue 0..35 hoặc 335..360)
                    else if (hue <= 35f || hue >= 335f) {
                        rawAway.add(PointF(x.toFloat() / w, y.toFloat() / h))
                    }
                }
                x += 2
            }
            y += 2
        }

        // Gom cụm các pixel liền kề thành tâm 1 cầu thủ (bán kính 3.5% chiều rộng radar)
        val homeClusters = clusterPoints(rawHome, 0.035f)
        val awayClusters = clusterPoints(rawAway, 0.035f)

        val passes = detectForwardRuns(homeClusters, awayClusters)
        val gapAlert = detectDefensiveGap(homeClusters)

        prevHomeDots = homeClusters
        return TacticalResult(homeClusters, awayClusters, passes, gapAlert)
    }

    private fun computeHue(r: Int, g: Int, b: Int, maxC: Int, delta: Int): Float {
        if (delta == 0) return 0f
        val h = when (maxC) {
            r -> 60f * (((g - b).toFloat() / delta) % 6f)
            g -> 60f * (((b - r).toFloat() / delta) + 2f)
            else -> 60f * (((r - g).toFloat() / delta) + 4f)
        }
        return if (h < 0f) h + 360f else h
    }

    private fun clusterPoints(points: List<PointF>, radius: Float): List<PointF> {
        val result = mutableListOf<PointF>()
        val used = BooleanArray(points.size)
        for (i in points.indices) {
            if (used[i]) continue
            var sumX = points[i].x
            var sumY = points[i].y
            var count = 1
            used[i] = true
            for (j in i + 1 until points.size) {
                if (!used[j] && hypot(points[i].x - points[j].x, points[i].y - points[j].y) <= radius) {
                    sumX += points[j].x
                    sumY += points[j].y
                    count++
                    used[j] = true
                }
            }
            result.add(PointF(sumX / count, sumY / count))
        }
        return result
    }

    private fun detectForwardRuns(home: List<PointF>, away: List<PointF>): List<PassSuggestion> {
        if (prevHomeDots.isEmpty() || away.size < 2) return emptyList()
        val suggestions = mutableListOf<PassSuggestion>()

        // Xác định hậu vệ cuối cùng (bỏ qua thủ môn đứng sát biên ngang)
        val sortedAwayX = if (attackLeftToRight) {
            away.map { it.x }.sortedDescending()
        } else {
            away.map { it.x }.sorted()
        }
        val offsideLineX = sortedAwayX[1]

        for (curr in home) {
            val prev = prevHomeDots.minByOrNull { hypot(curr.x - it.x, curr.y - it.y) } ?: continue
            val vx = curr.x - prev.x
            val vy = curr.y - prev.y

            // Kiểm tra tốc độ di chuyển dưới 0.12f (tránh nhận nhầm 2 cầu thủ khác nhau)
            if (hypot(vx, vy) > 0.12f) continue

            val isRunningForward = if (attackLeftToRight) vx > 0.014f else vx < -0.014f
            val nearOffsideLine = kotlin.math.abs(curr.x - offsideLineX) < 0.12f

            if (isRunningForward && nearOffsideLine) {
                suggestions.add(
                    PassSuggestion(
                        startX = curr.x,
                        startY = curr.y,
                        endX = (curr.x + vx * 6f).coerceIn(0f, 1f),
                        endY = (curr.y + vy * 6f).coerceIn(0f, 1f)
                    )
                )
            }
        }
        return suggestions
    }

    private fun detectDefensiveGap(home: List<PointF>): Boolean {
        val defenders = home.filter {
            if (attackLeftToRight) it.x < 0.42f else it.x > 0.58f
        }.sortedBy { it.y }

        if (defenders.size < 2) return false
        for (i in 0 until defenders.size - 1) {
            // Nếu khoảng cách dọc giữa 2 hậu vệ cạnh nhau vượt quá 35% chiều rộng sân -> Báo động hở sườn
            if (defenders[i + 1].y - defenders[i].y > 0.35f) return true
        }
        return false
    }
}