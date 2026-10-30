package com.example.minimap.processor

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class PassSuggestion(
    val startX: Float,
    val startY: Float,
    val endX: Float,
    val endY: Float
)

data class TacticalResult(
    val homeDots: List<PointF>,
    val awayDots: List<PointF>,
    val ballPos: PointF?,
    val passLines: List<PassSuggestion>,
    val defenseGapAlert: Boolean
)

class TacticalAnalyzer {
    private var prevHomeDots: List<PointF> = emptyList()

    // Đội Xanh trong ảnh đang tấn công từ Trái sang Phải (true)
    @Volatile
    var attackLeftToRight: Boolean = true

    fun analyze(frame: RgbaFrame): TacticalResult {
        val w = frame.width
        val h = frame.height
        val bytes = frame.pixels
        if (w <= 10 || h <= 10 || bytes.size < w * h * 4) {
            return TacticalResult(emptyList(), emptyList(), null, emptyList(), false)
        }

        val rawHome = mutableListOf<PointF>()
        val rawAway = mutableListOf<PointF>()
        val rawBall = mutableListOf<PointF>()

        // Quét từng pixel (step = 1) để không bỏ sót lõi màu 3x3px của từng chấm cầu thủ
        for (y in 2 until h - 2) {
            val rowOffset = y * w
            for (x in 2 until w - 2) {
                val idx = (rowOffset + x) * 4
                val r = bytes[idx].toInt() and 0xFF
                val g = bytes[idx + 1].toInt() and 0xFF
                val b = bytes[idx + 2].toInt() and 0xFF

                val maxC = max(r, max(g, b))
                val minC = min(r, min(g, b))
                val delta = maxC - minC

                // Ngưỡng lọc đã hiệu chỉnh khớp với độ mờ của Minimap eFootball (maxC > 120, delta > 42)
                if (maxC > 120 && delta > 42) {
                    val hue = computeHue(r, g, b, maxC, delta)
                    val normX = x.toFloat() / w
                    val normY = y.toFloat() / h

                    when {
                        // 1. Chấm Đội Xanh (Home): Xanh dương có B nổi trội hơn R
                        hue in 195f..255f && b > r + 40 -> {
                            rawHome.add(PointF(normX, normY))
                        }
                        // 2. Chấm Đội Đỏ (Away): Đỏ có R nổi trội hơn G và B
                        (hue <= 25f || hue >= 340f) && r > g + 40 && r > b + 40 -> {
                            rawAway.add(PointF(normX, normY))
                        }
                        // 3. Tâm Quả Bóng (Màu Vàng sáng trên Minimap)
                        hue in 42f..68f && r > 175 && g > 170 && b < 100 -> {
                            rawBall.add(PointF(normX, normY))
                        }
                    }
                }
            }
        }

        // Bán kính gom cụm nhỏ (0.018f ~ 4px) để tách rõ các cầu thủ đứng sát nhau ở giữa sân
        val homeClusters = clusterPoints(rawHome, 0.018f)
        val awayClusters = clusterPoints(rawAway, 0.018f)
        val ballPosition = clusterPoints(rawBall, 0.03f).firstOrNull()

        val passes = detectForwardRuns(homeClusters, awayClusters, ballPosition)
        val gapAlert = detectDefensiveGap(homeClusters)

        prevHomeDots = homeClusters
        return TacticalResult(homeClusters, awayClusters, ballPosition, passes, gapAlert)
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

    private fun detectForwardRuns(
        home: List<PointF>,
        away: List<PointF>,
        ball: PointF?
    ): List<PassSuggestion> {
        if (prevHomeDots.isEmpty() || away.size < 2) return emptyList()
        val suggestions = mutableListOf<PassSuggestion>()

        // Tìm hậu vệ cuối cùng của đối phương (bỏ qua thủ môn đứng sát gôn)
        val sortedAwayX = if (attackLeftToRight) {
            away.map { it.x }.sortedDescending()
        } else {
            away.map { it.x }.sorted()
        }
        val offsideLineX = sortedAwayX[1]

        for (curr in home) {
            // Chỉ xét cầu thủ đứng phía trước quả bóng (nếu nhận diện được bóng)
            if (ball != null) {
                if (attackLeftToRight && curr.x < ball.x - 0.02f) continue
                if (!attackLeftToRight && curr.x > ball.x + 0.02f) continue
            }

            val prev = prevHomeDots.minByOrNull { hypot(curr.x - it.x, curr.y - it.y) } ?: continue
            val vx = curr.x - prev.x
            val vy = curr.y - prev.y

            if (hypot(vx, vy) > 0.10f) continue

            val isRunningForward = if (attackLeftToRight) vx > 0.012f else vx < -0.012f
            val nearOffsideLine = abs(curr.x - offsideLineX) < 0.12f

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
            if (defenders[i + 1].y - defenders[i].y > 0.35f) return true
        }
        return false
    }
}
