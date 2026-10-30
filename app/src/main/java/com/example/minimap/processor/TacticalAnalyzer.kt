package com.example.minimap.processor

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

data class PassLane(
    val fromX: Float,
    val fromY: Float,
    val toX: Float,
    val toY: Float,
    val isThroughRun: Boolean
)

data class TacticalResult(
    val homeDots: List<PointF>,
    val awayDots: List<PointF>,
    val ballPos: PointF?,
    val activePlayerPos: PointF?,
    val unmarkedTarget: PointF?,
    val passLanes: List<PassLane>,
    val offsideLineX: Float,
    val defensiveLineX: Float,
    val defenseAlert: Boolean,
    val headlineBanner: String,
    val bannerColorType: Int // 0 = Bình thường, 1 = Cơ hội Tấn công (Xanh), 2 = Nguy hiểm Phòng ngự (Đỏ)
)

class TacticalAnalyzer {
    private data class ColorBlob(
        val x: Float,
        val y: Float,
        val r: Int,
        val g: Int,
        val b: Int
    )

    private var prevHomeDots: List<PointF> = emptyList()

    @Volatile
    var attackLeftToRight: Boolean = true

    fun analyze(frame: RgbaFrame): TacticalResult {
        val w = frame.width
        val h = frame.height
        val bytes = frame.pixels
        if (w <= 12 || h <= 12 || bytes.size < w * h * 4) {
            return emptyResult()
        }

        val rawBlobs = mutableListOf<ColorBlob>()
        val rawBall = mutableListOf<PointF>()
        val rawCyan = mutableListOf<PointF>()

        // 1. Quét toàn bộ khung Minimap chuẩn: Loại bỏ cỏ xanh và vạch vôi mờ, giữ lại mọi chấm cầu thủ
        for (y in 3 until h - 3) {
            val rowOffset = y * w
            for (x in 3 until w - 3) {
                val idx = (rowOffset + x) * 4
                val r = bytes[idx].toInt() and 0xFF
                val g = bytes[idx + 1].toInt() and 0xFF
                val b = bytes[idx + 2].toInt() and 0xFF

                // Loại bỏ màu mặt cỏ xanh của Minimap (G trội hơn R, B và độ sáng trung bình/thấp)
                val isPitchGrass = (g >= r + 10 && g >= b + 12 && g in 45..145 && r < 115 && b < 110)
                // Loại bỏ nền tối hoặc khung xám mờ
                val maxC = max(r, max(g, b))
                val minC = min(r, min(g, b))
                if (isPitchGrass || maxC < 125) continue

                val normX = x.toFloat() / w
                val normY = y.toFloat() / h

                // Nhận diện Quả bóng màu Vàng chanh sáng (R > 185, G > 175, B < 105)
                if (r > 185 && g > 175 && b < 105 && (r - b) > 90) {
                    rawBall.add(PointF(normX, normY))
                    continue
                }

                // Nhận diện Chấm Cầu Thủ Đang Điều Khiển (Màu Xanh Ngọc Cyan: G > 185, B > 195, R < 110)
                if (g > 185 && b > 195 && r < 110) {
                    rawCyan.add(PointF(normX, normY))
                    rawBlobs.add(ColorBlob(normX, normY, 220, 220, 220))
                    continue
                }

                // Nhận diện Chấm Đỏ / Cam / Vàng đậm
                val isRedOrWarm = (r > 145 && r > g + 38 && r > b + 38)
                // Nhận diện Chấm Xanh Dương
                val isBlue = (b > 140 && b > r + 35 && b > g + 15)
                // Nhận diện Tâm Chấm Trắng / Xám Sáng (Áo trắng như đội OL trong ảnh)
                // Để tránh bắt nhầm vạch kẻ sân mỏng 1px, kiểm tra điểm lân cận cũng sáng
                val isBrightWhiteCore = (r > 185 && g > 185 && b > 185 && (maxC - minC) < 35)

                if (isRedOrWarm || isBlue) {
                    rawBlobs.add(ColorBlob(normX, normY, r, g, b))
                } else if (isBrightWhiteCore) {
                    val rightIdx = (rowOffset + x + 1) * 4
                    val downIdx = ((y + 1) * w + x) * 4
                    val rRight = bytes[rightIdx].toInt() and 0xFF
                    val rDown = bytes[downIdx].toInt() and 0xFF
                    if (rRight > 175 && rDown > 175) {
                        rawBlobs.add(ColorBlob(normX, normY, r, g, b))
                    }
                }
            }
        }

        // 2. Gom cụm các pixel liền kề thành các chấm cầu thủ hoàn chỉnh
        val playerBlobs = clusterColorBlobs(rawBlobs, radius = 0.024f)
        val ballPos = clusterPoints(rawBall, 0.035f).firstOrNull()
        val activePos = clusterPoints(rawCyan, 0.035f).firstOrNull()

        if (playerBlobs.size < 4) {
            return emptyResult()
        }

        // 3. Tự động phân tách Đội Nhà (Home) và Đội Khách (Away) cho MỌI MÀU ÁO
        val (homeDots, awayDots) = separateTeamsAutomatically(playerBlobs, activePos)

        // 4. Tính toán Vạch Việt Vị (Offside Line) và Vạch Phòng Ngự (Defensive Line)
        val offsideLineX = computeOffsideLine(awayDots)
        val defensiveLineX = computeDefensiveLine(homeDots)

        // 5. Tìm cầu thủ đội nhà đang TRỐNG TRẢI NHẤT bên phần sân đối phương (Voronoi Space)
        val unmarkedTarget = findMostUnmarkedAttacker(homeDots, awayDots, offsideLineX)

        // 6. Vẽ Tuyến Chuyền Bóng An Toàn & Bắt Bài Chạy Chỗ Phá Bẫy Việt Vị
        val originPoint = ballPos ?: activePos ?: findBallCarrierFallback(homeDots)
        val passLanes = computeSmartPassLanes(originPoint, homeDots, awayDots, offsideLineX, unmarkedTarget)

        // 7. Kiểm tra Báo động Phòng ngự (Thủng trung lộ hoặc Bị phản công áp đảo quân số)
        val defenseAlert = checkDefensiveDanger(homeDots, awayDots, defensiveLineX)

        // 8. Tạo thông điệp chỉ dẫn chiến thuật rõ ràng cho người chơi nhìn lướt 0.5 giây là hiểu
        val (bannerText, bannerType) = buildHumanReadableBanner(
            passLanes = passLanes,
            unmarkedTarget = unmarkedTarget,
            defenseAlert = defenseAlert,
            homeCount = homeDots.size,
            awayCount = awayDots.size
        )

        prevHomeDots = homeDots

        return TacticalResult(
            homeDots = homeDots,
            awayDots = awayDots,
            ballPos = ballPos,
            activePlayerPos = activePos,
            unmarkedTarget = unmarkedTarget,
            passLanes = passLanes,
            offsideLineX = offsideLineX,
            defensiveLineX = defensiveLineX,
            defenseAlert = defenseAlert,
            headlineBanner = bannerText,
            bannerColorType = bannerType
        )
    }

    private fun separateTeamsAutomatically(
        blobs: List<ColorBlob>,
        activePos: PointF?
    ): Pair<List<PointF>, List<PointF>> {
        // Phân loại nhanh từng chấm theo đặc tính màu: Đỏ/Nóng vs Trắng/Xanh/Lạnh
        val redGroup = mutableListOf<PointF>()
        val nonRedGroup = mutableListOf<PointF>()

        for (b in blobs) {
            val isRed = (b.r > b.g + 35 && b.r > b.b + 35)
            if (isRed) {
                redGroup.add(PointF(b.x, b.y))
            } else {
                nonRedGroup.add(PointF(b.x, b.y))
            }
        }

        // Nếu 1 trong 2 nhóm trống (ví dụ Trắng đấu Xanh dương), tách bằng độ lệch kênh Blue
        if (redGroup.size < 2 || nonRedGroup.size < 2) {
            val blueGroup = mutableListOf<PointF>()
            val whiteGroup = mutableListOf<PointF>()
            for (b in blobs) {
                if (b.b > b.r + 30) blueGroup.add(PointF(b.x, b.y))
                else whiteGroup.add(PointF(b.x, b.y))
            }
            return assignHomeAndAwayBySide(whiteGroup, blueGroup, activePos)
        }

        return assignHomeAndAwayBySide(nonRedGroup, redGroup, activePos)
    }

    private fun assignHomeAndAwayBySide(
        groupA: List<PointF>,
        groupB: List<PointF>,
        activePos: PointF?
    ): Pair<List<PointF>, List<PointF>> {
        // Ưu tiên 1: Nhóm nào có chấm gần Cầu thủ đang điều khiển (Cyan cursor) hoặc Thủ môn bên sân nhà
        if (activePos != null) {
            val distA = groupA.minOfOrNull { hypot(it.x - activePos.x, it.y - activePos.y) } ?: 99f
            val distB = groupB.minOfOrNull { hypot(it.x - activePos.x, it.y - activePos.y) } ?: 99f
            if (distA <= distB) return Pair(groupA, groupB) else return Pair(groupB, groupA)
        }

        // Ưu tiên 2: Dựa vào Thủ môn đứng gần khung thành sân nhà (Trái nếu attackLeftToRight = true)
        val avgXA = if (groupA.isNotEmpty()) groupA.map { it.x }.average().toFloat() else 0.5f
        val avgXB = if (groupB.isNotEmpty()) groupB.map { it.x }.average().toFloat() else 0.5f

        return if (attackLeftToRight) {
            if (avgXA <= avgXB) Pair(groupA, groupB) else Pair(groupB, groupA)
        } else {
            if (avgXA >= avgXB) Pair(groupA, groupB) else Pair(groupB, groupA)
        }
    }

    private fun computeOffsideLine(away: List<PointF>): Float {
        if (away.isEmpty()) return if (attackLeftToRight) 0.75f else 0.25f
        val sortedX = if (attackLeftToRight) away.map { it.x }.sortedDescending() else away.map { it.x }.sorted()
        return if (sortedX.size >= 2) sortedX[1] else sortedX[0]
    }

    private fun computeDefensiveLine(home: List<PointF>): Float {
        if (home.isEmpty()) return if (attackLeftToRight) 0.25f else 0.75f
        val sortedX = if (attackLeftToRight) home.map { it.x }.sorted() else home.map { it.x }.sortedDescending()
        return if (sortedX.size >= 2) sortedX[1] else sortedX[0]
    }

    // Tìm cầu thủ tấn công đội nhà đang có khoảng trống lớn nhất (không bị ai kèm)
    private fun findMostUnmarkedAttacker(
        home: List<PointF>,
        away: List<PointF>,
        offsideLineX: Float
    ): PointF? {
        if (home.isEmpty() || away.isEmpty()) return null

        val candidates = home.filter { pt ->
            if (attackLeftToRight) {
                pt.x > 0.45f && pt.x <= offsideLineX + 0.03f
            } else {
                pt.x < 0.55f && pt.x >= offsideLineX - 0.03f
            }
        }

        var bestTarget: PointF? = null
        var maxClearance = 0.09f // Phải cách hậu vệ địch ít nhất 9% chiều dài sân mới tính là Trống trải

        for (att in candidates) {
            val nearestDefDist = away.minOfOrNull { def -> hypot(att.x - def.x, att.y - def.y) } ?: 0f
            if (nearestDefDist > maxClearance) {
                maxClearance = nearestDefDist
                bestTarget = att
            }
        }
        return bestTarget
    }

    private fun computeSmartPassLanes(
        origin: PointF?,
        home: List<PointF>,
        away: List<PointF>,
        offsideLineX: Float,
        unmarkedTarget: PointF?
    ): List<PassLane> {
        val lanes = mutableListOf<PassLane>()

        // 1. Bắt nhịp Tiền đạo đang bứt tốc cắt mặt vạch việt vị
        for (curr in home) {
            val prev = prevHomeDots.minByOrNull { hypot(curr.x - it.x, curr.y - it.y) }
            if (prev != null) {
                val vx = curr.x - prev.x
                val vy = curr.y - prev.y
                val speed = hypot(vx, vy)
                val runningForward = if (attackLeftToRight) vx > 0.011f else vx < -0.011f
                val nearOffside = abs(curr.x - offsideLineX) < 0.13f

                if (speed in 0.011f..0.11f && runningForward && nearOffside) {
                    lanes.add(
                        PassLane(
                            fromX = curr.x,
                            fromY = curr.y,
                            toX = (curr.x + vx * 5.5f).coerceIn(0.02f, 0.98f),
                            toY = (curr.y + vy * 5.5f).coerceIn(0.05f, 0.95f),
                            isThroughRun = true
                        )
                    )
                }
            }
        }

        // 2. Vẽ đường chuyền an toàn từ người cầm bóng tới cầu thủ trống trải nhất
        if (origin != null && unmarkedTarget != null) {
            val dist = hypot(unmarkedTarget.x - origin.x, unmarkedTarget.y - origin.y)
            if (dist > 0.06f && !isPassBlocked(origin, unmarkedTarget, away)) {
                lanes.add(
                    PassLane(
                        fromX = origin.x,
                        fromY = origin.y,
                        toX = unmarkedTarget.x,
                        toY = unmarkedTarget.y,
                        isThroughRun = false
                    )
                )
            }
        }

        return lanes.take(3)
    }

    // Kiểm tra xem có cầu thủ đối phương nào đứng chắn giữa đường chuyền hay không
    private fun isPassBlocked(from: PointF, to: PointF, opponents: List<PointF>): Boolean {
        val dx = to.x - from.x
        val dy = to.y - from.y
        val lenSq = dx * dx + dy * dy
        if (lenSq < 0.0001f) return false

        for (opp in opponents) {
            val t = ((opp.x - from.x) * dx + (opp.y - from.y) * dy) / lenSq
            if (t in 0.15f..0.88f) {
                val projX = from.x + t * dx
                val projY = from.y + t * dy
                if (hypot(opp.x - projX, opp.y - projY) < 0.038f) {
                    return true
                }
            }
        }
        return false
    }

    private fun checkDefensiveDanger(
        home: List<PointF>,
        away: List<PointF>,
        defLineX: Float
    ): Boolean {
        val defenders = home.filter {
            if (attackLeftToRight) it.x <= defLineX + 0.14f else it.x >= defLineX - 0.14f
        }.sortedBy { it.y }

        val enemyForwards = away.count {
            if (attackLeftToRight) it.x <= defLineX + 0.16f else it.x >= defLineX - 0.16f
        }

        // Nguy hiểm nếu tiền đạo địch đông hơn hậu vệ ta ở sân nhà, hoặc 2 trung vệ đứng cách nhau quá xa
        if (enemyForwards > defenders.size && enemyForwards >= 2) return true
        for (i in 0 until defenders.size - 1) {
            if (defenders[i + 1].y - defenders[i].y > 0.36f) return true
        }
        return false
    }

    private fun buildHumanReadableBanner(
        passLanes: List<PassLane>,
        unmarkedTarget: PointF?,
        defenseAlert: Boolean,
        homeCount: Int,
        awayCount: Int
    ): Pair<String, Int> {
        val throughRun = passLanes.firstOrNull { it.isThroughRun }
        if (throughRun != null) {
            val zone = getWingName(throughRun.toY)
            return Pair("⚡ CHỌC KHE NGAY: $zone ĐANG TÁCH LƯNG!", 1)
        }

        if (unmarkedTarget != null) {
            val zone = getWingName(unmarkedTarget.y)
            return Pair("🎯 KHOẢNG TRỐNG LỚN: CHUYỀN SANG $zone!", 1)
        }

        if (defenseAlert) {
            return Pair("⚠ BÁO ĐỘNG THỦ: LÙI VỀ BỊT TRUNG LỘ (ĐỪNG PRESSING)!", 2)
        }

        return Pair("📡 RADAR: ĐỘI TA ($homeCount) - ĐỐI THỦ ($awayCount) | SẴN SÀNG", 0)
    }

    private fun getWingName(normY: Float): String = when {
        normY < 0.34f -> "CÁNH TRÊN (TRÁI)"
        normY > 0.66f -> "CÁNH DƯỚI (PHẢI)"
        else -> "TRUNG LỘ"
    }

    private fun findBallCarrierFallback(home: List<PointF>): PointF? {
        if (home.isEmpty()) return null
        return if (attackLeftToRight) home.maxByOrNull { it.x } else home.minByOrNull { it.x }
    }

    private fun clusterColorBlobs(blobs: List<ColorBlob>, radius: Float): List<ColorBlob> {
        val res = mutableListOf<ColorBlob>()
        val used = BooleanArray(blobs.size)
        for (i in blobs.indices) {
            if (used[i]) continue
            var sx = blobs[i].x
            var sy = blobs[i].y
            var sr = blobs[i].r
            var sg = blobs[i].g
            var sb = blobs[i].b
            var c = 1
            used[i] = true
            for (j in i + 1 until blobs.size) {
                if (!used[j] && hypot(blobs[i].x - blobs[j].x, blobs[i].y - blobs[j].y) <= radius) {
                    sx += blobs[j].x
                    sy += blobs[j].y
                    sr += blobs[j].r
                    sg += blobs[j].g
                    sb += blobs[j].b
                    c++
                    used[j] = true
                }
            }
            res.add(ColorBlob(sx / c, sy / c, sr / c, sg / c, sb / c))
        }
        return res
    }

    private fun clusterPoints(points: List<PointF>, radius: Float): List<PointF> {
        val res = mutableListOf<PointF>()
        val used = BooleanArray(points.size)
        for (i in points.indices) {
            if (used[i]) continue
            var sx = points[i].x
            var sy = points[i].y
            var c = 1
            used[i] = true
            for (j in i + 1 until points.size) {
                if (!used[j] && hypot(points[i].x - points[j].x, points[i].y - points[j].y) <= radius) {
                    sx += points[j].x
                    sy += points[j].y
                    c++
                    used[j] = true
                }
            }
            res.add(PointF(sx / c, sy / c))
        }
        return res
    }

    private fun emptyResult() = TacticalResult(
        emptyList(), emptyList(), null, null, null, emptyList(),
        0.5f, 0.5f, false, "📡 ĐANG ĐỢI HIỂN THỊ MINIMAP TRONG TRẬN...", 0
    )
}
