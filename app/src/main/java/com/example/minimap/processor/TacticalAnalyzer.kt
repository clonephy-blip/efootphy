package com.example.minimap.processor

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

data class Pitch3DPassOption(
    val from3DX: Float,
    val from3DY: Float,
    val to3DX: Float,
    val to3DY: Float,
    val isThroughRun: Boolean,
    val isOffScreen: Boolean,
    val openSpacePct: Int
)

data class AutoFormationAnalysis(
    val isWaitingForOpponentView: Boolean,
    val detectedFormation: String,
    val tacticalWeakness: String,
    val actionableAdvice: String,
    val weakZoneSpots: List<PointF>,
    val detectedPlayerCards: List<PointF>
)

data class TacticalResult(
    val screenMode: Int,
    val attackLeftToRight: Boolean,
    val controlledFoot3D: PointF?,
    val pitch3DOptions: List<Pitch3DPassOption>,
    val minimapBestPass: Pair<PointF, PointF>?,
    val offsideLineX: Float,
    val dangerRunner3D: PointF?,
    val autoFormation: AutoFormationAnalysis?
)

class TacticalAnalyzer {
    private class TrackedPlayer(
        var x: Float,
        var y: Float,
        var vx: Float = 0f,
        var vy: Float = 0f,
        var missedFrames: Int = 0
    )

    private data class RawDot(
        val x: Float,
        val y: Float,
        val r: Int,
        val g: Int,
        val b: Int,
        val peakStrength: Int
    )

    private val homeTracks = mutableListOf<TrackedPlayer>()
    private var smoothedFoot3D: PointF? = null
    private var missed3DFootFrames = 0

    private var directionConfidence: Int = 12
    @Volatile var attackLeftToRight: Boolean = true
    @Volatile var manualDirectionLock: Boolean = false

    @Volatile private var requestFormationScan: Boolean = false
    @Volatile private var activeFormationReport: AutoFormationAnalysis? = null
    private var preferWideFlankPass: Boolean = false
    private var preferHalfSpacePass: Boolean = false

    fun toggleManualDirection() {
        manualDirectionLock = true
        attackLeftToRight = !attackLeftToRight
    }

    fun triggerManualFormationScan() {
        requestFormationScan = true
    }

    fun clearFormationScan() {
        requestFormationScan = false
        activeFormationReport = null
    }

    fun isFormationScanActive(): Boolean = activeFormationReport != null || requestFormationScan

    fun analyze(frame: RgbaFrame): TacticalResult {
        // 1. NẾU NGƯỜI DÙNG BẤM NÚT "🔍 SƠ ĐỒ" -> QUÉT SƠ ĐỒ ĐỐI THỦ NGAY LẬP TỨC
        if (requestFormationScan && frame.gridW > 20 && frame.gridH > 20) {
            requestFormationScan = false
            activeFormationReport = scanGamePlanScreen(frame.gridPixels, frame.gridW, frame.gridH)
        }
        activeFormationReport?.let { report ->
            return TacticalResult(
                screenMode = ScreenMode.MODE_GAMEPLAN_FORMATION,
                attackLeftToRight = attackLeftToRight,
                controlledFoot3D = null,
                pitch3DOptions = emptyList(),
                minimapBestPass = null,
                offsideLineX = 0.5f,
                dangerRunner3D = null,
                autoFormation = report
            )
        }

        // 2. CẬP NHẬT TỌA ĐỘ BÀN CHÂN CẦU THỦ ĐANG ĐIỀU KHIỂN TRÊN SÂN 3D (Xóa ngay nếu màn hình đen/mất dấu > 4 frame)
        if (frame.controlled3DX >= 0f && frame.controlled3DY >= 0f) {
            missed3DFootFrames = 0
            val rawPt = PointF(frame.controlled3DX, frame.controlled3DY)
            smoothedFoot3D = smoothPoint(smoothedFoot3D, rawPt, alpha = 0.60f)
        } else {
            missed3DFootFrames++
            if (missed3DFootFrames > 4) {
                smoothedFoot3D = null
            }
        }
        val currentFoot3D = smoothedFoot3D

        // 3. QUÉT MINIMAP ĐỘ CHÍNH XÁC CAO (Lòng Minimap nay đã sạch 100%, không bị vẽ đè)
        val w = frame.width
        val h = frame.height
        val bytes = frame.pixels
        if (w <= 20 || h <= 20 || bytes.size < w * h * 4) {
            return emptyLiveResult(currentFoot3D)
        }

        val candidates = mutableListOf<RawDot>()
        var activeCursorDot: PointF? = null
        var ballYellowDot: PointF? = null

        val rad = 4
        val ringOffsets = intArrayOf(
            -rad, 0, rad, 0, 0, -rad, 0, rad,
            -3, -3, 3, -3, -3, 3, 3, 3
        )

        for (y in rad until h - rad) {
            val rowOffset = y * w
            for (x in rad until w - rad) {
                val idx = (rowOffset + x) * 4
                val r = bytes[idx].toInt() and 0xFF
                val g = bytes[idx + 1].toInt() and 0xFF
                val b = bytes[idx + 2].toInt() and 0xFF

                if (max(r, max(g, b)) < 130) continue
                val centerLuma = (r * 3 + g * 4 + b) shr 3

                var grassRingCount = 0
                var ringLumaSum = 0
                var ringRedSum = 0
                var ringBlueSum = 0

                for (k in 0 until 16 step 2) {
                    val sx = x + ringOffsets[k]
                    val sy = y + ringOffsets[k + 1]
                    val sIdx = (sy * w + sx) * 4
                    val sr = bytes[sIdx].toInt() and 0xFF
                    val sg = bytes[sIdx + 1].toInt() and 0xFF
                    val sb = bytes[sIdx + 2].toInt() and 0xFF

                    ringLumaSum += (sr * 3 + sg * 4 + sb) shr 3
                    ringRedSum += sr
                    ringBlueSum += sb

                    if (sg >= sr + 5 && sg >= sb + 8 && sg in 38..168 && sr < 142) {
                        grassRingCount++
                    }
                }

                if (grassRingCount < 5) continue

                val avgRingLuma = ringLumaSum shr 3
                val peakScore = (centerLuma - avgRingLuma) + abs(r - (ringRedSum shr 3)) + abs(b - (ringBlueSum shr 3))
                if (peakScore < 70) continue

                val normX = x.toFloat() / w
                val normY = y.toFloat() / h

                // Quả bóng màu Vàng sáng trên Minimap
                if (r > 195 && g > 182 && b < 95 && (r - b) > 100) {
                    ballYellowDot = PointF(normX, normY)
                    continue
                }

                // Chấm Xanh Ngọc (Cầu thủ đang điều khiển trên Minimap)
                if (g > 185 && b > 192 && r < 105) {
                    activeCursorDot = PointF(normX, normY)
                    candidates.add(RawDot(normX, normY, r, g, b, peakScore + 120))
                    continue
                }

                candidates.add(RawDot(normX, normY, r, g, b, peakScore))
            }
        }

        val cleanDots = suppressDuplicatePeaks(candidates, minRadius = 0.046f, maxDots = 23)
        if (cleanDots.size < 4 || currentFoot3D == null) {
            return emptyLiveResult(currentFoot3D)
        }

        val (homeRaw, awayRaw) = splitHomeAndAwayTeams(cleanDots, activeCursorDot)
        if (!manualDirectionLock) {
            updateAutoAttackDirection(homeRaw, awayRaw)
        }

        updateKalmanTracks(homeRaw)
        val offsideX = computeOffsideLine(awayRaw)

        // Khóa gốc người cầm bóng trên Minimap: Ưu tiên chấm Xanh Ngọc (cầu thủ đang điều khiển)
        val carrierMap = activeCursorDot ?: ballYellowDot ?: homeRaw.minByOrNull {
            hypot(it.x - currentFoot3D.x, it.y - currentFoot3D.y)
        }

        val options3D = computeAccurate3DPasses(
            anchor3D = currentFoot3D,
            carrierMap = carrierMap,
            awayDots = awayRaw,
            offsideLineX = offsideX
        )

        return TacticalResult(
            screenMode = ScreenMode.MODE_LIVE_MATCH,
            attackLeftToRight = attackLeftToRight,
            controlledFoot3D = currentFoot3D,
            pitch3DOptions = options3D,
            minimapBestPass = null,
            offsideLineX = offsideX,
            dangerRunner3D = null,
            autoFormation = null
        )
    }

    private fun computeAccurate3DPasses(
        anchor3D: PointF,
        carrierMap: PointF?,
        awayDots: List<PointF>,
        offsideLineX: Float
    ): List<Pitch3DPassOption> {
        if (carrierMap == null || homeTracks.isEmpty()) return emptyList()

        data class ScoredOption(val opt: Pitch3DPassOption, val score: Float)
        val results = mutableListOf<ScoredOption>()

        for (t in homeTracks) {
            if (t.missedFrames > 2) continue
            val distMap = hypot(t.x - carrierMap.x, t.y - carrierMap.y)
            // Bỏ qua chính mình (< 0.055f) hoặc cầu thủ đứng quá xa (> 0.42f)
            if (distMap < 0.055f || distMap > 0.42f) continue

            // Kiểm tra việt vị
            val isOffside = if (attackLeftToRight) {
                t.x > offsideLineX + 0.015f && t.x > carrierMap.x
            } else {
                t.x < offsideLineX - 0.015f && t.x < carrierMap.x
            }
            if (isOffside) continue

            val forwardSpeed = if (attackLeftToRight) t.vx else -t.vx
            val isSprinting = forwardSpeed > 0.005f

            val aimMapX = if (isSprinting) (t.x + t.vx * 4.5f).coerceIn(0.04f, 0.96f) else t.x
            val aimMapY = if (isSprinting) (t.y + t.vy * 4.5f).coerceIn(0.06f, 0.94f) else t.y
            val aimPt = PointF(aimMapX, aimMapY)

            if (isLaneBlocked(carrierMap, aimPt, awayDots)) continue

            val clearance = awayDots.minOfOrNull { hypot(aimPt.x - it.x, aimPt.y - it.y) } ?: 0.25f
            if (clearance < 0.055f) continue

            val forwardGain = if (attackLeftToRight) (aimMapX - carrierMap.x) else (carrierMap.x - aimMapX)
            // Chỉ gợi ý chuyền lên phía trước hoặc chuyền ngang mở biên thuận lợi (Không gợi ý chuyền ngược về sân nhà!)
            if (forwardGain < -0.02f) continue

            var bonus = if (isSprinting) 0.10f else 0f
            if (preferWideFlankPass && (aimMapY < 0.30f || aimMapY > 0.70f)) bonus += 0.08f
            if (preferHalfSpacePass && aimMapY in 0.30f..0.70f) bonus += 0.07f

            val utility = forwardGain * 1.8f + clearance * 1.4f + bonus
            if (utility <= 0.05f) continue

            // HỆ SỐ CHIẾU CHUẨN ĐO TỪ SÂN 3D THỰC TẾ (scaleY = 0.92f, scaleX = 2.15f)
            val dMapX = aimMapX - carrierMap.x
            val dMapY = aimMapY - carrierMap.y

            val raw3DY = anchor3D.y + dMapY * 0.92f
            val perspectiveX = 0.82f + 0.36f * raw3DY.coerceIn(0.20f, 0.85f)
            val raw3DX = anchor3D.x + dMapX * 2.15f * perspectiveX

            // Nếu điểm rơi nằm trên khán đài (raw3DY < 0.24f) hoặc ngoài mép màn hình -> Đánh dấu isOffScreen để vẽ Mũi tên La bàn ngắn!
            val offScreen = raw3DX !in 0.08f..0.92f || raw3DY !in 0.24f..0.84f
            val openPct = ((clearance / 0.22f) * 100f).toInt().coerceIn(55, 99)

            results.add(
                ScoredOption(
                    opt = Pitch3DPassOption(
                        from3DX = anchor3D.x,
                        from3DY = anchor3D.y,
                        to3DX = raw3DX.coerceIn(0.06f, 0.94f),
                        to3DY = raw3DY.coerceIn(0.24f, 0.85f),
                        isThroughRun = isSprinting || forwardGain > 0.12f,
                        isOffScreen = offScreen,
                        openSpacePct = openPct
                    ),
                    score = utility
                )
            )
        }

        // Chỉ hiển thị 1 ĐƯỜNG CHUYỀN TỐT NHẤT (và tối đa 1 đường phụ nếu cực kỳ trống trải) để không bao giờ bị rối mắt!
        return results.sortedByDescending { it.score }.take(1).map { it.opt }
    }

    private fun scanGamePlanScreen(gridPixels: ByteArray, gridW: Int, gridH: Int): AutoFormationAnalysis {
        val brightSpots = mutableListOf<PointF>()
        val startX = (gridW * 0.20f).toInt()
        val endX = (gridW * 0.90f).toInt()
        val startY = (gridH * 0.14f).toInt()
        val endY = (gridH * 0.86f).toInt()

        for (y in startY until endY) {
            val rowOff = y * gridW
            for (x in startX until endX) {
                val idx = (rowOff + x) * 4
                val r = gridPixels[idx].toInt() and 0xFF
                val g = gridPixels[idx + 1].toInt() and 0xFF
                val b = gridPixels[idx + 2].toInt() and 0xFF

                val isPitchGreen = (g > r + 12 && g > b + 12 && g < 165)
                val luma = (r * 3 + g * 4 + b) shr 3
                if (!isPitchGreen && luma > 175) {
                    brightSpots.add(PointF(x.toFloat() / gridW, y.toFloat() / gridH))
                }
            }
        }

        val cards = clusterSimple(brightSpots, radius = 0.068f).take(11)
        val sortedX = cards.sortedBy { it.x }
        val outfield = if (sortedX.size > 1) sortedX.drop(1) else sortedX

        var def = 4
        var mid = 4
        var fwd = 2

        if (outfield.size >= 7) {
            val minX = outfield.first().x
            val spanX = (outfield.last().x - minX).coerceAtLeast(0.22f)
            var d = 0; var m = 0; var f = 0
            for (pt in outfield) {
                val rel = (pt.x - minX) / spanX
                when {
                    rel < 0.34f -> d++
                    rel < 0.68f -> m++
                    else -> f++
                }
            }
            def = d.coerceIn(3, 5)
            fwd = f.coerceIn(1, 4)
            mid = (10 - def - fwd).coerceIn(2, 6)
        }

        val widePlayers = cards.count { it.y < 0.28f || it.y > 0.72f }
        val weakZones = mutableListOf<PointF>()
        val formationLabel: String
        val weaknessText: String
        val adviceText: String

        if (widePlayers <= 2) {
            formationLabel = "$def-$mid-$fwd (Hẹp Trung Lộ)"
            weaknessText = "TRỐNG HOÁC 2 HÀNH LANG CÁNH"
            adviceText = "Đã nạp ưu tiên khoét 2 biên vào tia Laser! Bấm '✖ ĐÓNG SƠ ĐỒ' để vào đá."
            weakZones.add(PointF(0.58f, 0.22f))
            weakZones.add(PointF(0.58f, 0.78f))
            preferWideFlankPass = true
            preferHalfSpacePass = false
        } else if (fwd >= 3) {
            formationLabel = "$def-$mid-$fwd (3 Tiền Đạo)"
            weaknessText = "LỖ HỔNG Ở NÁCH TIỀN VỆ TRỤ (HALF-SPACE)"
            adviceText = "Đã nạp ưu tiên chọc khe nách trung lộ vào tia Laser! Bấm '✖ ĐÓNG SƠ ĐỒ' để vào đá."
            weakZones.add(PointF(0.54f, 0.36f))
            weakZones.add(PointF(0.54f, 0.64f))
            preferWideFlankPass = false
            preferHalfSpacePass = true
        } else {
            formationLabel = "$def-$mid-$fwd (Cân Bằng)"
            weaknessText = "KHOẢNG TRỐNG SAU LƯNG 2 HẬU VỆ BIÊN"
            adviceText = "Đã khóa mục tiêu sau lưng LB/RB đối phương! Bấm '✖ ĐÓNG SƠ ĐỒ' để vào đá."
            weakZones.add(PointF(0.66f, 0.24f))
            weakZones.add(PointF(0.66f, 0.76f))
            preferWideFlankPass = true
            preferHalfSpacePass = false
        }

        return AutoFormationAnalysis(
            isWaitingForOpponentView = false,
            detectedFormation = formationLabel,
            tacticalWeakness = weaknessText,
            actionableAdvice = adviceText,
            weakZoneSpots = weakZones,
            detectedPlayerCards = cards
        )
    }

    private fun updateAutoAttackDirection(home: List<PointF>, away: List<PointF>) {
        if (home.isEmpty() || away.isEmpty()) return
        val homeMinX = home.minOf { it.x }
        val homeMaxX = home.maxOf { it.x }
        val awayMinX = away.minOf { it.x }
        val awayMaxX = away.maxOf { it.x }

        if (homeMinX < 0.16f && awayMaxX > 0.84f && awayMinX > 0.18f) {
            directionConfidence = (directionConfidence + 2).coerceAtMost(30)
        } else if (homeMaxX > 0.84f && awayMinX < 0.16f && awayMaxX < 0.82f) {
            directionConfidence = (directionConfidence - 2).coerceAtLeast(-30)
        }
        if (directionConfidence >= 4) attackLeftToRight = true
        else if (directionConfidence <= -4) attackLeftToRight = false
    }

    private fun updateKalmanTracks(measuredHome: List<PointF>) {
        val matched = BooleanArray(measuredHome.size)
        val alpha = 0.65f
        val beta = 0.30f

        val iterator = homeTracks.iterator()
        while (iterator.hasNext()) {
            val track = iterator.next()
            val predX = track.x + track.vx
            val predY = track.y + track.vy

            var bestIdx = -1
            var bestDist = 0.07f
            for (i in measuredHome.indices) {
                if (matched[i]) continue
                val d = hypot(measuredHome[i].x - predX, measuredHome[i].y - predY)
                if (d < bestDist) {
                    bestDist = d
                    bestIdx = i
                }
            }

            if (bestIdx != -1) {
                matched[bestIdx] = true
                val meas = measuredHome[bestIdx]
                val rx = meas.x - predX
                val ry = meas.y - predY
                track.x = predX + alpha * rx
                track.y = predY + alpha * ry
                track.vx = (track.vx + beta * rx).coerceIn(-0.05f, 0.05f)
                track.vy = (track.vy + beta * ry).coerceIn(-0.05f, 0.05f)
                track.missedFrames = 0
            } else {
                track.missedFrames++
                if (track.missedFrames > 5) iterator.remove()
            }
        }

        for (i in measuredHome.indices) {
            if (!matched[i] && homeTracks.size < 11) {
                homeTracks.add(TrackedPlayer(measuredHome[i].x, measuredHome[i].y))
            }
        }
    }

    private fun suppressDuplicatePeaks(candidates: List<RawDot>, minRadius: Float, maxDots: Int): List<RawDot> {
        val sorted = candidates.sortedByDescending { it.peakStrength }
        val result = mutableListOf<RawDot>()
        for (c in sorted) {
            if (result.none { hypot(c.x - it.x, c.y - it.y) < minRadius }) {
                result.add(c)
                if (result.size >= maxDots) break
            }
        }
        return result
    }

    private fun splitHomeAndAwayTeams(dots: List<RawDot>, activeCursor: PointF?): Pair<List<PointF>, List<PointF>> {
        val warmGroup = mutableListOf<PointF>()
        val coolOrLightGroup = mutableListOf<PointF>()

        for (d in dots) {
            if (d.r > d.b + 32 && d.r > 135) warmGroup.add(PointF(d.x, d.y))
            else coolOrLightGroup.add(PointF(d.x, d.y))
        }

        val teamA: List<PointF>
        val teamB: List<PointF>
        if (warmGroup.size >= 2 && coolOrLightGroup.size >= 2) {
            teamA = coolOrLightGroup
            teamB = warmGroup
        } else {
            val byLuma = dots.sortedBy { it.r + it.g + it.b }
            val mid = byLuma.size / 2
            teamA = byLuma.drop(mid).map { PointF(it.x, it.y) }
            teamB = byLuma.take(mid).map { PointF(it.x, it.y) }
        }

        if (activeCursor != null) {
            val dA = teamA.minOfOrNull { hypot(it.x - activeCursor.x, it.y - activeCursor.y) } ?: 9f
            val dB = teamB.minOfOrNull { hypot(it.x - activeCursor.x, it.y - activeCursor.y) } ?: 9f
            return if (dA <= dB) Pair(teamA.take(11), teamB.take(11)) else Pair(teamB.take(11), teamA.take(11))
        }

        val minXA = teamA.minOfOrNull { it.x } ?: 0.5f
        val minXB = teamB.minOfOrNull { it.x } ?: 0.5f
        return if (attackLeftToRight) {
            if (minXA <= minXB) Pair(teamA.take(11), teamB.take(11)) else Pair(teamB.take(11), teamA.take(11))
        } else {
            if (minXA >= minXB) Pair(teamA.take(11), teamB.take(11)) else Pair(teamB.take(11), teamA.take(11))
        }
    }

    private fun isLaneBlocked(from: PointF, to: PointF, opponents: List<PointF>): Boolean {
        val dx = to.x - from.x
        val dy = to.y - from.y
        val lenSq = dx * dx + dy * dy
        if (lenSq < 0.0005f) return false

        for (opp in opponents) {
            val t = ((opp.x - from.x) * dx + (opp.y - from.y) * dy) / lenSq
            if (t in 0.12f..0.90f) {
                val px = from.x + t * dx
                val py = from.y + t * dy
                if (hypot(opp.x - px, opp.y - py) < 0.042f) return true
            }
        }
        return false
    }

    private fun computeOffsideLine(away: List<PointF>): Float {
        if (away.isEmpty()) return if (attackLeftToRight) 0.78f else 0.22f
        val sorted = if (attackLeftToRight) away.map { it.x }.sortedDescending() else away.map { it.x }.sorted()
        return if (sorted.size >= 2) sorted[1] else sorted[0]
    }

    private fun smoothPoint(prev: PointF?, curr: PointF, alpha: Float): PointF {
        if (prev == null) return curr
        return PointF(prev.x + alpha * (curr.x - prev.x), prev.y + alpha * (curr.y - prev.y))
    }

    private fun clusterSimple(pts: List<PointF>, radius: Float): List<PointF> {
        val out = mutableListOf<PointF>()
        val used = BooleanArray(pts.size)
        for (i in pts.indices) {
            if (used[i]) continue
            var sx = pts[i].x; var sy = pts[i].y; var c = 1
            used[i] = true
            for (j in i + 1 until pts.size) {
                if (!used[j] && hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y) <= radius) {
                    sx += pts[j].x; sy += pts[j].y; c++; used[j] = true
                }
            }
            out.add(PointF(sx / c, sy / c))
        }
        return out
    }

    private fun emptyLiveResult(foot3D: PointF?) = TacticalResult(
        screenMode = ScreenMode.MODE_LIVE_MATCH,
        attackLeftToRight = attackLeftToRight,
        controlledFoot3D = foot3D,
        pitch3DOptions = emptyList(),
        minimapBestPass = null,
        offsideLineX = 0.5f,
        dangerRunner3D = null,
        autoFormation = null
    )
}
