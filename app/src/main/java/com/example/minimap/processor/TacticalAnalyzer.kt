package com.example.minimap.processor

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

// Đường chuyền thực tế tăng cường vẽ trực tiếp trên MẶT CỎ SÂN 3D (Từ chân cầu thủ đang điều khiển)
data class Pitch3DPassOption(
    val from3DX: Float,          // Tọa độ X bàn chân cầu thủ đang cầm bóng trên Sân 3D (0..1)
    val from3DY: Float,          // Tọa độ Y bàn chân cầu thủ đang cầm bóng trên Sân 3D (0..1)
    val to3DX: Float,            // Tọa độ X điểm rơi bóng trên Sân 3D (0..1)
    val to3DY: Float,            // Tọa độ Y điểm rơi bóng trên Sân 3D (0..1)
    val isThroughRun: Boolean,   // true = Đồng đội đang bứt tốc chạy chỗ (Bấm Chọc khe), false = Đứng trống (Bấm Chuyền)
    val isOffScreen: Boolean,    // true = Đồng đội đang chạy ở ngoài khung hình Camera (Báo hiệu ở mép màn hình)
    val openSpacePct: Int        // Độ trống trải không bị ai kèm (50..99%)
)

// Kết quả AI tự động đọc màn hình Sơ đồ chiến thuật (Mảng 1 Tự động)
data class AutoFormationAnalysis(
    val isWaitingForOpponentView: Boolean, // true = Nhắc người chơi bấm xem sơ đồ đối thủ
    val detectedFormation: String,         // Ví dụ: "4-1-2-3", "4-2-2-2", "4-3-1-2"
    val tacticalWeakness: String,          // Tử huyệt chính của sơ đồ đối thủ
    val actionableAdvice: String,          // Gợi ý lối đánh ngắn gọn, thực chiến
    val weakZoneSpots: List<PointF>,       // Tọa độ (0..1) các vùng trống trên sa bàn để khoanh tròn Đỏ
    val detectedPlayerCards: List<PointF>  // Tọa độ 11 vị trí cầu thủ quét được
)

// Gói dữ liệu đầu ra 60FPS truyền thẳng sang lớp vẽ GPU ở Phần 3
data class TacticalResult(
    val screenMode: Int,
    val attackLeftToRight: Boolean,
    val controlledFoot3D: PointF?,              // Vị trí bàn chân cầu thủ ông đang cầm trên Sân 3D
    val pitch3DOptions: List<Pitch3DPassOption>,// Tối đa 2 đường chuyền ngon nhất vẽ ngay trên Sân 3D
    val minimapBestPass: Pair<PointF, PointF>?, // 1 đường chuyền chính vẽ thanh mảnh trên Minimap
    val offsideLineX: Float,                    // Vạch việt vị trên Minimap (0..1)
    val dangerRunner3D: PointF?,                // Cảnh báo cầu thủ địch đang lén xâm nhập khoảng trống sân nhà trên Sân 3D
    val autoFormation: AutoFormationAnalysis?   // Dữ liệu Mảng 1 khi ở màn hình Sơ đồ đội hình
)

class TacticalAnalyzer {
    // Cấu trúc theo dõi động học Kalman (Alpha-Beta Tracker) cho từng cầu thủ ở 60FPS
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

    // Bộ đếm tự động khóa chiều tấn công (Dương = Trái sang Phải, Âm = Phải sang Trái)
    private var directionConfidence: Int = 15
    @Volatile
    var attackLeftToRight: Boolean = true
    @Volatile
    var manualDirectionLock: Boolean = false

    // Bộ nhớ lưu chiến thuật tự động từ Mảng 1 sang Mảng 3
    private var preferWideFlankPass: Boolean = false
    private var preferHalfSpacePass: Boolean = false
    private var lastFormationReport: AutoFormationAnalysis? = null
    private var firstFormationSignature: Float = -1f

    fun toggleManualDirection() {
        manualDirectionLock = true
        attackLeftToRight = !attackLeftToRight
    }

    fun analyze(frame: RgbaFrame): TacticalResult {
        // 1. NẾU ĐANG Ở MÀN HÌNH SƠ ĐỒ CHIẾN THUẬT (GAME PLAN) -> KÍCH HOẠT MẢNG 1 TỰ ĐỘNG 100%
        if (frame.screenMode == ScreenMode.MODE_GAMEPLAN_FORMATION) {
            val report = analyzeGamePlanFormationAutomatically(
                frame.gridPixels,
                frame.gridW,
                frame.gridH
            )
            lastFormationReport = report
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

        // Reset chữ ký màn hình sơ đồ khi đã vào trong trận đấu
        firstFormationSignature = -1f

        // 2. LÀM MƯỢT TỌA ĐỘ BÀN CHÂN CẦU THỦ ĐANG ĐIỀU KHIỂN TRÊN SÂN 3D (60 FPS ZERO-JITTER)
        val raw3D = if (frame.controlled3DX >= 0f && frame.controlled3DY >= 0f) {
            PointF(frame.controlled3DX, frame.controlled3DY)
        } else {
            null
        }
        val currentFoot3D = smooth3DFootPosition(smoothedFoot3D, raw3D)
        smoothedFoot3D = currentFoot3D

        // 3. QUÉT MINIMAP BẰNG THUẬT TOÁN ĐỈNH CỤC BỘ (2D LOCAL PEAK) — DIỆT 100% NHIỄU ÁO CẦU THỦ 3D
        val w = frame.width
        val h = frame.height
        val bytes = frame.pixels
        if (w <= 20 || h <= 20 || bytes.size < w * h * 4) {
            return emptyLiveResult(currentFoot3D)
        }

        val candidates = mutableListOf<RawDot>()
        var activeCursorDot: PointF? = null
        var ballYellowDot: PointF? = null

        val ringRadius = 4
        val ringOffsets = intArrayOf(
            -ringRadius, 0, ringRadius, 0, 0, -ringRadius, 0, ringRadius,
            -3, -3, 3, -3, -3, 3, 3, 3
        )

        for (y in ringRadius until h - ringRadius) {
            val rowOffset = y * w
            for (x in ringRadius until w - ringRadius) {
                val idx = (rowOffset + x) * 4
                val r = bytes[idx].toInt() and 0xFF
                val g = bytes[idx + 1].toInt() and 0xFF
                val b = bytes[idx + 2].toInt() and 0xFF

                val maxC = max(r, max(g, b))
                if (maxC < 130) continue

                val centerLuma = (r * 3 + g * 4 + b) shr 3

                // KIỂM TRA ĐỈNH CỤC BỘ (LOCAL PEAK): Tâm chấm phải sáng hoặc nổi màu hơn vòng 4px xung quanh
                var pitchRingPixels = 0
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

                    // Mặt cỏ nền Minimap có kênh Green cao hơn Red, Blue và độ sáng vừa phải
                    if (sg >= sr + 5 && sg >= sb + 8 && sg in 38..168 && sr < 142) {
                        pitchRingPixels++
                    }
                }

                // Nếu vòng ngoài không có ít nhất 5/8 điểm là nền cỏ Minimap -> Đây là mảng quần áo 3D lớn bên dưới -> BỎ QUA!
                if (pitchRingPixels < 5) continue

                val avgRingLuma = ringLumaSum shr 3
                val avgRingRed = ringRedSum shr 3
                val avgRingBlue = ringBlueSum shr 3

                val lumaDiff = centerLuma - avgRingLuma
                val colorPop = abs(r - avgRingRed) + abs(b - avgRingBlue)
                val peakScore = lumaDiff + colorPop

                if (peakScore < 68) continue

                val normX = x.toFloat() / w
                val normY = y.toFloat() / h

                // Nhận diện Quả bóng màu Vàng chanh trên Minimap
                if (r > 192 && g > 180 && b < 98 && (r - b) > 95) {
                    ballYellowDot = PointF(normX, normY)
                    continue
                }

                // Nhận diện Chấm Xanh Ngọc (Cầu thủ đang được con trỏ điều khiển trên Minimap)
                if (g > 185 && b > 192 && r < 105) {
                    activeCursorDot = PointF(normX, normY)
                    candidates.add(RawDot(normX, normY, r, g, b, peakScore + 120))
                    continue
                }

                candidates.add(RawDot(normX, normY, r, g, b, peakScore))
            }
        }

        // Khử cực đại không gian (NMS): Đảm bảo mỗi cầu thủ chỉ tạo ra đúng 1 tâm điểm duy nhất
        val cleanDots = suppressDuplicatePeaks(candidates, minRadius = 0.046f, maxDots = 23)
        if (cleanDots.size < 4) {
            return emptyLiveResult(currentFoot3D)
        }

        // 4. TỰ ĐỘNG PHÂN TÁCH 2 ĐỘI & TỰ ĐỘNG KHÓA CHIỀU TẤN CÔNG QUA VỊ TRÍ THỦ MÔN
        val (homeRaw, awayRaw) = splitHomeAndAwayTeams(cleanDots, activeCursorDot)
        if (!manualDirectionLock) {
            updateAutoAttackDirection(homeRaw, awayRaw)
        }

        // 5. CẬP NHẬT BỘ LỌC ALPHA-BETA-KALMAN 60FPS CHO ĐỘI NHÀ ĐỂ LẤY VECTOR VẬN TỐC CHÍNH XÁC
        updateKalmanTracks(homeRaw)
        val offsideX = computeOffsideLine(awayRaw)

        // 6. CHIẾU HOMOGRAPHY TỪ MINIMAP LÊN MẶT CỎ SÂN 3D & TẠO RADAR MÉP MÀN HÌNH
        val ballOrCarrierMap = ballYellowDot ?: activeCursorDot ?: findClosestHomeDotToCenter(homeRaw)
        val anchorFoot3D = currentFoot3D ?: PointF(0.50f, 0.52f)

        val (options3D, bestMapLine) = computeRealTime3DPasses(
            anchor3D = anchorFoot3D,
            carrierMap = ballOrCarrierMap,
            awayDots = awayRaw,
            offsideLineX = offsideX
        )

        // 7. PHÁT HIỆN ĐỐI THỦ LÉN CHẠY CHỖ NGUY HIỂM Ở SÂN NHÀ
        val dangerSpot3D = detectDefensiveThreat3D(
            anchor3D = anchorFoot3D,
            carrierMap = ballOrCarrierMap,
            homeDots = homeRaw,
            awayDots = awayRaw
        )

        return TacticalResult(
            screenMode = ScreenMode.MODE_LIVE_MATCH,
            attackLeftToRight = attackLeftToRight,
            controlledFoot3D = currentFoot3D,
            pitch3DOptions = options3D,
            minimapBestPass = bestMapLine,
            offsideLineX = offsideX,
            dangerRunner3D = dangerSpot3D,
            autoFormation = null
        )
    }

    // Tự động nhận diện đội nhà đang thủ bên Trái (tấn công ->) hay thủ bên Phải (tấn công <-) dựa vào Thủ môn
    private fun updateAutoAttackDirection(home: List<PointF>, away: List<PointF>) {
        if (home.isEmpty() || away.isEmpty()) return

        val homeMinX = home.minOf { it.x }
        val homeMaxX = home.maxOf { it.x }
        val awayMinX = away.minOf { it.x }
        val awayMaxX = away.maxOf { it.x }

        // Nếu đội nhà có cầu thủ đứng sát gôn trái (< 0.14) và đội khách có thủ môn sát gôn phải (> 0.86)
        if (homeMinX < 0.16f && awayMaxX > 0.84f && awayMinX > 0.18f) {
            directionConfidence = (directionConfidence + 2).coerceAtMost(40)
        } else if (homeMaxX > 0.84f && awayMinX < 0.16f && awayMaxX < 0.82f) {
            directionConfidence = (directionConfidence - 2).coerceAtLeast(-40)
        } else {
            val homeAvgX = home.map { it.x }.average().toFloat()
            val awayAvgX = away.map { it.x }.average().toFloat()
            if (homeAvgX < awayAvgX - 0.05f) {
                directionConfidence = (directionConfidence + 1).coerceAtMost(40)
            } else if (homeAvgX > awayAvgX + 0.05f) {
                directionConfidence = (directionConfidence - 1).coerceAtLeast(-40)
            }
        }

        if (directionConfidence >= 6) {
            attackLeftToRight = true
        } else if (directionConfidence <= -6) {
            attackLeftToRight = false
        }
    }

    // Bộ lọc Alpha-Beta-Kalman cập nhật vị trí và vận tốc tức thời ở 60Hz
    private fun updateKalmanTracks(measuredHome: List<PointF>) {
        val matchedMeas = BooleanArray(measuredHome.size)
        val alpha = 0.65f // Hệ số bám vị trí tức thời (độ trễ < 10ms)
        val beta = 0.35f  // Hệ số làm mượt vector vận tốc

        // Dự đoán bước tiếp theo và ghép cặp với điểm đo mới nhất
        val iterator = homeTracks.iterator()
        while (iterator.hasNext()) {
            val track = iterator.next()
            val predX = track.x + track.vx
            val predY = track.y + track.vy

            var bestIdx = -1
            var bestDist = 0.075f
            for (i in measuredHome.indices) {
                if (matchedMeas[i]) continue
                val d = hypot(measuredHome[i].x - predX, measuredHome[i].y - predY)
                if (d < bestDist) {
                    bestDist = d
                    bestIdx = i
                }
            }

            if (bestIdx != -1) {
                matchedMeas[bestIdx] = true
                val meas = measuredHome[bestIdx]
                val resX = meas.x - predX
                val resY = meas.y - predY
                track.x = predX + alpha * resX
                track.y = predY + alpha * resY
                track.vx = (track.vx + beta * resX).coerceIn(-0.06f, 0.06f)
                track.vy = (track.vy + beta * resY).coerceIn(-0.06f, 0.06f)
                track.missedFrames = 0
            } else {
                track.missedFrames++
                if (track.missedFrames > 6) {
                    iterator.remove()
                }
            }
        }

        // Thêm các cầu thủ mới xuất hiện trên radar
        for (i in measuredHome.indices) {
            if (!matchedMeas[i] && homeTracks.size < 11) {
                homeTracks.add(TrackedPlayer(measuredHome[i].x, measuredHome[i].y))
            }
        }
    }

    // Tính toán đường chuyền từ chân cầu thủ 3D hiện tại tới đồng đội thuận lợi nhất trên Sân 3D
    private fun computeRealTime3DPasses(
        anchor3D: PointF,
        carrierMap: PointF?,
        awayDots: List<PointF>,
        offsideLineX: Float
    ): Pair<List<Pitch3DPassOption>, Pair<PointF, PointF>?> {
        if (carrierMap == null || homeTracks.isEmpty()) return Pair(emptyList(), null)

        data class RankedCandidate(
            val targetMap: PointF,
            val option3D: Pitch3DPassOption,
            val score: Float
        )

        val list = mutableListOf<RankedCandidate>()

        for (t in homeTracks) {
            if (t.missedFrames > 2) continue
            val distMap = hypot(t.x - carrierMap.x, t.y - carrierMap.y)
            if (distMap < 0.055f || distMap > 0.52f) continue

            // Loại bỏ cầu thủ đã rơi vào thế việt vị
            val isOffside = if (attackLeftToRight) {
                t.x > offsideLineX + 0.018f && t.x > carrierMap.x
            } else {
                t.x < offsideLineX - 0.018f && t.x < carrierMap.x
            }
            if (isOffside) continue

            val forwardSpeed = if (attackLeftToRight) t.vx else -t.vx
            val isSprintingForward = forwardSpeed > 0.0055f

            // Nếu đồng đội đang bứt tốc chạy chỗ, tính điểm rơi bóng đón đầu (Lead Pass)
            val aimMapX = if (isSprintingForward) {
                (t.x + t.vx * 6.0f).coerceIn(0.04f, 0.96f)
            } else {
                t.x
            }
            val aimMapY = if (isSprintingForward) {
                (t.y + t.vy * 6.0f).coerceIn(0.05f, 0.95f)
            } else {
                t.y
            }
            val aimPoint = PointF(aimMapX, aimMapY)

            // Kiểm tra hành lang chuyền bóng có bị cầu thủ đối phương đứng chắn không
            if (isLaneIntercepted(carrierMap, aimPoint, awayDots)) continue

            // Khoảng cách từ điểm nhận bóng tới hậu vệ địch gần nhất
            val spaceClearance = awayDots.minOfOrNull {
                hypot(aimPoint.x - it.x, aimPoint.y - it.y)
            } ?: 0.25f

            if (spaceClearance < 0.052f) continue

            val territoryGain = if (attackLeftToRight) (aimMapX - carrierMap.x) else (carrierMap.x - aimMapX)
            var tacticalBonus = if (isSprintingForward) 0.12f else 0f

            // Cộng điểm ưu tiên nếu khớp với tử huyệt sơ đồ đối thủ đã quét ở Mảng 1
            if (preferWideFlankPass && (aimMapY < 0.30f || aimMapY > 0.70f)) {
                tacticalBonus += 0.09f
            } else if (preferHalfSpacePass && aimMapY in 0.30f..0.70f) {
                tacticalBonus += 0.08f
            }

            val totalScore = territoryGain * 1.7f + spaceClearance * 1.4f + tacticalBonus
            if (totalScore <= 0.04f) continue

            // Phép chiếu phối cảnh (Perspective Homography) từ khoảng cách Minimap sang mặt cỏ Sân 3D
            val dMapX = aimMapX - carrierMap.x
            val dMapY = aimMapY - carrierMap.y

            val raw3DY = anchor3D.y + dMapY * 1.38f
            val perspectiveFactor = 0.76f + 0.48f * raw3DY.coerceIn(0.10f, 0.90f)
            val raw3DX = anchor3D.x + dMapX * 2.35f * perspectiveFactor

            val outsideCamera = raw3DX !in 0.06f..0.94f || raw3DY !in 0.12f..0.86f
            val clampedX = raw3DX.coerceIn(0.05f, 0.95f)
            val clampedY = raw3DY.coerceIn(0.12f, 0.86f)
            val openPct = ((spaceClearance / 0.22f) * 100f).toInt().coerceIn(55, 99)

            list.add(
                RankedCandidate(
                    targetMap = aimPoint,
                    option3D = Pitch3DPassOption(
                        from3DX = anchor3D.x,
                        from3DY = anchor3D.y,
                        to3DX = clampedX,
                        to3DY = clampedY,
                        isThroughRun = isSprintingForward || territoryGain > 0.13f,
                        isOffScreen = outsideCamera,
                        openSpacePct = openPct
                    ),
                    score = totalScore
                )
            )
        }

        val topTwo = list.sortedByDescending { it.score }.take(2)
        val bestMinimapPair = topTwo.firstOrNull()?.let { Pair(carrierMap, it.targetMap) }
        return Pair(topTwo.map { it.option3D }, bestMinimapPair)
    }

    // ==================== MẢNG 1 TỰ ĐỘNG: QUÉT & PHÂN TÍCH MÀN HÌNH SƠ ĐỒ ĐỐI THỦ ====================
    private fun analyzeGamePlanFormationAutomatically(
        gridPixels: ByteArray,
        gridW: Int,
        gridH: Int
    ): AutoFormationAnalysis {
        if (gridW <= 20 || gridH <= 20 || gridPixels.size < gridW * gridH * 4) {
            return AutoFormationAnalysis(
                isWaitingForOpponentView = true,
                detectedFormation = "Đang đọc sa bàn...",
                tacticalWeakness = "Đang phân tích",
                actionableAdvice = "Hãy mở xem Sơ đồ đội hình của đối thủ",
                weakZoneSpots = emptyList(),
                detectedPlayerCards = emptyList()
            )
        }

        val brightSpots = mutableListOf<PointF>()
        val startX = (gridW * 0.24f).toInt()
        val endX = (gridW * 0.90f).toInt()
        val startY = (gridH * 0.12f).toInt()
        val endY = (gridH * 0.88f).toInt()

        for (y in startY until endY) {
            val rowOff = y * gridW
            for (x in startX until endX) {
                val idx = (rowOff + x) * 4
                val r = gridPixels[idx].toInt() and 0xFF
                val g = gridPixels[idx + 1].toInt() and 0xFF
                val b = gridPixels[idx + 2].toInt() and 0xFF

                val isPitchGreen = (g > r + 12 && g > b + 12 && g < 165)
                val luma = (r * 3 + g * 4 + b) shr 3
                if (!isPitchGreen && luma > 178) {
                    brightSpots.add(PointF(x.toFloat() / gridW, y.toFloat() / gridH))
                }
            }
        }

        val cards = clusterSimple(brightSpots, radius = 0.068f).take(11)
        val currentSig = cards.sumOf { (it.x * 100 + it.y * 200).toDouble() }.toFloat()

        // Ghi nhận sơ đồ đầu tiên khi vừa mở Game Plan (thường là đội nhà), nhắc người chơi bấm sang xem sơ đồ đối thủ
        if (firstFormationSignature < 0f && cards.size >= 7) {
            firstFormationSignature = currentSig
        }
        val stillViewingOwnTeam = firstFormationSignature > 0f && abs(currentSig - firstFormationSignature) < 8f

        val sortedX = cards.sortedBy { it.x }
        val outfield = if (sortedX.size > 1) sortedX.drop(1) else sortedX

        var def = 4
        var mid = 4
        var fwd = 2

        if (outfield.size >= 7) {
            val minX = outfield.first().x
            val spanX = (outfield.last().x - minX).coerceAtLeast(0.22f)
            var d = 0
            var m = 0
            var f = 0
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
            formationLabel = "$def-$mid-$fwd (Bó Hẹp Trung Lộ)"
            weaknessText = "TRỐNG HOÁC 2 HÀNH LANG CÁNH (ĐỊCH KHÔNG CÓ TIỀN VỆ BIÊN)"
            adviceText = "Radar 60FPS đã tự động ưu tiên vẽ đường chuyền dạt ra 2 biên! Cấm kéo CB dâng cao ở trung lộ."
            weakZones.add(PointF(0.58f, 0.20f))
            weakZones.add(PointF(0.58f, 0.80f))
            preferWideFlankPass = true
            preferHalfSpacePass = false
        } else if (fwd >= 3) {
            formationLabel = "$def-$mid-$fwd (Dồn 3 Tiền Đạo Ép Sân)"
            weaknessText = "HỞ NÁCH TIỀN VỆ TRỤ (HALF-SPACE TRUNG LỘ)"
            adviceText = "Radar 60FPS đã tự động khóa mục tiêu chọc khe vào nách trung lộ ngay sau lưng hàng tiền vệ địch!"
            weakZones.add(PointF(0.54f, 0.36f))
            weakZones.add(PointF(0.54f, 0.64f))
            preferWideFlankPass = false
            preferHalfSpacePass = true
        } else {
            formationLabel = "$def-$mid-$fwd (Cân Bằng 2 Lớp)"
            weaknessText = "KHOẢNG TRỐNG SAU LƯNG 2 HẬU VỆ CÁNH KHI DÂNG CAO"
            adviceText = "Chờ tia Laser Xanh Lá trên mặt cỏ 3D bật sáng ở biên là phất Chọc khe ngay!"
            weakZones.add(PointF(0.66f, 0.24f))
            weakZones.add(PointF(0.66f, 0.76f))
            preferWideFlankPass = true
            preferHalfSpacePass = false
        }

        return AutoFormationAnalysis(
            isWaitingForOpponentView = stillViewingOwnTeam,
            detectedFormation = formationLabel,
            tacticalWeakness = weaknessText,
            actionableAdvice = adviceText,
            weakZoneSpots = weakZones,
            detectedPlayerCards = cards
        )
    }

    private fun detectDefensiveThreat3D(
        anchor3D: PointF,
        carrierMap: PointF?,
        homeDots: List<PointF>,
        awayDots: List<PointF>
    ): PointF? {
        if (carrierMap == null || homeDots.isEmpty() || awayDots.isEmpty()) return null

        val dangerousEnemies = awayDots.filter { e ->
            if (attackLeftToRight) e.x < 0.44f else e.x > 0.56f
        }
        val mostUnmarkedEnemy = dangerousEnemies.maxByOrNull { e ->
            homeDots.minOfOrNull { h -> hypot(e.x - h.x, e.y - h.y) } ?: 0f
        } ?: return null

        val minDefDist = homeDots.minOfOrNull { h ->
            hypot(mostUnmarkedEnemy.x - h.x, mostUnmarkedEnemy.y - h.y)
        } ?: 0f

        if (minDefDist < 0.115f) return null

        val dx = mostUnmarkedEnemy.x - carrierMap.x
        val dy = mostUnmarkedEnemy.y - carrierMap.y
        val p3DY = (anchor3D.y + dy * 1.38f).coerceIn(0.14f, 0.85f)
        val p3DX = (anchor3D.x + dx * 2.35f).coerceIn(0.06f, 0.94f)
        return PointF(p3DX, p3DY)
    }

    private fun suppressDuplicatePeaks(
        candidates: List<RawDot>,
        minRadius: Float,
        maxDots: Int
    ): List<RawDot> {
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

    private fun splitHomeAndAwayTeams(
        dots: List<RawDot>,
        activeCursor: PointF?
    ): Pair<List<PointF>, List<PointF>> {
        val warmGroup = mutableListOf<PointF>()
        val coolOrLightGroup = mutableListOf<PointF>()

        for (d in dots) {
            if (d.r > d.b + 32 && d.r > 135) {
                warmGroup.add(PointF(d.x, d.y))
            } else {
                coolOrLightGroup.add(PointF(d.x, d.y))
            }
        }

        val teamA: List<PointF>
        val teamB: List<PointF>
        if (warmGroup.size >= 2 && coolOrLightGroup.size >= 2) {
            teamA = coolOrLightGroup
            teamB = warmGroup
        } else {
            val byBrightness = dots.sortedBy { it.r + it.g + it.b }
            val mid = byBrightness.size / 2
            teamA = byBrightness.drop(mid).map { PointF(it.x, it.y) }
            teamB = byBrightness.take(mid).map { PointF(it.x, it.y) }
        }

        if (activeCursor != null) {
            val distA = teamA.minOfOrNull { hypot(it.x - activeCursor.x, it.y - activeCursor.y) } ?: 9f
            val distB = teamB.minOfOrNull { hypot(it.x - activeCursor.x, it.y - activeCursor.y) } ?: 9f
            return if (distA <= distB) {
                Pair(teamA.take(11), teamB.take(11))
            } else {
                Pair(teamB.take(11), teamA.take(11))
            }
        }

        val leftMostA = teamA.minOfOrNull { it.x } ?: 0.5f
        val leftMostB = teamB.minOfOrNull { it.x } ?: 0.5f
        return if (attackLeftToRight) {
            if (leftMostA <= leftMostB) Pair(teamA.take(11), teamB.take(11)) else Pair(teamB.take(11), teamA.take(11))
        } else {
            if (leftMostA >= leftMostB) Pair(teamA.take(11), teamB.take(11)) else Pair(teamB.take(11), teamA.take(11))
        }
    }

    private fun isLaneIntercepted(from: PointF, to: PointF, opponents: List<PointF>): Boolean {
        val dx = to.x - from.x
        val dy = to.y - from.y
        val lenSq = dx * dx + dy * dy
        if (lenSq < 0.0005f) return false

        for (opp in opponents) {
            val t = ((opp.x - from.x) * dx + (opp.y - from.y) * dy) / lenSq
            if (t in 0.12f..0.90f) {
                val projX = from.x + t * dx
                val projY = from.y + t * dy
                if (hypot(opp.x - projX, opp.y - projY) < 0.042f) return true
            }
        }
        return false
    }

    private fun computeOffsideLine(away: List<PointF>): Float {
        if (away.isEmpty()) return if (attackLeftToRight) 0.78f else 0.22f
        val sorted = if (attackLeftToRight) away.map { it.x }.sortedDescending() else away.map { it.x }.sorted()
        return if (sorted.size >= 2) sorted[1] else sorted[0]
    }

    private fun findClosestHomeDotToCenter(home: List<PointF>): PointF? {
        return home.minByOrNull { hypot(it.x - 0.5f, it.y - 0.5f) }
    }

    private fun smooth3DFootPosition(prev: PointF?, curr: PointF?): PointF? {
        if (curr == null) return prev
        if (prev == null) return curr
        val alpha = 0.62f
        return PointF(
            prev.x + alpha * (curr.x - prev.x),
            prev.y + alpha * (curr.y - prev.y)
        )
    }

    private fun clusterSimple(pts: List<PointF>, radius: Float): List<PointF> {
        val out = mutableListOf<PointF>()
        val used = BooleanArray(pts.size)
        for (i in pts.indices) {
            if (used[i]) continue
            var sx = pts[i].x
            var sy = pts[i].y
            var c = 1
            used[i] = true
            for (j in i + 1 until pts.size) {
                if (!used[j] && hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y) <= radius) {
                    sx += pts[j].x
                    sy += pts[j].y
                    c++
                    used[j] = true
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
