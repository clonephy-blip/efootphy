package com.example.minimap.hud

import org.json.JSONArray
import org.json.JSONObject

data class BallData(
    val x: Float,
    val y: Float,
    val vx: Float = 0f,
    val vy: Float = 0f,
    val state: String = "OBSERVED"
)

data class ShadowCandidateData(val x: Float, val y: Float, val quality: Float)

data class PlayerData(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val team: Int, // 0: Team A (Home/Cyan), 1: Team B (Away/Red), 2: Goalkeeper (Yellow), 3: Referee
    val label: String = "",
    val hasBall: Boolean = false,
    val isControlled: Boolean = false
) {
    val feetX: Float get() = (x1 + x2) / 2f
    val feetY: Float get() = y2
    val radiusX: Float get() = (x2 - x1) * 0.45f
    val radiusY: Float get() = (x2 - x1) * 0.22f
}

data class PassLine(
    val fromX: Float,
    val fromY: Float,
    val toX: Float,
    val toY: Float,
    val safe: Boolean = true
)

data class HudPacket(
    val timestamp: Long = System.currentTimeMillis(),
    val receivedAtMs: Long = android.os.SystemClock.elapsedRealtime(),
    val fps: Float = 60f,
    val latencyMs: Float = 0f,
    val status: String = "PC Brain Active",
    val ball: BallData? = null,
    val players: List<PlayerData> = emptyList(),
    val passLines: List<PassLine> = emptyList(),
    val sequence: Long = -1L,
    val transportRttMs: Float = -1f,
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
    val sourceAgeMs: Float = 0f,
    val shadowCandidates: List<ShadowCandidateData> = emptyList()
) {
    companion object {
        fun fromJson(jsonStr: String): HudPacket? {
            return try {
                val obj = JSONObject(jsonStr)
                val timestamp = obj.optLong("timestamp", System.currentTimeMillis())
                val receivedAt = android.os.SystemClock.elapsedRealtime()
                val fps = obj.optDouble("fps", 60.0).toFloat()
                val lat = obj.optDouble("latency", 0.0).toFloat()
                val status = obj.optString("status", "PC Brain Active")
                val seq = obj.optLong("seq", -1L).takeIf { it >= 0 } ?: -1L
                val rawRtt = obj.optDouble("transport_rtt_ms", -1.0)
                val transportRtt = if (rawRtt.isFinite() && rawRtt in 0.0..10000.0) rawRtt.toFloat() else -1f
                
                var ball: BallData? = null
                if (obj.has("ball") && !obj.isNull("ball")) {
                    val bObj = obj.getJSONObject("ball")
                    ball = BallData(
                        bObj.getDouble("x").toFloat(),
                        bObj.getDouble("y").toFloat(),
                        bObj.optDouble("vx", 0.0).toFloat(),
                        bObj.optDouble("vy", 0.0).toFloat(),
                        bObj.optString("state", "OBSERVED").also { require(it in setOf("OBSERVED", "PREDICTED")) }
                    )
                    require(ball.x.isFinite() && ball.y.isFinite() && ball.x in 0f..1f && ball.y in 0f..1f && ball.vx.isFinite() && ball.vy.isFinite())
                }

                val players = mutableListOf<PlayerData>()
                val pArray = obj.optJSONArray("players") ?: JSONArray()
                for (i in 0 until pArray.length()) {
                    val p = pArray.getJSONObject(i)
                    val x1 = if (p.has("x1")) p.optDouble("x1", 0.0).toFloat() else (p.optDouble("feet_x", 0.5) - p.optDouble("radius_x", 0.04)).toFloat()
                    val x2 = if (p.has("x2")) p.optDouble("x2", 0.0).toFloat() else (p.optDouble("feet_x", 0.5) + p.optDouble("radius_x", 0.04)).toFloat()
                    val y2 = if (p.has("y2")) p.optDouble("y2", 0.0).toFloat() else p.optDouble("feet_y", 0.5).toFloat()
                    val y1 = if (p.has("y1")) p.optDouble("y1", 0.0).toFloat() else (y2 - 0.08f)

                    players.add(
                        PlayerData(
                            x1 = x1,
                            y1 = y1,
                            x2 = x2,
                            y2 = y2,
                            team = p.optInt("team", -1),
                            label = p.optString("label", ""),
                            hasBall = p.optBoolean("has_ball", false),
                            isControlled = p.optBoolean("controlled", false)
                        )
                    )
                }

                val passes = mutableListOf<PassLine>()
                val passArray = obj.optJSONArray("pass_lines") ?: JSONArray()
                for (i in 0 until passArray.length()) {
                    val pl = passArray.getJSONObject(i)
                    passes.add(
                        PassLine(
                            fromX = pl.getDouble("from_x").toFloat(),
                            fromY = pl.getDouble("from_y").toFloat(),
                            toX = pl.getDouble("to_x").toFloat(),
                            toY = pl.getDouble("to_y").toFloat(),
                            safe = pl.optBoolean("safe", true)
                        )
                    )
                }

                val shadows = mutableListOf<ShadowCandidateData>()
                val shadowArray = obj.optJSONArray("shadow_candidates") ?: JSONArray()
                require(shadowArray.length() <= 8)
                for(i in 0 until shadowArray.length()) {
                    val sh = shadowArray.getJSONObject(i)
                    val x = sh.getDouble("x").toFloat(); val y = sh.getDouble("y").toFloat()
                    val q = sh.getDouble("quality").toFloat()
                    require(x.isFinite() && y.isFinite() && q.isFinite() && x in 0f..1f && y in 0f..1f && q in 0f..1f)
                    shadows.add(ShadowCandidateData(x, y, q))
                }
                val age = obj.optDouble("source_age_ms", 0.0).toFloat()
                require(age.isFinite() && age in 0f..60000f)
                HudPacket(
                    timestamp = timestamp,
                    receivedAtMs = receivedAt,
                    fps = fps,
                    latencyMs = lat,
                    status = status,
                    ball = ball,
                    players = players,
                    passLines = passes,
                    sequence = seq,
                    transportRttMs = transportRtt,
                    sourceWidth = obj.optInt("frame_width", 0),
                    sourceHeight = obj.optInt("frame_height", 0),
                    sourceAgeMs = age,
                    shadowCandidates = shadows
                )
            } catch (e: Exception) {
                null
            }
        }
    }
}
