package com.example.minimap.hud

import org.json.JSONArray
import org.json.JSONObject

data class BallData(
    val x: Float,
    val y: Float,
    val vx: Float = 0f,
    val vy: Float = 0f
)

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
    val fps: Float = 60f,
    val latencyMs: Float = 0f,
    val status: String = "PC Brain Active",
    val ball: BallData? = null,
    val players: List<PlayerData> = emptyList(),
    val passLines: List<PassLine> = emptyList()
) {
    companion object {
        fun fromJson(jsonStr: String): HudPacket? {
            return try {
                val obj = JSONObject(jsonStr)
                val fps = obj.optDouble("fps", 60.0).toFloat()
                val lat = obj.optDouble("latency", 0.0).toFloat()
                val status = obj.optString("status", "PC Brain Active")
                
                var ball: BallData? = null
                if (obj.has("ball") && !obj.isNull("ball")) {
                    val bObj = obj.getJSONObject("ball")
                    ball = BallData(
                        bObj.getDouble("x").toFloat(),
                        bObj.getDouble("y").toFloat(),
                        bObj.optDouble("vx", 0.0).toFloat(),
                        bObj.optDouble("vy", 0.0).toFloat()
                    )
                }

                val players = mutableListOf<PlayerData>()
                val pArray = obj.optJSONArray("players") ?: JSONArray()
                for (i in 0 until pArray.length()) {
                    val p = pArray.getJSONObject(i)
                    players.add(
                        PlayerData(
                            x1 = p.getDouble("x1").toFloat(),
                            y1 = p.getDouble("y1").toFloat(),
                            x2 = p.getDouble("x2").toFloat(),
                            y2 = p.getDouble("y2").toFloat(),
                            team = p.optInt("team", 0),
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

                HudPacket(
                    fps = fps,
                    latencyMs = lat,
                    status = status,
                    ball = ball,
                    players = players,
                    passLines = passes
                )
            } catch (e: Exception) {
                null
            }
        }
    }
}
