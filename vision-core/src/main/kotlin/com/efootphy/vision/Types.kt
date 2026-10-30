package com.efootphy.vision

import kotlin.math.hypot

data class Vec2(val x: Double, val y: Double) {
    init { require(x.isFinite() && y.isFinite()) }
    operator fun plus(other: Vec2) = Vec2(x + other.x, y + other.y)
    operator fun minus(other: Vec2) = Vec2(x - other.x, y - other.y)
    operator fun times(scale: Double) = Vec2(x * scale, y * scale)
    fun distance(other: Vec2) = hypot(x - other.x, y - other.y)
    fun inPitch() = x in 0.0..1.0 && y in 0.0..1.0
}

/** Coordinates must be de-letterboxed into the original image before construction. */
data class Box2(val left: Double, val top: Double, val right: Double, val bottom: Double) {
    init {
        require(listOf(left, top, right, bottom).all { it.isFinite() && it in 0.0..1.0 })
        require(right > left && bottom > top)
    }
    val feet get() = Vec2((left + right) / 2.0, bottom)
}

enum class Team { HOME, AWAY, UNKNOWN }
enum class Scene { LIVE, FORMATION, UNKNOWN }
enum class Source { MINIMAP, FIELD }
enum class Kind { PLAYER, PHYSICAL_BALL, BALL_MARKER, CONTROL_MARKER }
enum class TrackStatus { TENTATIVE, OBSERVED, COASTING }

/** atNs uses one monotonic clock. score is a model score, not a calibrated probability. */
data class Observation(
    val point: Vec2,
    val team: Team,
    val score: Double,
    val atNs: Long,
    val epoch: Long,
    val source: Source,
    val kind: Kind = Kind.PLAYER
) {
    init { require(score.isFinite() && score in 0.0..1.0 && atNs >= 0L) }
}

data class FieldDetection(
    val box: Box2,
    val team: Team,
    val score: Double,
    val atNs: Long,
    val epoch: Long,
    val kind: Kind = Kind.PLAYER
) {
    init { require(score.isFinite() && score in 0.0..1.0 && atNs >= 0L) }
}

data class FusionInput(
    val atNs: Long,
    val epoch: Long,
    val scene: Scene,
    val minimap: List<Observation> = emptyList(),
    val field: List<FieldDetection> = emptyList(),
    val calibration: Calibration? = null,
    // Only set after a frozen, independent evaluation and verified label semantics.
    val modelsValidated: Boolean = false
)

data class PlayerTrack(
    val id: Long,
    val team: Team,
    val point: Vec2,
    val velocityPerSecond: Vec2,
    val lastObservedNs: Long,
    val status: TrackStatus,
    val score: Double,
    val sources: Set<Source>
)

data class Guidance(val radarAllowed: Boolean, val fieldAllowed: Boolean, val reasons: Set<String>)
data class WorldState(
    val atNs: Long,
    val epoch: Long,
    val scene: Scene,
    val players: List<PlayerTrack>,
    val ball: Observation?,
    val fusedPairs: Int,
    val guidance: Guidance
)

object Timing {
    fun intervalNs(hz: Int): Long {
        require(hz in 1..60)
        return 1_000_000_000L / hz
    }
    fun fresh(nowNs: Long, observedNs: Long, maxAgeNs: Long): Boolean =
        observedNs >= 0 && nowNs >= observedNs && nowNs - observedNs <= maxAgeNs
}
