package com.efootphy.vision

import kotlin.math.hypot

data class FusionPolicy(
    val observationMaxAgeNs: Long = 180_000_000L,
    val sensorSyncMaxNs: Long = 60_000_000L,
    val coastMaxNs: Long = 300_000_000L,
    val fusionDistance: Double = 0.035,
    val trackDistance: Double = 0.055,
    val minimumScore: Double = 0.5,
    val confirmationHits: Int = 3,
    val requiredPlayersPerTeam: Int = 11
) {
    init {
        require(observationMaxAgeNs>0 && sensorSyncMaxNs>=0 && coastMaxNs>=observationMaxAgeNs)
        require(fusionDistance>0 && trackDistance>0 && minimumScore in 0.0..1.0)
        require(confirmationHits>0 && requiredPlayersPerTeam in 1..11)
    }
}

/** All fused/tracked coordinates are on the same normalized, calibrated pitch. */
class FusionEngine(private val policy: FusionPolicy = FusionPolicy()) {
    private data class Measurement(val point: Vec2,val team: Team,val score: Double,val atNs: Long,val sources: Set<Source>)
    private class Track(
        val id: Long,
        var point: Vec2,
        var team: Team,
        var score: Double,
        var lastObservedNs: Long,
        var sources: Set<Source>,
        var velocity: Vec2 = Vec2(0.0,0.0),
        var hits: Int = 1,
        var confirmed: Boolean = false
    )
    private val tracks=mutableListOf<Track>()
    private var nextId=1L
    private var previous: WorldState?=null
    private val consumedSensorTime=mutableMapOf<Source,Long>()

    fun reset() { tracks.clear(); consumedSensorTime.clear(); previous=null }

    fun update(input: FusionInput): WorldState {
        require(input.atNs>=0)
        val old=previous
        if(old!=null && old.epoch==input.epoch && input.atNs<=old.atNs) return old
        if(old!=null && old.epoch!=input.epoch) { tracks.clear(); consumedSensorTime.clear() }
        if(input.scene!=Scene.LIVE) {
            tracks.clear()
            consumedSensorTime.clear()
            return WorldState(input.atNs,input.epoch,input.scene,emptyList(),null,0,
                Guidance(false,false,setOf("not_live"))).also { previous=it }
        }
        fun valid(atNs: Long,epoch: Long,score: Double) = epoch==input.epoch &&
            Timing.fresh(input.atNs,atNs,policy.observationMaxAgeNs) && score>=policy.minimumScore
        val map=input.minimap.filter { it.source==Source.MINIMAP && it.kind==Kind.PLAYER &&
            it.point.inPitch() && valid(it.atNs,it.epoch,it.score) &&
            it.atNs>(consumedSensorTime[Source.MINIMAP] ?: -1L) }
        val calibration=input.calibration?.takeIf { it.usable(input.atNs,input.epoch) }
        val field=if(calibration==null) emptyList() else input.field.mapNotNull { detection ->
            if(detection.kind!=Kind.PLAYER || !valid(detection.atNs,detection.epoch,detection.score) ||
                detection.atNs<=(consumedSensorTime[Source.FIELD] ?: -1L)) return@mapNotNull null
            // A camera estimate from another time must not project a player during a pan/cut.
            if(kotlin.math.abs(detection.atNs-calibration.atNs)>policy.sensorSyncMaxNs) return@mapNotNull null
            val point=calibration.imageToPitch.map(detection.box.feet) ?: return@mapNotNull null
            if(!point.inPitch()) return@mapNotNull null
            Observation(point,detection.team,detection.score,detection.atNs,detection.epoch,Source.FIELD)
        }
        map.maxOfOrNull { it.atNs }?.let { consumedSensorTime[Source.MINIMAP]=it }
        field.maxOfOrNull { it.atNs }?.let { consumedSensorTime[Source.FIELD]=it }
        fun compatible(a: Team,b: Team) = a==b || a==Team.UNKNOWN || b==Team.UNKNOWN
        val costs=map.map { a -> DoubleArray(field.size) { index ->
            val b=field[index]
            val distance=a.point.distance(b.point)
            if(compatible(a.team,b.team) && kotlin.math.abs(a.atNs-b.atNs)<=policy.sensorSyncMaxNs &&
                distance<=policy.fusionDistance) distance else Double.POSITIVE_INFINITY
        } }
        val paired=Assignment.solve(costs,policy.fusionDistance+1e-9)
        val used=BooleanArray(field.size)
        val measurements=mutableListOf<Measurement>()
        var fused=0
        for(i in map.indices) {
            val a=map[i];val index=paired[i]
            if(index<0) measurements.add(Measurement(a.point,a.team,a.score,a.atNs,setOf(a.source)))
            else {
                val b=field[index];used[index]=true;fused++
                val point=(a.point*a.score+b.point*b.score)*(1.0/(a.score+b.score))
                measurements.add(Measurement(point,if(a.team==Team.UNKNOWN) b.team else a.team,
                    maxOf(a.score,b.score),minOf(a.atNs,b.atNs),setOf(Source.MINIMAP,Source.FIELD)))
            }
        }
        for(i in field.indices) if(!used[i]) {
            val b=field[i];measurements.add(Measurement(b.point,b.team,b.score,b.atNs,setOf(b.source)))
        }
        tracks.removeAll { !Timing.fresh(input.atNs,it.lastObservedNs,policy.coastMaxNs) }
        val trackCosts=tracks.map { track -> DoubleArray(measurements.size) { index ->
            val m=measurements[index]
            if(!compatible(track.team,m.team) || m.atNs<track.lastObservedNs) Double.POSITIVE_INFINITY
            else {
                val seconds=(m.atNs-track.lastObservedNs)/1e9
                val prediction=track.point+track.velocity*seconds
                prediction.distance(m.point).let { if(it<=policy.trackDistance) it else Double.POSITIVE_INFINITY }
            }
        } }
        val matched=Assignment.solve(trackCosts,policy.trackDistance+1e-9)
        val usedMeasurements=BooleanArray(measurements.size)
        val observedIds=mutableSetOf<Long>()
        for(i in tracks.indices) {
            val index=matched[i]
            if(index<0) continue
            usedMeasurements[index]=true
            val track=tracks[i];val m=measurements[index]
            // Repeated source timestamps must not increase hit count or manufacture freshness.
            if(m.atNs<=track.lastObservedNs) {
                track.sources=track.sources+m.sources
                track.score=maxOf(track.score,m.score)
                if(track.team==Team.UNKNOWN) track.team=m.team
                continue
            }
            val dt=(m.atNs-track.lastObservedNs)/1e9
            val predicted=track.point+track.velocity*dt
            val residual=m.point-predicted
            track.point=predicted+residual*0.65
            val velocity=track.velocity+residual*(0.25/dt)
            val speed=hypot(velocity.x,velocity.y)
            track.velocity=if(speed>1.0) velocity*(1.0/speed) else velocity
            track.team=if(track.team==Team.UNKNOWN) m.team else track.team
            track.score=m.score;track.lastObservedNs=m.atNs;track.sources=m.sources
            track.hits++
            track.confirmed=track.confirmed || track.hits>=policy.confirmationHits
            observedIds.add(track.id)
        }
        for(i in measurements.indices) if(!usedMeasurements[i]) {
            val m=measurements[i]
            val track=Track(nextId++,m.point,m.team,m.score,m.atNs,m.sources,
                confirmed=policy.confirmationHits<=1)
            tracks.add(track);observedIds.add(track.id)
        }
        val snapshot=tracks.map { track ->
            val dt=(input.atNs-track.lastObservedNs)/1e9
            PlayerTrack(track.id,track.team,track.point+track.velocity*dt,track.velocity,
                track.lastObservedNs,when {
                    !track.confirmed -> TrackStatus.TENTATIVE
                    track.id in observedIds -> TrackStatus.OBSERVED
                    else -> TrackStatus.COASTING
                },track.score,track.sources.toSet())
        }
        // A yellow/control marker is never silently promoted to a physical-ball observation.
        val ball=input.minimap.filter { it.kind==Kind.PHYSICAL_BALL && it.source==Source.MINIMAP &&
            it.point.inPitch() && valid(it.atNs,it.epoch,it.score) }.maxByOrNull { it.score }
        val reasons=linkedSetOf<String>()
        if(!input.modelsValidated) reasons.add("models_unvalidated")
        if(ball==null) reasons.add("ball_unknown")
        val usable=snapshot.filter { it.status!=TrackStatus.TENTATIVE && it.point.inPitch() &&
            Timing.fresh(input.atNs,it.lastObservedNs,policy.observationMaxAgeNs) }
        for(team in listOf(Team.HOME,Team.AWAY)) {
            val count=usable.count { it.team==team }
            if(count<policy.requiredPlayersPerTeam) reasons.add("incomplete_${team.name.lowercase()}")
            if(count>11) reasons.add("overcount_${team.name.lowercase()}")
        }
        if(usable.any { it.team==Team.UNKNOWN }) reasons.add("team_unknown")
        val radarAllowed=reasons.isEmpty()
        val fieldAllowed=radarAllowed && calibration!=null
        if(calibration==null) reasons.add("camera_uncalibrated")
        return WorldState(input.atNs,input.epoch,input.scene,snapshot,ball,fused,
            Guidance(radarAllowed,fieldAllowed,reasons.toSet())).also { previous=it }
    }
}
