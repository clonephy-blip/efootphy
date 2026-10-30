package com.efootphy.vision

import org.junit.Assert.*
import org.junit.Test

class FusionEngineTest {
    private fun obs(x: Double=.2,y: Double=.3,t: Long=0,team: Team=Team.HOME,epoch: Long=0,kind: Kind=Kind.PLAYER)=
        Observation(Vec2(x,y),team,.95,t,epoch,Source.MINIMAP,kind)
    private fun field(x: Double=.2,y: Double=.3,t: Long=0,team: Team=Team.HOME,epoch: Long=0)=
        FieldDetection(Box2(x-.01,y-.04,x+.01,y),team,.9,t,epoch)
    private fun camera(t: Long=0,epoch: Long=0): Calibration {
        val h=Homography.of(doubleArrayOf(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))!!
        return Calibration(h,h,t,epoch,8,8,0.0,1280,720)
    }
    private fun engine()=FusionEngine(FusionPolicy(confirmationHits=2,requiredPlayersPerTeam=1))

    @Test fun samePlayerFromTwoViewsIsCountedOnce() {
        val world=engine().update(FusionInput(0,0,Scene.LIVE,listOf(obs()),listOf(field()),camera()))
        assertEquals(1,world.players.size)
        assertEquals(1,world.fusedPairs)
        assertEquals(setOf(Source.MINIMAP,Source.FIELD),world.players.single().sources)
    }
    @Test fun opposingTeamsAreNotMergedDespiteProximity() {
        val world=engine().update(FusionInput(0,0,Scene.LIVE,listOf(obs()),listOf(field(team=Team.AWAY)),camera()))
        assertEquals(2,world.players.size)
        assertEquals(0,world.fusedPairs)
    }
    @Test fun uncalibratedFieldDetectionDoesNotInventPitchCoordinates() {
        assertTrue(engine().update(FusionInput(0,0,Scene.LIVE,field=listOf(field()))).players.isEmpty())
    }
    @Test fun usesFeetRatherThanBoundingBoxCenter() {
        val world=engine().update(FusionInput(0,0,Scene.LIVE,field=listOf(field(.4,.6)),calibration=camera()))
        assertEquals(.6,world.players.single().point.y,1e-9)
    }
    @Test fun cameraAndPlayerFromDifferentTimesAreNotCombined() {
        val world=engine().update(FusionInput(100_000_000L,0,Scene.LIVE,
            field=listOf(field(t=100_000_000L)),calibration=camera(0)))
        assertTrue(world.players.isEmpty())
    }
    @Test fun cameraEpochChangeDropsOldTracks() {
        val e=engine();val first=e.update(FusionInput(0,0,Scene.LIVE,listOf(obs())))
        val second=e.update(FusionInput(10,1,Scene.LIVE,listOf(obs(epoch=1,t=10))))
        assertEquals(1,second.players.size)
        assertNotEquals(first.players.single().id,second.players.single().id)
    }
    @Test fun menuClearsWorldAndDisablesGuidance() {
        val e=engine();e.update(FusionInput(0,0,Scene.LIVE,listOf(obs())))
        val world=e.update(FusionInput(10,0,Scene.FORMATION))
        assertTrue(world.players.isEmpty())
        assertFalse(world.guidance.radarAllowed)
    }
    @Test fun oldAndFutureObservationsRejected() {
        val world=engine().update(FusionInput(400_000_000L,0,Scene.LIVE,
            listOf(obs(t=0),obs(t=500_000_000L))))
        assertTrue(world.players.isEmpty())
    }
    @Test fun markersAreNotPhysicalBallsOrExtraPlayers() {
        val world=engine().update(FusionInput(0,0,Scene.LIVE,
            listOf(obs(kind=Kind.BALL_MARKER),obs(kind=Kind.CONTROL_MARKER))))
        assertNull(world.ball)
        assertTrue(world.players.isEmpty())
        assertTrue("ball_unknown" in world.guidance.reasons)
    }
    @Test fun unvalidatedModelsCannotEnableAdvice() {
        val e=engine()
        fun input(t: Long)=FusionInput(t,0,Scene.LIVE,listOf(obs(t=t),obs(x=.7,t=t,team=Team.AWAY),
            obs(x=.5,t=t,kind=Kind.PHYSICAL_BALL)))
        e.update(input(0));val world=e.update(input(40_000_000L))
        assertFalse(world.guidance.radarAllowed)
        assertTrue("models_unvalidated" in world.guidance.reasons)
    }
    @Test fun syntheticQualifiedInputAllowsRadarButNotUncalibratedAr() {
        val e=engine()
        fun input(t: Long)=FusionInput(t,0,Scene.LIVE,listOf(obs(t=t),obs(x=.7,t=t,team=Team.AWAY),
            obs(x=.5,t=t,kind=Kind.PHYSICAL_BALL)),modelsValidated=true)
        e.update(input(0));val world=e.update(input(40_000_000L))
        assertTrue(world.guidance.radarAllowed)
        assertFalse(world.guidance.fieldAllowed)
    }
    @Test fun oneTeamMissingBlocksAdvice() {
        val e=engine()
        fun input(t: Long)=FusionInput(t,0,Scene.LIVE,
            listOf(obs(t=t),obs(x=.5,t=t,kind=Kind.PHYSICAL_BALL)),modelsValidated=true)
        e.update(input(0));val world=e.update(input(40_000_000L))
        assertFalse(world.guidance.radarAllowed)
        assertTrue("incomplete_away" in world.guidance.reasons)
    }
    @Test fun repeatedMapWithNewFieldUpdatesDoesNotCreateGhostIds() {
        val e=engine();val id=e.update(FusionInput(0,0,Scene.LIVE,listOf(obs()))).players.single().id
        for(t in listOf(20_000_000L,40_000_000L,80_000_000L)) {
            val world=e.update(FusionInput(t,0,Scene.LIVE,listOf(obs()),listOf(field(t=t)),camera(t)))
            assertEquals(1,world.players.size)
            assertEquals(id,world.players.single().id)
            assertEquals(t,world.players.single().lastObservedNs)
        }
    }
    @Test fun outOfOrderAndDuplicateFramesNeverRefreshState() {
        val e=engine();val state=e.update(FusionInput(100,0,Scene.LIVE,listOf(obs(t=100))))
        assertSame(state,e.update(FusionInput(90,0,Scene.LIVE,listOf(obs(t=90)))))
        assertSame(state,e.update(FusionInput(100,0,Scene.LIVE,listOf(obs(t=100)))))
    }
    @Test fun temporaryOcclusionCoastsAndThenExpires() {
        val e=engine();val id=e.update(FusionInput(0,0,Scene.LIVE,listOf(obs()))).players.single().id
        e.update(FusionInput(40_000_000L,0,Scene.LIVE,listOf(obs(t=40_000_000L))))
        val coast=e.update(FusionInput(90_000_000L,0,Scene.LIVE))
        assertEquals(TrackStatus.COASTING,coast.players.single().status)
        assertEquals(id,coast.players.single().id)
        assertTrue(e.update(FusionInput(600_000_000L,0,Scene.LIVE)).players.isEmpty())
    }
    @Test fun velocityUsesElapsedTimeAtDifferentSampleRates() {
        fun run(hz: Int): PlayerTrack {
            val e=engine();var world: WorldState?=null
            for(i in 0..hz) {
                val t=(i*1e9/hz).toLong()
                world=e.update(FusionInput(t,0,Scene.LIVE,listOf(obs(x=.2+.15*i/hz,t=t))))
            }
            return world!!.players.single()
        }
        val a=run(15);val b=run(30)
        assertEquals(.15,a.velocityPerSecond.x,.025)
        assertEquals(.15,b.velocityPerSecond.x,.025)
        assertTrue(a.point.distance(b.point)<.02)
    }
    @Test fun twentyTwoMapPlayersAndPartialFieldViewRemainTwentyTwo() {
        val e=FusionEngine(FusionPolicy(confirmationHits=2))
        fun map(t: Long)=(0..10).flatMap { i -> listOf(obs(x=.1+i*.075,y=.25,t=t),
            obs(x=.1+i*.075,y=.75,t=t,team=Team.AWAY)) }
        e.update(FusionInput(0,0,Scene.LIVE,map(0)))
        val t=50_000_000L
        val partial=map(t).take(6).map { field(it.point.x,it.point.y,t,it.team) }
        val world=e.update(FusionInput(t,0,Scene.LIVE,map(t),partial,camera(t)))
        assertEquals(22,world.players.size)
        assertEquals(6,world.fusedPairs)
        assertEquals(11,world.players.count { it.team==Team.HOME })
        assertEquals(11,world.players.count { it.team==Team.AWAY })
    }
    @Test fun outsidePitchObservationsAreNotClampedIntoPlay() {
        assertTrue(engine().update(FusionInput(0,0,Scene.LIVE,listOf(obs(x=1.2)))).players.isEmpty())
    }
}
