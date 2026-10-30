package com.efootphy.vision

import org.junit.Assert.*
import org.junit.Test

class ProjectionTest {
    private val transform=Homography.of(doubleArrayOf(.8,.05,.1,-.02,.55,.2,.13,.08,1.0))!!
    private val grid=(0..2).flatMap { y -> (0..2).map { x -> Vec2(.1+x*.4,.1+y*.4) } }
    private fun landmarks()=grid.mapIndexed { index,p -> Landmark(index,p,transform.map(p)!!) }

    @Test fun fittedProjectiveTransformRoundTrips() {
        val fit=Homography.fit(grid,grid.map { transform.map(it)!! })!!
        val point=Vec2(.37,.69)
        assertTrue(fit.map(point)!!.distance(transform.map(point)!!)<1e-8)
        assertTrue(fit.inverse()!!.map(fit.map(point)!!)!!.distance(point)<1e-8)
    }
    @Test fun ransacRejectsTwoWrongLandmarks() {
        val points=landmarks()+listOf(Landmark(20,Vec2(.3,.2),Vec2(.9,.9)),Landmark(21,Vec2(.7,.8),Vec2(.05,.1)))
        val fit=PitchCalibration.estimate(points,2340,1080,1_000_000L,3)!!
        assertEquals(9,fit.inliers)
        assertTrue(fit.p95ErrorPx<1e-5)
        assertTrue(fit.usable(101_000_000L,3))
    }
    @Test fun calibrationExpiresAndIsScopedToCameraEpoch() {
        val fit=PitchCalibration.estimate(landmarks(),1280,720,1_000_000L,3)!!
        assertFalse(fit.usable(201_000_001L,3))
        assertFalse(fit.usable(0L,3))
        assertFalse(fit.usable(1_000_000L,4))
    }
    @Test fun duplicateLandmarkIdsRejected() {
        assertNull(PitchCalibration.estimate(landmarks().map { it.copy(id=0) },1280,720,0,0))
    }
    @Test fun collinearLandmarksRejected() {
        val points=(0..5).map { Vec2(it*.1,it*.1) }
        assertNull(Homography.fit(points,points))
    }
    @Test fun singularTransformHasNoInverse() {
        assertNull(Homography.of(DoubleArray(9))!!.inverse())
    }
    @Test fun pointsConcentratedInTinyRegionRejected() {
        val points=landmarks().map { it.copy(pitch=it.pitch*.01) }
        assertNull(PitchCalibration.estimate(points,1280,720,0,0))
    }
    @Test fun fourPointsAreNotEnoughForTrustedOverlay() {
        val corners=listOf(landmarks()[0],landmarks()[2],landmarks()[6],landmarks()[8])
        val fit=PitchCalibration.estimate(corners,1280,720,0,0)!!
        assertFalse(fit.usable(0,0))
    }
    @Test fun excessiveReprojectionErrorRejected() {
        val fit=PitchCalibration.estimate(landmarks(),1280,720,0,0)!!
        assertFalse(fit.copy(p95ErrorPx=3.01).usable(0,0))
    }
    @Test fun horizonProjectionIsUnknown() {
        val h=Homography.of(doubleArrayOf(1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,-.5))!!
        assertNull(h.map(Vec2(.5,.3)))
    }
}
