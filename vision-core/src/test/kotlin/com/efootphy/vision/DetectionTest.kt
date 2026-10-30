package com.efootphy.vision

import org.junit.Assert.*
import org.junit.Test

class DetectionTest {
    @Test fun landscapeLetterboxRoundTripRestoresOriginalCenter() {
        val box=Letterbox(274,164,320)
        val target=box.restore(137*box.scale+box.left,82*box.scale+box.top,
            20*box.scale,10*box.scale)!!
        assertEquals(.5,target.feet.x,1e-9)
        assertEquals((82+5.0)/164,target.feet.y,1e-9)
    }
    @Test fun paddingDetectionIsDiscarded() {
        assertNull(Letterbox(274,164,320).restore(100.0,5.0,10.0,10.0))
    }
    @Test fun malformedDetectionDiscarded() {
        val box=Letterbox(320,320,320)
        assertNull(box.restore(Double.NaN,100.0,20.0,20.0))
        assertNull(box.restore(100.0,100.0,-1.0,20.0))
    }
    @Test fun edgeBoxIsClippedWithoutMovingItsVisibleCenterToAnUnrelatedPlace() {
        val out=Letterbox(320,320,320).restore(5.0,160.0,30.0,20.0)!!
        assertEquals(0.0,out.left,0.0)
        assertEquals(20.0/320,out.right,1e-9)
    }
    @Test fun sameClassDuplicateSuppressedButDifferentMarkerRolesPreserved() {
        val raw=arrayOf(floatArrayOf(160f,160f),floatArrayOf(160f,160f),floatArrayOf(20f,20f),
            floatArrayOf(20f,20f),floatArrayOf(.9f,.8f),floatArrayOf(.7f,.6f),floatArrayOf(.1f,.1f))
        val out=DetectorPostprocess.decode(raw,Letterbox(320,320,320),3)
        assertEquals(2,out.size)
        assertEquals(setOf(0,1),out.map { it.label }.toSet())
    }
    @Test fun nanScoresAreNotEmitted() {
        val raw=arrayOf(floatArrayOf(160f),floatArrayOf(160f),floatArrayOf(20f),floatArrayOf(20f),floatArrayOf(Float.NaN))
        assertTrue(DetectorPostprocess.decode(raw,Letterbox(320,320,320),1).isEmpty())
    }
    @Test(expected=IllegalArgumentException::class) fun unexpectedOutputLayoutRejected() {
        DetectorPostprocess.decode(arrayOf(floatArrayOf(1f)),Letterbox(320,320,320),3)
    }
}
