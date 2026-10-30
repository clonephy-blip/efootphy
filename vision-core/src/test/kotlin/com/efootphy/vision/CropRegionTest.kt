package com.efootphy.vision

import org.junit.Assert.*
import org.junit.Test

class CropRegionTest {
    @Test fun corruptPreferencesCannotInvertOrOverflowCrop() {
        val roi=CropRegion.restore(2f,-4f,-8f,3f)
        assertEquals(.98f,roi.left,.00001f)
        assertEquals(1f,roi.right,.00001f)
        assertEquals(0f,roi.top,.00001f)
        assertEquals(1f,roi.bottom,.00001f)
    }
    @Test fun nonfinitePreferencesFallBackToMeasuredDefault() {
        assertEquals(CropRegion.DEFAULT,CropRegion.restore(Float.NaN,0f,1f,1f))
    }
    @Test fun movingPreservesSizeAtDisplayEdge() {
        val roi=CropRegion.DEFAULT.moveX(1f).moveY(1f)
        assertEquals(CropRegion.DEFAULT.width,roi.width,.00001f)
        assertEquals(CropRegion.DEFAULT.height,roi.height,.00001f)
        assertEquals(1f,roi.right,.00001f)
        assertEquals(1f,roi.bottom,.00001f)
    }
    @Test fun smallestCropAtRightBottomEdgeStillResizesSafely() {
        val roi=CropRegion.restore(1f,1f,0f,0f).resizeWidth(0f).resizeHeight(0f)
        assertTrue(roi.width>=.01999f && roi.height>=.01999f)
        assertTrue(roi.right<=1f && roi.bottom<=1f)
    }
}
