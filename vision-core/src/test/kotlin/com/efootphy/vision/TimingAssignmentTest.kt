package com.efootphy.vision

import org.junit.Assert.*
import org.junit.Test

class TimingAssignmentTest {
    @Test fun ratesUseOneBillionNanoseconds() {
        assertEquals(66_666_666L,Timing.intervalNs(15))
        assertEquals(16_666_666L,Timing.intervalNs(60))
    }
    @Test(expected=IllegalArgumentException::class) fun zeroRateRejected() { Timing.intervalNs(0) }
    @Test(expected=IllegalArgumentException::class) fun excessiveRateRejected() { Timing.intervalNs(61) }
    @Test fun freshnessRejectsFutureAndExpiredFrames() {
        assertTrue(Timing.fresh(200,100,100))
        assertFalse(Timing.fresh(201,100,100))
        assertFalse(Timing.fresh(90,100,100))
        assertFalse(Timing.fresh(90,-1,100))
    }
    @Test fun globalAssignmentAvoidsGreedyTrap() {
        assertArrayEquals(intArrayOf(1,0),Assignment.solve(listOf(doubleArrayOf(1.0,2.0),doubleArrayOf(1.1,100.0)),10.0))
    }
    @Test fun unmatchedAndRectangularCases() {
        assertArrayEquals(intArrayOf(-1,-1),Assignment.solve(listOf(doubleArrayOf(Double.POSITIVE_INFINITY),doubleArrayOf(5.0)),2.0))
        assertArrayEquals(intArrayOf(-1),Assignment.solve(listOf(DoubleArray(0)),2.0))
        assertArrayEquals(IntArray(0),Assignment.solve(emptyList(),2.0))
    }
    @Test(expected=IllegalArgumentException::class) fun raggedMatrixRejected() {
        Assignment.solve(listOf(doubleArrayOf(1.0),DoubleArray(0)),2.0)
    }
    @Test(expected=IllegalArgumentException::class) fun nanCostRejected() {
        Assignment.solve(listOf(doubleArrayOf(Double.NaN)),2.0)
    }
}
