package com.efootphy.vision

import java.util.Random
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.sqrt

class Homography private constructor(private val h: DoubleArray) {
    fun map(point: Vec2): Vec2? {
        val z = h[6] * point.x + h[7] * point.y + h[8]
        if (abs(z) < 1e-10) return null
        val x = (h[0] * point.x + h[1] * point.y + h[2]) / z
        val y = (h[3] * point.x + h[4] * point.y + h[5]) / z
        return if (x.isFinite() && y.isFinite()) Vec2(x, y) else null
    }

    fun inverse(): Homography? {
        val a=h[0]; val b=h[1]; val c=h[2]; val d=h[3]; val e=h[4]; val f=h[5]
        val g=h[6]; val i=h[7]; val j=h[8]
        val det=a*(e*j-f*i)-b*(d*j-f*g)+c*(d*i-e*g)
        if (abs(det) < 1e-12) return null
        return of(doubleArrayOf(e*j-f*i,c*i-b*j,b*f-c*e,f*g-d*j,a*j-c*g,c*d-a*f,d*i-e*g,b*g-a*i,a*e-b*d).map { it/det }.toDoubleArray())
    }

    companion object {
        fun of(values: DoubleArray): Homography? =
            if (values.size == 9 && values.all { it.isFinite() }) Homography(values.copyOf()) else null

        private fun multiply(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(9) { k ->
            (0..2).sumOf { i -> a[(k/3)*3+i]*b[i*3+k%3] }
        }

        /** Hartley-normalized least squares; singular/collinear correspondences are rejected. */
        fun fit(from: List<Vec2>, to: List<Vec2>): Homography? {
            if (from.size != to.size || from.size < 4) return null
            fun normalization(points: List<Vec2>): DoubleArray? {
                val cx=points.map { it.x }.average(); val cy=points.map { it.y }.average()
                val distance=points.map { hypot(it.x-cx,it.y-cy) }.average()
                if (distance < 1e-10) return null
                val s=sqrt(2.0)/distance
                return doubleArrayOf(s,0.0,-s*cx,0.0,s,-s*cy,0.0,0.0,1.0)
            }
            val t1=normalization(from) ?: return null
            val t2=normalization(to) ?: return null
            val a=Array(8) { DoubleArray(9) }
            for (k in from.indices) {
                val x=of(t1)!!.map(from[k])!!; val y=of(t2)!!.map(to[k])!!
                val rows=arrayOf(
                    doubleArrayOf(x.x,x.y,1.0,0.0,0.0,0.0,-y.x*x.x,-y.x*x.y),
                    doubleArrayOf(0.0,0.0,0.0,x.x,x.y,1.0,-y.y*x.x,-y.y*x.y)
                )
                for (r in 0..1) for (i in 0..7) {
                    for (j in 0..7) a[i][j]+=rows[r][i]*rows[r][j]
                    a[i][8]+=rows[r][i]*(if(r==0) y.x else y.y)
                }
            }
            for (col in 0..7) {
                val pivot=(col..7).maxBy { abs(a[it][col]) }
                if (abs(a[pivot][col]) < 1e-10) return null
                val swap=a[col]; a[col]=a[pivot]; a[pivot]=swap
                val scale=a[col][col]
                for (j in col..8) a[col][j]/=scale
                for (row in 0..7) if(row!=col) {
                    val factor=a[row][col]
                    for(j in col..8) a[row][j]-=factor*a[col][j]
                }
            }
            val h=DoubleArray(9) { if(it==8) 1.0 else a[it][8] }
            val inverseT2=of(t2)!!.inverse() ?: return null
            val out=multiply(multiply(inverseT2.h,h),t1)
            if (abs(out[8]) < 1e-10) return null
            val result=of(out.map { it/out[8] }.toDoubleArray()) ?: return null
            return result.takeIf { it.inverse()!=null }
        }
    }
}

data class Landmark(val id: Int, val pitch: Vec2, val image: Vec2, val score: Double = 1.0)
data class Calibration(
    val pitchToImage: Homography,
    val imageToPitch: Homography,
    val atNs: Long,
    val epoch: Long,
    val inliers: Int,
    val candidates: Int,
    val p95ErrorPx: Double,
    val width: Int,
    val height: Int
) {
    fun usable(nowNs: Long, currentEpoch: Long): Boolean =
        epoch==currentEpoch && Timing.fresh(nowNs,atNs,200_000_000L) && inliers>=6 &&
            candidates>=inliers && inliers.toDouble()/candidates>=0.65 &&
            p95ErrorPx.isFinite() && p95ErrorPx<=3.0 && width>0 && height>0
}

object PitchCalibration {
    /** Fits pitch -> original-image coordinates; RANSAC error is in original-image pixels. */
    fun estimate(points: List<Landmark>, width: Int, height: Int, atNs: Long, epoch: Long): Calibration? {
        if(width<=0 || height<=0 || atNs<0 || points.map { it.id }.distinct().size!=points.size) return null
        val clean=points.filter { it.score.isFinite() && it.score in 0.5..1.0 && it.pitch.inPitch() && it.image.inPitch() }
        if(clean.size<4) return null
        fun residual(model: Homography,p: Landmark): Double {
            val projected=model.map(p.pitch) ?: return Double.POSITIVE_INFINITY
            return hypot((projected.x-p.image.x)*width,(projected.y-p.image.y)*height)
        }
        val random=Random(20261030L)
        var best=emptyList<Landmark>()
        var bestError=Double.POSITIVE_INFINITY
        repeat(if(clean.size==4) 1 else 160) {
            val selected=linkedSetOf<Int>()
            while(selected.size<4) selected.add(random.nextInt(clean.size))
            val sample=selected.map { clean[it] }
            val model=Homography.fit(sample.map { it.pitch },sample.map { it.image }) ?: return@repeat
            val inliers=clean.filter { residual(model,it)<=3.0 }
            val error=inliers.sumOf { residual(model,it) }
            if(inliers.size>best.size || inliers.size==best.size && error<bestError) {
                best=inliers; bestError=error
            }
        }
        if(best.size<4) return null
        fun coverage(values: List<Vec2>): Double =
            (values.maxOf { it.x }-values.minOf { it.x })*(values.maxOf { it.y }-values.minOf { it.y })
        if(coverage(best.map { it.pitch })<0.025 || coverage(best.map { it.image })<0.015) return null
        val model=Homography.fit(best.map { it.pitch },best.map { it.image }) ?: return null
        val inverse=model.inverse() ?: return null
        val errors=best.map { residual(model,it) }.sorted()
        val p95=errors[ceil((errors.size-1)*0.95).toInt()]
        return Calibration(model,inverse,atNs,epoch,best.size,clean.size,p95,width,height)
    }
}
