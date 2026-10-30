package com.efootphy.vision

import kotlin.math.round

data class Detection(val box: Box2,val label: Int,val score: Double)

class Letterbox(val width: Int,val height: Int,val target: Int) {
    init { require(width>0 && height>0 && target>0) }
    val scale=minOf(target.toDouble()/width,target.toDouble()/height)
    // Match Python/OpenCV preprocessing: ties round to even (e.g. 62.5 -> 62).
    val resizedWidth=round(width*scale).toInt().coerceAtLeast(1)
    val resizedHeight=round(height*scale).toInt().coerceAtLeast(1)
    val left=(target-resizedWidth)/2
    val top=(target-resizedHeight)/2

    /** Raw YOLO xywh in model pixels -> original-ROI normalized xyxy. */
    fun restore(cx: Double,cy: Double,w: Double,h: Double): Box2? {
        if(!listOf(cx,cy,w,h).all { it.isFinite() } || w<=0 || h<=0) return null
        val centerX=(cx-left)/scale
        val centerY=(cy-top)/scale
        if(centerX !in 0.0..width.toDouble() || centerY !in 0.0..height.toDouble()) return null
        val x1=((cx-w/2-left)/scale/width).coerceIn(0.0,1.0)
        val x2=((cx+w/2-left)/scale/width).coerceIn(0.0,1.0)
        val y1=((cy-h/2-top)/scale/height).coerceIn(0.0,1.0)
        val y2=((cy+h/2-top)/scale/height).coerceIn(0.0,1.0)
        return if(x2>x1 && y2>y1) Box2(x1,y1,x2,y2) else null
    }
}

object DetectorPostprocess {
    fun iou(a: Box2,b: Box2): Double {
        val overlap=(minOf(a.right,b.right)-maxOf(a.left,b.left)).coerceAtLeast(0.0)*
            (minOf(a.bottom,b.bottom)-maxOf(a.top,b.top)).coerceAtLeast(0.0)
        val union=(a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-overlap
        return if(union>0) overlap/union else 0.0
    }
    /** FP32 [4 + classes, anchors]. Class-aware, multi-label NMS preserves marker roles. */
    fun decode(output: Array<FloatArray>,box: Letterbox,classes: Int,
               threshold: Double=.35,nmsThreshold: Double=.45,maxDetections: Int=64): List<Detection> {
        require(classes>0 && output.size==4+classes && output.all { it.size==output[0].size })
        require(threshold in 0.0..1.0 && nmsThreshold in 0.0..1.0 && maxDetections>0)
        val candidates=mutableListOf<Detection>()
        for(i in output[0].indices) {
            val restored=box.restore(output[0][i].toDouble(),output[1][i].toDouble(),
                output[2][i].toDouble(),output[3][i].toDouble()) ?: continue
            for(label in 0 until classes) {
                val score=output[4+label][i].toDouble()
                if(score.isFinite() && score>=threshold && score<=1.0) candidates.add(Detection(restored,label,score))
            }
        }
        val kept=mutableListOf<Detection>()
        for(candidate in candidates.sortedByDescending { it.score }) {
            if(kept.none { it.label==candidate.label && iou(it.box,candidate.box)>nmsThreshold }) {
                kept.add(candidate)
                if(kept.size==maxDetections) break
            }
        }
        return kept
    }
}
