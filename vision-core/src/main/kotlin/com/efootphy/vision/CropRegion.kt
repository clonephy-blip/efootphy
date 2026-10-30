package com.efootphy.vision

/** One immutable snapshot prevents mixing ROI edges while the UI updates capture settings. */
data class CropRegion(val left: Float,val top: Float,val right: Float,val bottom: Float) {
    init {
        require(listOf(left,top,right,bottom).all { it.isFinite() && it in 0f..1f })
        require(right-left>=0.01999f && bottom-top>=0.01999f)
    }
    val width get()=right-left
    val height get()=bottom-top
    fun moveX(value: Float): CropRegion {
        val x=value.coerceIn(0f,1f-width)
        return copy(left=x,right=x+width)
    }
    fun moveY(value: Float): CropRegion {
        val y=value.coerceIn(0f,1f-height)
        return copy(top=y,bottom=y+height)
    }
    fun resizeWidth(value: Float)=copy(right=left+value.coerceAtLeast(.02f).coerceAtMost(1f-left))
    fun resizeHeight(value: Float)=copy(bottom=top+value.coerceAtLeast(.02f).coerceAtMost(1f-top))

    companion object {
        val DEFAULT=CropRegion(.44188f,.80000f,.55898f,.95185f)
        fun restore(left: Float,top: Float,right: Float,bottom: Float): CropRegion {
            if(!listOf(left,top,right,bottom).all { it.isFinite() }) return DEFAULT
            val x=left.coerceIn(0f,.98f);val y=top.coerceIn(0f,.98f)
            return CropRegion(x,y,right.coerceIn(x+.02f,1f),bottom.coerceIn(y+.02f,1f))
        }
    }
}
