package com.example.minimap.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.efootphy.vision.Detection
import com.efootphy.vision.DetectorPostprocess
import com.efootphy.vision.Letterbox
import com.example.minimap.processor.RgbaFrame
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Worker-thread confined. Reuses image/tensor storage; outputs are closed after every run. */
class OnnxMinimapDetector(model: LocalModel): AutoCloseable {
    private val session=ModelRepository.createSession(model.model).also { opened ->
        try { ModelRepository.validateShape(opened) }
        catch(error: Exception) { opened.close();throw error }
    }
    private val inputName=session.inputNames.single()
    private val buffer=ByteBuffer.allocateDirect(3*320*320*4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val tensor=OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),buffer,longArrayOf(1,3,320,320))
    private val canvasBitmap=Bitmap.createBitmap(320,320,Bitmap.Config.ARGB_8888)
    private val canvas=Canvas(canvasBitmap)
    private val paint=Paint(Paint.FILTER_BITMAP_FLAG)
    private val target=RectF()
    private var bitmap: Bitmap?=null
    private var cropPixels=IntArray(0)
    private val pixels=IntArray(320*320)

    fun detect(frame: RgbaFrame): List<Detection> {
        require(frame.width>0 && frame.height>0 && frame.pixels.size.toLong()>=frame.width.toLong()*frame.height*4)
        val transform=Letterbox(frame.width,frame.height,320)
        if(bitmap?.width!=frame.width || bitmap?.height!=frame.height) {
            bitmap?.recycle()
            bitmap=Bitmap.createBitmap(frame.width,frame.height,Bitmap.Config.ARGB_8888)
            cropPixels=IntArray(frame.width*frame.height)
        }
        for(i in cropPixels.indices) {
            val p=i*4
            cropPixels[i]=Color.rgb(frame.pixels[p].toInt() and 255,frame.pixels[p+1].toInt() and 255,frame.pixels[p+2].toInt() and 255)
        }
        bitmap!!.setPixels(cropPixels,0,frame.width,0,0,frame.width,frame.height)
        canvas.drawColor(Color.rgb(114,114,114))
        target.set(transform.left.toFloat(),transform.top.toFloat(),
            (transform.left+transform.resizedWidth).toFloat(),(transform.top+transform.resizedHeight).toFloat())
        canvas.drawBitmap(bitmap!!,null,target,paint)
        canvasBitmap.getPixels(pixels,0,320,0,0,320,320)
        val plane=320*320
        for(i in pixels.indices) {
            buffer.put(i,Color.red(pixels[i])/255f)
            buffer.put(plane+i,Color.green(pixels[i])/255f)
            buffer.put(plane*2+i,Color.blue(pixels[i])/255f)
        }
        session.run(mapOf(inputName to tensor)).use { result ->
            @Suppress("UNCHECKED_CAST")
            val output=result[0].value as Array<Array<FloatArray>>
            require(output.size==1)
            return DetectorPostprocess.decode(output[0],transform,3)
        }
    }

    override fun close() {
        try { tensor.close() } finally {
            try { session.close() } finally { bitmap?.recycle();canvasBitmap.recycle() }
        }
    }
}
