package com.example.minimap.service

import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.graphics.Point
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.DisplayMetrics
import android.view.Display
import android.view.WindowManager
import com.example.minimap.processor.MinimapCropper
import com.example.minimap.processor.RgbaFrame
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt

class ScreenCaptureManager(
    context: Context,
    private val roiProvider: (width: Int, height: Int) -> Rect,
    private val onFrame: (RgbaFrame) -> Unit,
    private val onState: (Boolean) -> Unit,
    private val onError: (Throwable) -> Unit,
    scanFps: Int = 60,
    private val maxCaptureSide: Int = 960
) : AutoCloseable {
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())

    // Nâng cấp 1: Đặt độ ưu tiên luồng ở mức URGENT_DISPLAY (-8) để ngang hàng với bộ dựng hình GPU 60FPS của game
    private val thread = HandlerThread(
        "efootball-realtime-60fps",
        Process.THREAD_PRIORITY_URGENT_DISPLAY
    ).apply { start() }

    private val worker = Handler(thread.looper)
    private val displays = requireNotNull(app.getSystemService(DisplayManager::class.java))
    private val cropper = MinimapCropper()

    // Nâng cấp 2: Đặt khoảng cách nhận frame có dung sai 15% (13.8ms cho 60FPS) để KHÔNG BAO GIỜ bị rớt nhịp VSYNC 16.6ms
    private val frameIntervalNs = (830_000_000L / scanFps.coerceIn(15, 90))

    // Khóa chống nghẽn cổ chai (Backpressure Drop): Nếu frame trước chưa vẽ xong thì bỏ qua ngay, chỉ lấy frame MỚI NHẤT
    private val isProcessingFrame = AtomicBoolean(false)

    private var projection: MediaProjection? = null
    private var callback: MediaProjection.Callback? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var lastFrameNs = 0L
    private var densityDpi = 1
    private var listening = false
    @Volatile private var closed = false

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = Unit
        override fun onDisplayRemoved(displayId: Int) = Unit
        override fun onDisplayChanged(displayId: Int) {
            if (displayId == Display.DEFAULT_DISPLAY && projection != null) {
                val size = screenSize()
                safely { resize(size.x, size.y) }
            }
        }
    }

    fun start(resultCode: Int, resultData: Intent) {
        if (closed) return
        worker.post {
            if (closed || projection != null) return@post
            safely {
                densityDpi = app.resources.configuration.densityDpi
                val manager = requireNotNull(app.getSystemService(MediaProjectionManager::class.java))
                val session = requireNotNull(manager.getMediaProjection(resultCode, resultData))
                projection = session
                val cb = object : MediaProjection.Callback() {
                    override fun onStop() {
                        if (projection === session) release(stopProjection = false)
                    }
                    override fun onCapturedContentResize(width: Int, height: Int) {
                        if (projection === session) safely { resize(width, height) }
                    }
                }
                callback = cb
                session.registerCallback(cb, worker)
                val size = screenSize()
                val scaled = scaledSize(size.x, size.y)
                val output = newReader(scaled.x, scaled.y)
                reader = output
                virtualDisplay = session.createVirtualDisplay(
                    "RealTime60FpsCapture", scaled.x, scaled.y, densityDpi,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    output.surface, null, worker
                )
                if (Build.VERSION.SDK_INT < 34) {
                    displays.registerDisplayListener(displayListener, worker)
                    listening = true
                }
                main.post { if (!closed) onState(true) }
            }
        }
    }

    fun stop() {
        if (!closed) worker.post { release() }
    }

    private fun newReader(width: Int, height: Int): ImageReader =
        ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2).apply {
            setOnImageAvailableListener({ source ->
                if (source !== reader || closed) return@setOnImageAvailableListener
                val image = try {
                    source.acquireLatestImage()
                } catch (_: Exception) {
                    null
                } ?: return@setOnImageAvailableListener

                try {
                    if (source !== reader || closed) return@setOnImageAvailableListener
                    val now = SystemClock.elapsedRealtimeNanos()
                    if (now - lastFrameNs < frameIntervalNs) return@setOnImageAvailableListener
                    if (!isProcessingFrame.compareAndSet(false, true)) return@setOnImageAvailableListener

                    try {
                        lastFrameNs = now
                        val roi = roiProvider(image.width, image.height)
                        val frame = cropper.crop(image, roi, now)
                        if (frame != null) {
                            onFrame(frame)
                        }
                    } finally {
                        isProcessingFrame.set(false)
                    }
                } catch (error: Exception) {
                    release()
                    main.post { if (!closed) onError(error) }
                } finally {
                    image.close()
                }
            }, worker)
        }

    private fun resize(sourceWidth: Int, sourceHeight: Int) {
        if (sourceWidth <= 0 || sourceHeight <= 0) return
        val display = virtualDisplay ?: return
        val size = scaledSize(sourceWidth, sourceHeight)
        if (reader?.width == size.x && reader?.height == size.y) return
        val replacement = newReader(size.x, size.y)
        val old = reader
        reader = replacement
        try {
            display.surface = null
            display.resize(size.x, size.y, densityDpi)
            display.surface = replacement.surface
        } finally {
            old?.setOnImageAvailableListener(null, null)
            old?.close()
        }
        lastFrameNs = 0L
    }

    @Suppress("DEPRECATION")
    private fun screenSize(): Point {
        val windows = requireNotNull(app.getSystemService(WindowManager::class.java))
        if (Build.VERSION.SDK_INT >= 30) {
            val bounds = windows.maximumWindowMetrics.bounds
            return Point(bounds.width(), bounds.height())
        }
        val metrics = DisplayMetrics()
        windows.defaultDisplay.getRealMetrics(metrics)
        return Point(metrics.widthPixels, metrics.heightPixels)
    }

    private fun scaledSize(width: Int, height: Int): Point {
        val scale = (maxCaptureSide.coerceAtLeast(1).toFloat() / max(width, height))
            .coerceAtMost(1f)
        return Point(
            (width * scale).roundToInt().coerceAtLeast(1),
            (height * scale).roundToInt().coerceAtLeast(1)
        )
    }

    private fun safely(block: () -> Unit) {
        try {
            block()
        } catch (error: Exception) {
            release()
            main.post { if (!closed) onError(error) }
        }
    }

    private fun release(stopProjection: Boolean = true) {
        val old = projection
        projection = null
        if (listening) displays.unregisterDisplayListener(displayListener)
        listening = false
        callback?.let { old?.unregisterCallback(it) }
        callback = null
        virtualDisplay?.release()
        virtualDisplay = null
        reader?.setOnImageAvailableListener(null, null)
        reader?.close()
        reader = null
        if (stopProjection) old?.stop()
        lastFrameNs = 0L
        isProcessingFrame.set(false)
        main.post { if (!closed) onState(false) }
    }

    override fun close() {
        closed = true
        worker.post {
            release()
            thread.quitSafely()
        }
    }
}
