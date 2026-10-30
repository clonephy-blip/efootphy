package com.example.minimap.service

import android.annotation.SuppressLint
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.content.res.Configuration
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.efootphy.vision.CropRegion
import com.efootphy.vision.Timing
import com.efootphy.vision.TrackStatus
import android.os.SystemClock
import com.example.minimap.processor.ScreenMode
import com.example.minimap.processor.TacticalAnalyzer
import com.example.minimap.processor.TacticalResult
import com.example.minimap.ui.CaptureConsentActivity
import com.example.minimap.BuildConfig
import com.example.minimap.ui.MainActivity
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.roundToInt

class FloatingOverlayService : Service() {
    companion object {
        const val ACTION_SHOW = "com.example.minimap.SHOW"
        const val ACTION_CAPTURE = "com.example.minimap.CAPTURE"
        const val ACTION_STOP = "com.example.minimap.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        private const val CHANNEL = "minimap_assistant"
        private const val NOTIFICATION_ID = 1001

        @Volatile var roi = CropRegion.DEFAULT

        fun loadRoi(context: Context) {
            val saved=context.getSharedPreferences("roi",Context.MODE_PRIVATE)
            roi=CropRegion.restore(saved.getFloat("left",roi.left),saved.getFloat("top",roi.top),
                saved.getFloat("right",roi.right),saved.getFloat("bottom",roi.bottom))
        }
    }

    private lateinit var windows: WindowManager
    private lateinit var capture: ScreenCaptureManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var analyzer: TacticalAnalyzer

    private val latestFrameResult = AtomicReference<TacticalResult?>(null)

    private var controlPod: LinearLayout? = null
    private var mainToggleBtn: TextView? = null
    private var drawingLayer: RealTimeArHudView? = null

    private var captureActive = false
    private var destroyed = false
    private var lastDisplayedDirection: Boolean? = null
    private var lastScreenMode: Int = ScreenMode.MODE_LIVE_MATCH

    override fun onCreate() {
        super.onCreate()
        analyzer = TacticalAnalyzer(this)
        loadRoi(this)
        windows = requireNotNull(getSystemService(WindowManager::class.java))
        requireNotNull(getSystemService(NotificationManager::class.java)).createNotificationChannel(
            NotificationChannel(CHANNEL, "eFootball Assistant", NotificationManager.IMPORTANCE_LOW)
        )
        capture = ScreenCaptureManager(
            context = this,
            roiProvider = { width, height ->
                val region=roi
                Rect(
                    (width * region.left).toInt(),
                    (height * region.top).toInt(),
                    (width * region.right).toInt(),
                    (height * region.bottom).toInt()
                )
            },
            onFrame = { frame ->
                val result = analyzer.analyze(frame)
                latestFrameResult.set(result)
                drawingLayer?.postInvalidateOnAnimation()

                if (result.attackLeftToRight != lastDisplayedDirection || result.screenMode != lastScreenMode) {
                    lastDisplayedDirection = result.attackLeftToRight
                    lastScreenMode = result.screenMode
                    mainHandler.post {
                        if (!destroyed && captureActive) {
                            updatePodLabels()
                        }
                    }
                }
            },
            onState = { running ->
                if (!destroyed) {
                    captureActive = running
                    if (!running) {
                        latestFrameResult.set(null)
                    }
                    updatePodLabels()
                    drawingLayer?.visibility = if (running) View.VISIBLE else View.GONE
                    updateForeground(running)
                }
            },
            onError = { error ->
                Log.e("Minimap", "Capture error", error)
                Toast.makeText(this, "Chạm BẬT QUÉT để cấp lại quyền màn hình.", Toast.LENGTH_SHORT).show()
            },
            scanFps = 15,
            onSessionReset = { analyzer.reset() }
        )
    }

    private fun updatePodLabels() {
        if (!captureActive) {
            mainToggleBtn?.text = "BẬT QUÉT"
            return
        }
        val arrow = if (analyzer.attackLeftToRight) "→" else "←"
        val modePrefix = if (analyzer.manualDirectionLock) "HƯỚNG" else "QUÉT"
        mainToggleBtn?.text = "$modePrefix $arrow"

    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP || !Settings.canDrawOverlays(this)) {
            stopSelf()
            return START_NOT_STICKY
        }
        val data = if (Build.VERSION.SDK_INT >= 33) {
            intent?.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent?.getParcelableExtra<Intent>(EXTRA_RESULT_DATA)
        }
        val code = intent?.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val startCapture = intent?.action == ACTION_CAPTURE &&
            code == Activity.RESULT_OK && data != null && !captureActive
        try {
            updateForeground(captureActive || startCapture)
            if (controlPod == null) showWindows()
            if (startCapture) {
                captureActive = true
                mainToggleBtn?.text = "…"
                capture.start(Activity.RESULT_OK, requireNotNull(data))
            }
        } catch (error: Exception) {
            Log.e("Minimap", "Cannot start overlay", error)
            stopSelf()
        }
        return START_NOT_STICKY
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun showWindows() {
        val hud = RealTimeArHudView(this).apply {
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
            visibility = View.GONE
        }
        drawingLayer = hud
        windows.addView(
            hud,
            params(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
            ).apply {
                alpha = 0.78f
            }
        )

        val pod = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = GradientDrawable().apply {
                cornerRadius = dp(12).toFloat()
                setColor(Color.argb(210, 15, 23, 42))
                setStroke(dp(1), Color.rgb(56, 189, 248))
            }
            setPadding(dp(5), dp(5), dp(5), dp(5))
        }

        val btnMain = TextView(this).apply {
            text = "BẬT QUÉT"
            gravity = Gravity.CENTER
            textSize = 11f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setPadding(dp(8), dp(6), dp(8), dp(6))
            background = GradientDrawable().apply {
                cornerRadius = dp(8).toFloat()
                setColor(Color.rgb(16, 185, 129))
            }
            setOnClickListener {
                if (captureActive) {
                    capture.stop()
                } else {
                    startActivity(
                        Intent(this@FloatingOverlayService, CaptureConsentActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            }
            setOnLongClickListener {
                analyzer.toggleManualDirection()
                updatePodLabels()
                val dirText = if (analyzer.attackLeftToRight) "Trái sang Phải (→)" else "Phải sang Trái (←)"
                Toast.makeText(this@FloatingOverlayService, "Đã khóa hướng thủ công: $dirText", Toast.LENGTH_SHORT).show()
                true
            }
        }
        mainToggleBtn = btnMain

        pod.addView(btnMain)
        controlPod = pod

        val lp = params(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
        ).apply { x = dp(14); y = dp(85) }

        val slop = ViewConfiguration.get(this).scaledTouchSlop
        var downX = 0f
        var downY = 0f
        var originX = 0
        var originY = 0
        var dragging = false
        var longTriggered = false
        val longRunnable = Runnable {
            if (!dragging) {
                longTriggered = true
                btnMain.performLongClick()
            }
        }

        btnMain.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY
                    originX = lp.x; originY = lp.y
                    dragging = false; longTriggered = false
                    mainHandler.postDelayed(longRunnable, ViewConfiguration.getLongPressTimeout().toLong())
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (abs(dx) > slop || abs(dy) > slop) {
                        dragging = true
                        mainHandler.removeCallbacks(longRunnable)
                    }
                    if (dragging) {
                        lp.x = originX + dx.roundToInt()
                        lp.y = originY + dy.roundToInt()
                        clampPod(lp)
                        windows.updateViewLayout(pod, lp)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    mainHandler.removeCallbacks(longRunnable)
                    if (!dragging && !longTriggered) view.performClick()
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    mainHandler.removeCallbacks(longRunnable)
                    true
                }
                else -> false
            }
        }
        windows.addView(pod, lp)
    }

    private fun params(width: Int, height: Int, flags: Int) = WindowManager.LayoutParams(
        width, height, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        flags, PixelFormat.TRANSLUCENT
    ).apply { gravity = Gravity.TOP or Gravity.LEFT }

    @Suppress("DEPRECATION")
    private fun clampPod(lp: WindowManager.LayoutParams) {
        val size = android.graphics.Point()
        if (Build.VERSION.SDK_INT >= 30) {
            val bounds = windows.maximumWindowMetrics.bounds
            size.set(bounds.width(), bounds.height())
        } else windows.defaultDisplay.getSize(size)
        lp.x = lp.x.coerceIn(0, (size.x - dp(75)).coerceAtLeast(0))
        lp.y = lp.y.coerceIn(0, (size.y - dp(85)).coerceAtLeast(0))
        val podWidth = controlPod?.width?.takeIf { it > 0 } ?: dp(96)
        val podHeight = controlPod?.height?.takeIf { it > 0 } ?: dp(85)
        val region=roi
        val sourceRoi = Rect((size.x * region.left).toInt(), (size.y * region.top).toInt(),
            (size.x * region.right).toInt(), (size.y * region.bottom).toInt())
        if (Rect.intersects(Rect(lp.x, lp.y, lp.x + podWidth, lp.y + podHeight), sourceRoi)) {
            lp.y = (sourceRoi.top - podHeight - dp(8)).coerceAtLeast(0)
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        controlPod?.let { view ->
            val lp = view.layoutParams as WindowManager.LayoutParams
            clampPod(lp)
            windows.updateViewLayout(view, lp)
        }
    }

    private fun updateForeground(projecting: Boolean) {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stop = PendingIntent.getService(
            this, 1,
            Intent(this, FloatingOverlayService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("eFootball Assistant · nghiên cứu")
            .setContentText(if (projecting) "Đang đọc radar · tối đa 15 lần/giây" else "Sẵn sàng")
            .setContentIntent(open).setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Dừng", stop).build())
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            val idleType = if (Build.VERSION.SDK_INT >= 34)
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0
            val type = idleType or if (projecting)
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION else 0
            startForeground(NOTIFICATION_ID, notification, type)
        } else startForeground(NOTIFICATION_ID, notification)
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).roundToInt()
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        destroyed = true
        mainHandler.removeCallbacksAndMessages(null)
        capture.close { analyzer.close() }
        listOfNotNull(controlPod, drawingLayer).forEach {
            if (it.isAttachedToWindow) windows.removeView(it)
        }
        controlPod = null
        drawingLayer = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private inner class RealTimeArHudView(context: Context) : View(context) {
        private val paint=Paint(Paint.ANTI_ALIAS_FLAG)
        private val textPaint=Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize=dp(11).toFloat();color=Color.WHITE;typeface=Typeface.DEFAULT_BOLD
        }
        private val panel=RectF()
        private val field=RectF()
        private val source=RectF()
        private val expire=Runnable { invalidate() }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            removeCallbacks(expire)
            val result=latestFrameResult.get() ?: return
            val now=SystemClock.elapsedRealtimeNanos()
            val lifetime=250_000_000L
            if(!Timing.fresh(now,result.timestampNs,lifetime)) return
            postDelayed(expire,((result.timestampNs+lifetime-now)/1_000_000L).coerceAtLeast(1L))
            val w=width.toFloat();val h=height.toFloat()
            // Draw the calibration outline outside the source crop to avoid capture feedback.
            val region=roi
            source.set(w*region.left,h*region.top,w*region.right,h*region.bottom)
            paint.style=Paint.Style.STROKE;paint.strokeWidth=dp(1).toFloat();paint.color=Color.CYAN
            val gap=dp(3).toFloat()
            canvas.drawRect(source.left-gap,source.top-gap,source.right+gap,source.bottom+gap,paint)
            // All detections are shown in a separate helper radar, never on the source minimap.
            val pw=minOf(dp(190).toFloat(),w*.28f)
            val ph=pw*.60f
            panel.set(dp(10).toFloat(),dp(48).toFloat(),dp(10)+pw,dp(48)+ph)
            paint.style=Paint.Style.FILL;paint.color=Color.rgb(15,23,42)
            canvas.drawRoundRect(panel,dp(8).toFloat(),dp(8).toFloat(),paint)
            paint.style=Paint.Style.STROKE;paint.color=Color.rgb(80,120,100);paint.strokeWidth=dp(1).toFloat()
            field.set(panel.left+dp(7),panel.top+dp(22),panel.right-dp(7),panel.bottom-dp(7))
            canvas.drawRect(field,paint)
            canvas.drawLine(field.centerX(),field.top,field.centerX(),field.bottom,paint)
            canvas.drawText(result.status,panel.left+dp(7),panel.top+dp(14),textPaint)
            for(player in result.world?.players.orEmpty()) {
                if(!player.point.inPitch()) continue
                val x=field.left+player.point.x.toFloat()*field.width()
                val y=field.top+player.point.y.toFloat()*field.height()
                paint.style=Paint.Style.FILL
                paint.color=if(player.status==TrackStatus.COASTING) Color.GRAY else Color.WHITE
                canvas.drawCircle(x,y,dp(2).toFloat(),paint)
                if(BuildConfig.DEBUG) {
                    textPaint.textSize=dp(7).toFloat()
                    canvas.drawText(player.id.toString(),x+dp(2),y-dp(2),textPaint)
                    textPaint.textSize=dp(11).toFloat()
                }
            }
            for(marker in result.detections) {
                if(marker.label!=0 && marker.label!=1) continue
                val x=field.left+((marker.box.left+marker.box.right)/2).toFloat()*field.width()
                val y=field.top+((marker.box.top+marker.box.bottom)/2).toFloat()*field.height()
                paint.style=Paint.Style.STROKE
                paint.color=if(marker.label==0) Color.YELLOW else Color.CYAN
                canvas.drawCircle(x,y,dp(4).toFloat(),paint)
            }
        }

        override fun onDetachedFromWindow() {
            removeCallbacks(expire)
            super.onDetachedFromWindow()
        }
    }
}
