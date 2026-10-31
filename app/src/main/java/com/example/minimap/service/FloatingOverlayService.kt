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
import android.graphics.DashPathEffect
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
import com.example.minimap.processor.AutoFormationAnalysis
import com.example.minimap.processor.TacticalAnalyzer
import com.example.minimap.processor.TacticalResult
import com.example.minimap.ui.CaptureConsentActivity
import com.example.minimap.ui.MainActivity
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

class FloatingOverlayService : Service() {
    companion object {
        const val ACTION_SHOW = "com.example.minimap.SHOW"
        const val ACTION_CAPTURE = "com.example.minimap.CAPTURE"
        const val ACTION_STOP = "com.example.minimap.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        private const val CHANNEL = "minimap_assistant"
        private const val NOTIFICATION_ID = 1001

        @Volatile var roiLeftRatio = 0.436f
        @Volatile var roiTopRatio = 0.795f
        @Volatile var roiRightRatio = 0.564f
        @Volatile var roiBottomRatio = 0.948f
    }

    private lateinit var windows: WindowManager
    private lateinit var capture: ScreenCaptureManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val analyzer = TacticalAnalyzer()
    private val latestFrameResult = AtomicReference<TacticalResult?>(null)

    private var controlPod: LinearLayout? = null
    private var mainToggleBtn: TextView? = null
    private var scanFormationBtn: TextView? = null
    private var flipDirBtn: TextView? = null
    private var drawingLayer: RealTimeArHudView? = null

    private var captureActive = false
    private var destroyed = false
    private var lastDirState: Boolean? = null

    override fun onCreate() {
        super.onCreate()
        windows = requireNotNull(getSystemService(WindowManager::class.java))
        requireNotNull(getSystemService(NotificationManager::class.java)).createNotificationChannel(
            NotificationChannel(CHANNEL, "eFootball RealTime 60FPS", NotificationManager.IMPORTANCE_LOW)
        )
        capture = ScreenCaptureManager(
            context = this,
            roiProvider = { width, height ->
                Rect(
                    (width * roiLeftRatio).toInt(),
                    (height * roiTopRatio).toInt(),
                    (width * roiRightRatio).toInt(),
                    (height * roiBottomRatio).toInt()
                )
            },
            onFrame = { frame ->
                val result = analyzer.analyze(frame)
                latestFrameResult.set(result)
                drawingLayer?.postInvalidateOnAnimation()

                if (result.attackLeftToRight != lastDirState) {
                    lastDirState = result.attackLeftToRight
                    mainHandler.post {
                        if (!destroyed && captureActive) updatePodLabels()
                    }
                }
            },
            onState = { running ->
                if (!destroyed) {
                    captureActive = running
                    if (!running) latestFrameResult.set(null)
                    updatePodLabels()
                    drawingLayer?.visibility = if (running) View.VISIBLE else View.GONE
                    updateForeground(running)
                }
            },
            onError = { error ->
                Log.e("Minimap", "Capture error", error)
                Toast.makeText(this, "Chạm nút BẬT 60FPS để cấp lại quyền.", Toast.LENGTH_SHORT).show()
            },
            scanFps = 60
        )
    }

    private fun updatePodLabels() {
        if (!captureActive) {
            mainToggleBtn?.text = "BẬT 60FPS"
            scanFormationBtn?.visibility = View.GONE
            flipDirBtn?.visibility = View.GONE
            return
        }
        mainToggleBtn?.text = "⏹ TẮT"
        scanFormationBtn?.visibility = View.VISIBLE
        flipDirBtn?.visibility = View.VISIBLE

        val arrow = if (analyzer.attackLeftToRight) "→" else "←"
        val mode = if (analyzer.manualDirectionLock) "CÔNG" else "AUTO"
        flipDirBtn?.text = "⇄ $mode $arrow"
        scanFormationBtn?.text = if (analyzer.isFormationScanActive()) "✖ ĐÓNG SƠ ĐỒ" else "🔍 SƠ ĐỒ"
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
            ).apply { alpha = 0.78f }
        )

        val pod = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = GradientDrawable().apply {
                cornerRadius = dp(10).toFloat()
                setColor(Color.argb(215, 15, 23, 42))
                setStroke(dp(1), Color.rgb(100, 116, 139))
            }
            setPadding(dp(5), dp(5), dp(5), dp(5))
        }

        val btnMain = TextView(this).apply {
            text = "BẬT 60FPS"
            gravity = Gravity.CENTER
            textSize = 10f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setPadding(dp(8), dp(5), dp(8), dp(5))
            background = GradientDrawable().apply {
                cornerRadius = dp(7).toFloat()
                setColor(Color.rgb(16, 185, 129))
            }
            setOnClickListener {
                if (captureActive) {
                    analyzer.clearFormationScan()
                    capture.stop()
                } else {
                    startActivity(
                        Intent(this@FloatingOverlayService, CaptureConsentActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            }
        }
        mainToggleBtn = btnMain

        val btnScan = TextView(this).apply {
            text = "🔍 SƠ ĐỒ"
            gravity = Gravity.CENTER
            textSize = 10f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.BLACK)
            visibility = View.GONE
            setPadding(dp(8), dp(5), dp(8), dp(5))
            background = GradientDrawable().apply {
                cornerRadius = dp(7).toFloat()
                setColor(Color.rgb(250, 204, 21))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(4) }
            setOnClickListener {
                if (analyzer.isFormationScanActive()) {
                    analyzer.clearFormationScan()
                } else {
                    analyzer.triggerManualFormationScan()
                }
                updatePodLabels()
                drawingLayer?.postInvalidateOnAnimation()
            }
        }
        scanFormationBtn = btnScan

        val btnDir = TextView(this).apply {
            text = "⇄ AUTO →"
            gravity = Gravity.CENTER
            textSize = 10f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.WHITE)
            visibility = View.GONE
            setPadding(dp(8), dp(5), dp(8), dp(5))
            background = GradientDrawable().apply {
                cornerRadius = dp(7).toFloat()
                setColor(Color.rgb(51, 65, 85))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(4) }
            setOnClickListener {
                analyzer.toggleManualDirection()
                updatePodLabels()
                drawingLayer?.postInvalidateOnAnimation()
            }
        }
        flipDirBtn = btnDir

        pod.addView(btnMain)
        pod.addView(btnScan)
        pod.addView(btnDir)
        controlPod = pod

        val lp = params(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
        ).apply { x = dp(12); y = dp(90) }

        val slop = ViewConfiguration.get(this).scaledTouchSlop
        var downX = 0f
        var downY = 0f
        var originX = 0
        var originY = 0
        var dragging = false

        btnMain.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX
                    downY = event.rawY
                    originX = lp.x
                    originY = lp.y
                    dragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (abs(dx) > slop || abs(dy) > slop) dragging = true
                    if (dragging) {
                        lp.x = originX + dx.roundToInt()
                        lp.y = originY + dy.roundToInt()
                        clampPod(lp)
                        windows.updateViewLayout(pod, lp)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!dragging) view.performClick()
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
        lp.x = lp.x.coerceIn(0, (size.x - dp(80)).coerceAtLeast(0))
        lp.y = lp.y.coerceIn(0, (size.y - dp(110)).coerceAtLeast(0))
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
            .setContentTitle("eFootball RealTime 60FPS Vision")
            .setContentText(if (projecting) "Đang đồng bộ 60/60 FPS" else "Sẵn sàng")
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
        capture.close()
        listOfNotNull(controlPod, drawingLayer).forEach {
            if (it.isAttachedToWindow) windows.removeView(it)
        }
        controlPod = null
        drawingLayer = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private inner class RealTimeArHudView(context: Context) : View(context) {
        private val roiRect = RectF()
        private val cardRect = RectF()

        private val footRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.rgb(255, 195, 0)
        }
        private val primaryPassPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.rgb(0, 255, 110)
        }
        private val secondaryPassPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.rgb(255, 215, 0)
            pathEffect = DashPathEffect(floatArrayOf(12f, 8f), 0f)
        }
        private val landingFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(60, 0, 255, 110)
        }
        private val weakZonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.rgb(255, 60, 60)
        }
        private val minimapBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(1).toFloat()
            color = Color.argb(120, 255, 215, 0)
        }
        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = dp(12).toFloat()
            color = Color.WHITE
            typeface = Typeface.DEFAULT_BOLD
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }
        private val panelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(225, 15, 23, 42)
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return

            val res = latestFrameResult.get() ?: return

            res.autoFormation?.let { report ->
                drawFormationReport(canvas, w, h, report)
                return
            }

            roiRect.set(w * roiLeftRatio, h * roiTopRatio, w * roiRightRatio, h * roiBottomRatio)
            canvas.drawRect(roiRect, minimapBorderPaint)
            val offX = roiRect.left + res.offsideLineX * roiRect.width()
            canvas.drawLine(offX, roiRect.top - dp(6), offX, roiRect.top, secondaryPassPaint)
            canvas.drawLine(offX, roiRect.bottom, offX, roiRect.bottom + dp(6), secondaryPassPaint)

            val foot = res.controlledFoot3D ?: return
            val fx = foot.x * w
            val fy = foot.y * h

            canvas.save()
            canvas.scale(1f, 0.52f, fx, fy)
            canvas.drawCircle(fx, fy, dp(20).toFloat(), footRingPaint)
            canvas.restore()

            res.pitch3DOptions.forEachIndexed { idx, opt ->
                val sx = opt.from3DX * w
                val sy = opt.from3DY * h
                val ex = opt.to3DX * w
                val ey = opt.to3DY * h
                val paint = if (idx == 0) primaryPassPaint else secondaryPassPaint

                if (opt.isOffScreen) {
                    val angle = atan2((ey - sy).toDouble(), (ex - sx).toDouble())
                    val pointerLen = dp(58).toFloat()
                    val px = sx + (pointerLen * cos(angle)).toFloat()
                    val py = sy + (pointerLen * sin(angle)).toFloat()
                    drawArrow(canvas, sx, sy, px, py, paint)
                    textPaint.color = Color.rgb(250, 204, 21)
                    canvas.drawText("🚀 PHẤT DÀI (${opt.openSpacePct}%)", px - dp(35), py - dp(8), textPaint)
                } else {
                    drawArrow(canvas, sx, sy, ex, ey, paint)
                    canvas.save()
                    canvas.scale(1f, 0.52f, ex, ey)
                    if (idx == 0) canvas.drawCircle(ex, ey, dp(22).toFloat(), landingFillPaint)
                    canvas.drawCircle(ex, ey, dp(22).toFloat(), paint)
                    canvas.restore()

                    val tag = if (opt.isThroughRun) "⚡ CHỌC KHE (${opt.openSpacePct}%)" else "✔ CHUYỀN (${opt.openSpacePct}%)"
                    textPaint.color = if (idx == 0) Color.rgb(167, 243, 208) else Color.YELLOW
                    canvas.drawText(tag, (ex - dp(35)).coerceIn(dp(10).toFloat(), w - dp(130)), (ey - dp(14)).coerceAtLeast(dp(28).toFloat()), textPaint)
                }
            }
        }

        private fun drawFormationReport(canvas: Canvas, w: Float, h: Float, report: AutoFormationAnalysis) {
            for (card in report.detectedPlayerCards) {
                canvas.drawCircle(card.x * w, card.y * h, dp(14).toFloat(), footRingPaint)
            }
            for (weak in report.weakZoneSpots) {
                val wx = weak.x * w
                val wy = weak.y * h
                canvas.drawCircle(wx, wy, dp(38).toFloat(), weakZonePaint)
                textPaint.color = Color.YELLOW
                canvas.drawText("🎯 LỖ HỔNG", wx - dp(36), wy + dp(5), textPaint)
            }

            cardRect.set(w * 0.16f, dp(10).toFloat(), w * 0.86f, dp(104).toFloat())
            canvas.drawRoundRect(cardRect, dp(12).toFloat(), dp(12).toFloat(), panelBgPaint)
            canvas.drawRoundRect(cardRect, dp(12).toFloat(), dp(12).toFloat(), primaryPassPaint)

            var ty = cardRect.top + dp(24)
            val lx = cardRect.left + dp(14)

            textPaint.color = Color.rgb(56, 189, 248)
            canvas.drawText("🤖 SƠ ĐỒ ĐỐI THỦ: ${report.detectedFormation}", lx, ty, textPaint)
            ty += dp(22)
            textPaint.color = Color.rgb(250, 204, 21)
            canvas.drawText("🔥 TỬ HUYỆT: ${report.tacticalWeakness}", lx, ty, textPaint)
            ty += dp(22)
            textPaint.color = Color.WHITE
            canvas.drawText("💡 ${report.actionableAdvice}", lx, ty, textPaint)
        }

        private fun drawArrow(canvas: Canvas, sx: Float, sy: Float, ex: Float, ey: Float, paint: Paint) {
            canvas.drawLine(sx, sy, ex, ey, paint)
            val angle = atan2((ey - sy).toDouble(), (ex - sx).toDouble())
            val head = dp(10).toFloat()
            val a1 = angle - Math.PI / 6
            val a2 = angle + Math.PI / 6
            canvas.drawLine(ex, ey, (ex - head * cos(a1)).toFloat(), (ey - head * sin(a1)).toFloat(), paint)
            canvas.drawLine(ex, ey, (ex - head * cos(a2)).toFloat(), (ey - head * sin(a2)).toFloat(), paint)
        }
    }
}
