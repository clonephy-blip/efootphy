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
import com.example.minimap.processor.ScreenMode
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

        // Tọa độ chuẩn xác 100% ôm khít khung Minimap eFootball trên màn hình ngang
        @Volatile var roiLeftRatio = 0.436f
        @Volatile var roiTopRatio = 0.795f
        @Volatile var roiRightRatio = 0.564f
        @Volatile var roiBottomRatio = 0.948f
    }

    private lateinit var windows: WindowManager
    private lateinit var capture: ScreenCaptureManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val analyzer = TacticalAnalyzer()

    // Bộ đệm nguyên tử Lock-Free: Luồng 60FPS cập nhật trực tiếp, không gây nghẽn hàng đợi giao diện
    private val latestFrameResult = AtomicReference<TacticalResult?>(null)

    private var controlPod: LinearLayout? = null
    private var mainToggleBtn: TextView? = null
    private var subActionBtn: TextView? = null
    private var drawingLayer: RealTimeArHudView? = null

    private var captureActive = false
    private var destroyed = false
    private var lastDisplayedDirection: Boolean? = null
    private var lastScreenMode: Int = ScreenMode.MODE_LIVE_MATCH
    @Volatile private var hideFormationOverlayManually = false

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
                // Kích hoạt vẽ ngay ở xung nhịp VSYNC 60Hz tiếp theo (Thread-safe, không qua Handler queue)
                drawingLayer?.postInvalidateOnAnimation()

                // Chỉ cập nhật chữ trên nút nổi khi chiều tấn công hoặc chế độ màn hình thực sự thay đổi
                if (result.attackLeftToRight != lastDisplayedDirection || result.screenMode != lastScreenMode) {
                    lastDisplayedDirection = result.attackLeftToRight
                    lastScreenMode = result.screenMode
                    if (result.screenMode == ScreenMode.MODE_LIVE_MATCH) {
                        hideFormationOverlayManually = false
                    }
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
                Toast.makeText(this, "Chạm nút AR để cấp lại quyền màn hình.", Toast.LENGTH_SHORT).show()
            },
            scanFps = 60
        )
    }

    private fun updatePodLabels() {
        if (!captureActive) {
            mainToggleBtn?.text = "BẬT 60FPS"
            subActionBtn?.visibility = View.GONE
            return
        }
        val arrow = if (analyzer.attackLeftToRight) "→" else "←"
        val modePrefix = if (analyzer.manualDirectionLock) "KHÓA" else "AUTO"
        mainToggleBtn?.text = "$modePrefix $arrow"

        if (lastScreenMode == ScreenMode.MODE_GAMEPLAN_FORMATION) {
            subActionBtn?.visibility = View.VISIBLE
            subActionBtn?.text = if (hideFormationOverlayManually) "👁 HIỆN SƠ ĐỒ" else "✖ ẨN SƠ ĐỒ"
        } else {
            subActionBtn?.visibility = View.GONE
        }
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
        // Độ mờ 0.78f đảm bảo chuẩn chạm xuyên thấu 100% của Android 12/13/14/15
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
            text = "BẬT 60FPS"
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

        val btnSub = TextView(this).apply {
            text = "✖ ẨN SƠ ĐỒ"
            gravity = Gravity.CENTER
            textSize = 10f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.BLACK)
            visibility = View.GONE
            setPadding(dp(6), dp(5), dp(6), dp(5))
            background = GradientDrawable().apply {
                cornerRadius = dp(8).toFloat()
                setColor(Color.rgb(250, 204, 21))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(4) }
            setOnClickListener {
                hideFormationOverlayManually = !hideFormationOverlayManually
                updatePodLabels()
                drawingLayer?.postInvalidateOnAnimation()
            }
        }
        subActionBtn = btnSub

        pod.addView(btnMain)
        pod.addView(btnSub)
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

    // ==================== LỚP VẼ THỰC TẾ TĂNG CƯỜNG 60FPS TRÊN MẶT CỎ SÂN 3D ====================
    private inner class RealTimeArHudView(context: Context) : View(context) {
        private val roiRect = RectF()
        private val cardRect = RectF()

        private val footRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.CYAN
        }
        private val throughLaserPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.rgb(0, 255, 128)
        }
        private val safePassPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.rgb(250, 204, 21)
            pathEffect = DashPathEffect(floatArrayOf(14f, 8f), 0f)
        }
        private val landingZoneFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(70, 0, 255, 128)
        }
        private val dangerRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.rgb(255, 55, 55)
        }
        private val minimapFramePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(1).toFloat()
            color = Color.argb(130, 56, 189, 248)
        }
        private val offsideLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.YELLOW
            pathEffect = DashPathEffect(floatArrayOf(8f, 6f), 0f)
        }
        private val tagTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = dp(12).toFloat()
            color = Color.WHITE
            typeface = Typeface.DEFAULT_BOLD
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }
        private val modalBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(225, 15, 23, 42)
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return

            val res = latestFrameResult.get() ?: return

            // 1. NẾU ĐANG MỞ MÀN HÌNH SƠ ĐỒ ĐỘI HÌNH (GAME PLAN) -> TỰ ĐỘNG HIỂN THỊ MẢNG 1
            if (res.screenMode == ScreenMode.MODE_GAMEPLAN_FORMATION && !hideFormationOverlayManually) {
                res.autoFormation?.let { report ->
                    drawAutoFormationScanner(canvas, w, h, report)
                    return
                }
            }

            // 2. CHẾ ĐỘ TRONG TRẬN ĐẤU (60 FPS ON-PITCH AR VISION):
            // Giữ Minimap trong suốt tuyệt đối, chỉ kẻ vạch việt vị và 1 tia mảnh nối bóng tới người nhận
            roiRect.set(w * roiLeftRatio, h * roiTopRatio, w * roiRightRatio, h * roiBottomRatio)
            canvas.drawRect(roiRect, minimapFramePaint)

            val offX = roiRect.left + res.offsideLineX * roiRect.width()
            canvas.drawLine(offX, roiRect.top, offX, roiRect.bottom, offsideLinePaint)

            res.minimapBestPass?.let { (fromMap, toMap) ->
                val mx1 = roiRect.left + fromMap.x * roiRect.width()
                val my1 = roiRect.top + fromMap.y * roiRect.height()
                val mx2 = roiRect.left + toMap.x * roiRect.width()
                val my2 = roiRect.top + toMap.y * roiRect.height()
                drawCleanArrow(canvas, mx1, my1, mx2, my2, throughLaserPaint)
            }

            // Vẽ vòng tròn khóa vị trí dưới bàn chân cầu thủ ông đang điều khiển trên Sân 3D
            res.controlledFoot3D?.let { foot ->
                val fx = foot.x * w
                val fy = foot.y * h
                canvas.save()
                canvas.scale(1f, 0.52f, fx, fy)
                canvas.drawCircle(fx, fy, dp(22).toFloat(), footRingPaint)
                canvas.restore()
            }

            // Vẽ tối đa 2 Tia Laser đường chuyền trực tiếp trên mặt cỏ 3D
            res.pitch3DOptions.forEachIndexed { idx, opt ->
                val sx = opt.from3DX * w
                val sy = opt.from3DY * h
                val ex = opt.to3DX * w
                val ey = opt.to3DY * h

                val paint = if (idx == 0) throughLaserPaint else safePassPaint
                drawCleanArrow(canvas, sx, sy, ex, ey, paint)

                // Vẽ Vòng Tròn Điểm Rơi (Landing Target) trên mặt cỏ 3D
                canvas.save()
                canvas.scale(1f, 0.52f, ex, ey)
                if (idx == 0) {
                    canvas.drawCircle(ex, ey, dp(26).toFloat(), landingZoneFillPaint)
                }
                canvas.drawCircle(ex, ey, dp(26).toFloat(), paint)
                canvas.restore()

                // Nếu đồng đội đang chạy chỗ NGOÀI KHUNG HÌNH CAMERA -> Hiện ký hiệu Radar Mép Màn Hình
                val label = when {
                    opt.isOffScreen -> ">>> CÁNH TRỐNG (${opt.openSpacePct}%)"
                    opt.isThroughRun -> "⚡ CHỌC KHE (${opt.openSpacePct}%)"
                    else -> "✔ (${opt.openSpacePct}%)"
                }
                tagTextPaint.color = if (idx == 0) Color.rgb(167, 243, 208) else Color.YELLOW
                canvas.drawText(label, (ex - dp(38)).coerceIn(dp(10).toFloat(), w - dp(130)), (ey - dp(16)).coerceAtLeast(dp(24).toFloat()), tagTextPaint)
            }

            // Cảnh báo Đỏ trên mặt cỏ 3D nếu có cầu thủ địch đang lén đứng trống ở sân nhà
            res.dangerRunner3D?.let { danger ->
                val dx = danger.x * w
                val dy = danger.y * h
                canvas.save()
                canvas.scale(1f, 0.52f, dx, dy)
                canvas.drawCircle(dx, dy, dp(28).toFloat(), dangerRingPaint)
                canvas.restore()
                tagTextPaint.color = Color.rgb(255, 100, 100)
                canvas.drawText("⚠ BỌC LÓT!", dx - dp(32), dy - dp(18), tagTextPaint)
            }
        }

        // Giao diện Mảng 1 Tự Động khi phát hiện màn hình Sơ đồ chiến thuật (Game Plan)
        private fun drawAutoFormationScanner(
            canvas: Canvas,
            w: Float,
            h: Float,
            report: AutoFormationAnalysis
        ) {
            if (report.isWaitingForOpponentView) {
                // Nhắc nhẹ ở mép trên khi người chơi đang nhìn đội hình của chính mình
                cardRect.set(w * 0.22f, dp(10).toFloat(), w * 0.78f, dp(52).toFloat())
                canvas.drawRoundRect(cardRect, dp(10).toFloat(), dp(10).toFloat(), modalBgPaint)
                canvas.drawRoundRect(cardRect, dp(10).toFloat(), dp(10).toFloat(), footRingPaint)
                tagTextPaint.color = Color.CYAN
                canvas.drawText(
                    "👁 MẢNG 1 TỰ ĐỘNG: Hãy bấm xem Sơ đồ Đối thủ 1 giây để AI quét tử huyệt!",
                    cardRect.left + dp(14),
                    cardRect.centerY() + dp(5),
                    tagTextPaint
                )
                return
            }

            // Khi người chơi vừa chuyển sang xem sơ đồ đối thủ -> Khoanh vùng tử huyệt ngay lập tức!
            for (card in report.detectedPlayerCards) {
                canvas.drawCircle(card.x * w, card.y * h, dp(14).toFloat(), footRingPaint)
            }

            for (weak in report.weakZoneSpots) {
                val wx = weak.x * w
                val wy = weak.y * h
                canvas.drawCircle(wx, wy, dp(40).toFloat(), dangerRingPaint)
                tagTextPaint.color = Color.YELLOW
                canvas.drawText("🎯 KHOÉT VÙNG NÀY", wx - dp(52), wy + dp(5), tagTextPaint)
            }

            cardRect.set(w * 0.18f, dp(10).toFloat(), w * 0.86f, dp(105).toFloat())
            canvas.drawRoundRect(cardRect, dp(12).toFloat(), dp(12).toFloat(), modalBgPaint)
            canvas.drawRoundRect(cardRect, dp(12).toFloat(), dp(12).toFloat(), throughLaserPaint)

            var ty = cardRect.top + dp(24)
            val lx = cardRect.left + dp(14)

            tagTextPaint.color = Color.rgb(56, 189, 248)
            canvas.drawText("🤖 SƠ ĐỒ ĐỐI THỦ: ${report.detectedFormation}", lx, ty, tagTextPaint)

            ty += dp(22)
            tagTextPaint.color = Color.rgb(250, 204, 21)
            canvas.drawText("🔥 TỬ HUYỆT: ${report.tacticalWeakness}", lx, ty, tagTextPaint)

            ty += dp(22)
            tagTextPaint.color = Color.WHITE
            canvas.drawText("💡 ${report.actionableAdvice}", lx, ty, tagTextPaint)
        }

        private fun drawCleanArrow(canvas: Canvas, sx: Float, sy: Float, ex: Float, ey: Float, paint: Paint) {
            canvas.drawLine(sx, sy, ex, ey, paint)
            val angle = atan2((ey - sy).toDouble(), (ex - sx).toDouble())
            val head = dp(11).toFloat()
            val a1 = angle - Math.PI / 6
            val a2 = angle + Math.PI / 6
            canvas.drawLine(ex, ey, (ex - head * cos(a1)).toFloat(), (ey - head * sin(a1)).toFloat(), paint)
            canvas.drawLine(ex, ey, (ex - head * cos(a2)).toFloat(), (ey - head * sin(a2)).toFloat(), paint)
        }
    }
}
