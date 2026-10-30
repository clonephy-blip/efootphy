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
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import com.example.minimap.BuildConfig
import com.example.minimap.processor.TacticalAnalyzer
import com.example.minimap.processor.TacticalResult
import com.example.minimap.ui.CaptureConsentActivity
import com.example.minimap.ui.MainActivity
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

        // Đã đo chuẩn từng pixel từ ảnh chụp thực tế trận OL vs GEN (Màn hình 20:9)
        @Volatile var roiLeftRatio = 0.442f
        @Volatile var roiTopRatio = 0.798f
        @Volatile var roiRightRatio = 0.558f
        @Volatile var roiBottomRatio = 0.946f
        @Volatile var showDebugDots = true
    }

    private lateinit var windows: WindowManager
    private lateinit var capture: ScreenCaptureManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val analyzer = TacticalAnalyzer()

    private var bubble: TextView? = null
    private var drawingLayer: TacticalHudView? = null
    private var captureActive = false
    private var destroyed = false
    private var lastLogMs = 0L

    override fun onCreate() {
        super.onCreate()
        windows = requireNotNull(getSystemService(WindowManager::class.java))
        requireNotNull(getSystemService(NotificationManager::class.java)).createNotificationChannel(
            NotificationChannel(CHANNEL, "Minimap Assistant", NotificationManager.IMPORTANCE_LOW)
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
                mainHandler.post {
                    if (!destroyed && captureActive) {
                        drawingLayer?.updateTactics(result, analyzer.attackLeftToRight)
                    }
                }
                val now = SystemClock.elapsedRealtime()
                if (BuildConfig.DEBUG && now - lastLogMs >= 4_000) {
                    Log.d("Minimap", "Home=${result.homeDots.size}, Away=${result.awayDots.size}")
                    lastLogMs = now
                }
            },
            onState = { running ->
                if (!destroyed) {
                    captureActive = running
                    updateBubbleText()
                    drawingLayer?.visibility = if (running) View.VISIBLE else View.GONE
                    updateForeground(running)
                }
            },
            onError = { error ->
                Log.e("Minimap", "Capture failed", error)
                Toast.makeText(this, "Không chụp được màn hình; chạm bubble để thử lại.", Toast.LENGTH_LONG).show()
            }
        )
    }

    private fun updateBubbleText() {
        val dirArrow = if (analyzer.attackLeftToRight) "→" else "←"
        bubble?.text = if (captureActive) "ON$dirArrow" else "MM"
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
            if (bubble == null) showWindows()
            if (startCapture) {
                captureActive = true
                bubble?.text = "…"
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
        val hud = TacticalHudView(this).apply {
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
                alpha = 0.78f // Chuẩn pass-through Android 12+
            }
        )

        val icon = TextView(this).apply {
            text = "MM"
            contentDescription = "Chạm bật/tắt, Giữ 1s đổi sân Hiệp 1/Hiệp 2"
            gravity = Gravity.CENTER
            textSize = 13f
            setTextColor(Color.WHITE)
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.rgb(20, 95, 170))
                setStroke(dp(2), Color.CYAN)
            }
            setOnClickListener {
                if (captureActive) {
                    text = "…"
                    capture.stop()
                } else {
                    startActivity(
                        Intent(this@FloatingOverlayService, CaptureConsentActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            }
            setOnLongClickListener {
                analyzer.attackLeftToRight = !analyzer.attackLeftToRight
                updateBubbleText()
                val dirText = if (analyzer.attackLeftToRight) "Trái sang Phải (→)" else "Phải sang Trái (←)"
                Toast.makeText(this@FloatingOverlayService, "Đổi hướng tấn công: $dirText", Toast.LENGTH_SHORT).show()
                true
            }
        }
        bubble = icon
        val lp = params(dp(50), dp(50), WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE)
            .apply { x = dp(18); y = dp(80) }
        val slop = ViewConfiguration.get(this).scaledTouchSlop
        var downX = 0f
        var downY = 0f
        var originX = 0
        var originY = 0
        var dragging = false
        var longPressTriggered = false
        val longPressRunnable = Runnable {
            if (!dragging) {
                longPressTriggered = true
                icon.performLongClick()
            }
        }

        icon.setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX
                    downY = event.rawY
                    originX = lp.x
                    originY = lp.y
                    dragging = false
                    longPressTriggered = false
                    mainHandler.postDelayed(longPressRunnable, ViewConfiguration.getLongPressTimeout().toLong())
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (abs(dx) > slop || abs(dy) > slop) {
                        dragging = true
                        mainHandler.removeCallbacks(longPressRunnable)
                    }
                    if (dragging) {
                        lp.x = originX + dx.roundToInt()
                        lp.y = originY + dy.roundToInt()
                        clampBubble(lp)
                        windows.updateViewLayout(view, lp)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    mainHandler.removeCallbacks(longPressRunnable)
                    if (!dragging && !longPressTriggered) view.performClick()
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    mainHandler.removeCallbacks(longPressRunnable)
                    true
                }
                else -> false
            }
        }
        windows.addView(icon, lp)
    }

    private fun params(width: Int, height: Int, flags: Int) = WindowManager.LayoutParams(
        width, height, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        flags, PixelFormat.TRANSLUCENT
    ).apply { gravity = Gravity.TOP or Gravity.LEFT }

    @Suppress("DEPRECATION")
    private fun clampBubble(lp: WindowManager.LayoutParams) {
        val size = android.graphics.Point()
        if (Build.VERSION.SDK_INT >= 30) {
            val bounds = windows.maximumWindowMetrics.bounds
            size.set(bounds.width(), bounds.height())
        } else windows.defaultDisplay.getSize(size)
        lp.x = lp.x.coerceIn(0, (size.x - lp.width).coerceAtLeast(0))
        lp.y = lp.y.coerceIn(0, (size.y - lp.height - dp(48)).coerceAtLeast(0))
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        bubble?.let { view ->
            val lp = view.layoutParams as WindowManager.LayoutParams
            clampBubble(lp)
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
            .setContentTitle("eFootball Tactical AI v2.0")
            .setContentText(if (projecting) "Đang quét Radar & Phân tích khoảng trống" else "Bubble sẵn sàng")
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
        listOfNotNull(bubble, drawingLayer).forEach {
            if (it.isAttachedToWindow) windows.removeView(it)
        }
        bubble = null
        drawingLayer = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    // Lớp HUD 2 Tầng: Tầng 1 (Bảng báo hiệu lớn phía trên) + Tầng 2 (Sa bàn Minimap & Mũi tên giữa sân)
    private inner class TacticalHudView(context: Context) : View(context) {
        private var currentResult: TacticalResult? = null
        private var leftToRight = true
        private val roiRect = RectF()
        private val bannerRect = RectF()

        private val roiBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.argb(210, 0, 255, 200)
        }
        private val dangerBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.RED
        }
        private val offsideLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.YELLOW
            pathEffect = DashPathEffect(floatArrayOf(10f, 8f), 0f)
        }
        private val defLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.CYAN
            pathEffect = DashPathEffect(floatArrayOf(8f, 8f), 0f)
        }
        private val homeDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(2).toFloat()
            color = Color.CYAN
        }
        private val targetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.YELLOW
        }
        private val passPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = dp(3).toFloat()
            color = Color.rgb(0, 255, 100)
        }
        private val bannerBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(190, 15, 23, 42)
        }
        private val bannerTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = dp(14).toFloat()
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            isFakeBoldText = true
        }

        fun updateTactics(result: TacticalResult, attackDir: Boolean) {
            currentResult = result
            leftToRight = attackDir
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return

            // 1. Khung Minimap đã đo chuẩn từ ảnh thực tế
            roiRect.set(
                w * roiLeftRatio,
                h * roiTopRatio,
                w * roiRightRatio,
                h * roiBottomRatio
            )

            val res = currentResult
            canvas.drawRect(roiRect, if (res?.defenseAlert == true) dangerBorderPaint else roiBorderPaint)
            if (res == null) return

            // 2. Vẽ BẢNG ĐIỆN TỬ CHIẾN THUẬT ở chính giữa mép trên màn hình (Trong tầm mắt người chơi)
            val bannerW = w * 0.46f
            val bannerH = dp(32).toFloat()
            bannerRect.set((w - bannerW) / 2f, dp(10).toFloat(), (w + bannerW) / 2f, dp(10) + bannerH)

            bannerBgPaint.color = when (res.bannerColorType) {
                1 -> Color.argb(215, 6, 95, 70)   // Xanh lá đậm khi có cơ hội Chọc khe / Khoảng trống
                2 -> Color.argb(215, 153, 27, 27) // Đỏ đậm khi bị phản công / Hở sườn
                else -> Color.argb(175, 15, 23, 42)
            }
            canvas.drawRoundRect(bannerRect, dp(8).toFloat(), dp(8).toFloat(), bannerBgPaint)
            canvas.drawText(
                res.headlineBanner,
                bannerRect.centerX(),
                bannerRect.centerY() + dp(5),
                bannerTextPaint
            )

            // 3. Vẽ Vạch Bẫy Việt Vị (Vàng đứt nét) & Vạch Phòng Ngự (Xanh đứt nét) lên Minimap
            val offX = roiRect.left + res.offsideLineX * roiRect.width()
            canvas.drawLine(offX, roiRect.top, offX, roiRect.bottom, offsideLinePaint)

            val defX = roiRect.left + res.defensiveLineX * roiRect.width()
            canvas.drawLine(defX, roiRect.top, defX, roiRect.bottom, defLinePaint)

            // 4. Đánh dấu các cầu thủ đội nhà đã nhận diện được (Vòng tròn xanh nhỏ)
            if (showDebugDots) {
                for (pt in res.homeDots) {
                    val px = roiRect.left + pt.x * roiRect.width()
                    val py = roiRect.top + pt.y * roiRect.height()
                    canvas.drawCircle(px, py, dp(4).toFloat(), homeDotPaint)
                }
            }

            // 5. Khoanh Vòng Tròn Vàng nổi bật vào Cầu Thủ Đang Trống Trải Nhất (Không ai kèm)
            res.unmarkedTarget?.let { target ->
                val tx = roiRect.left + target.x * roiRect.width()
                val ty = roiRect.top + target.y * roiRect.height()
                canvas.drawCircle(tx, ty, dp(9).toFloat(), targetPaint)

                // Vẽ thêm Mũi Tên Định Hướng Lớn ở Giữa Sân để mắt không cần nhìn xuống Minimap
                val centerX = w * 0.50f
                val centerY = h * 0.48f
                val dirX = if (leftToRight) centerX + dp(75) else centerX - dp(75)
                val dirY = when {
                    target.y < 0.35f -> centerY - dp(55) // Chỉ lên cánh trên
                    target.y > 0.65f -> centerY + dp(55) // Chỉ xuống cánh dưới
                    else -> centerY                      // Chỉ thẳng trung lộ
                }
                drawArrow(canvas, centerX, centerY, dirX, dirY, passPaint)
            }

            // 6. Vẽ Tuyến Chuyền Bóng An Toàn & Mũi Tên Chọc Khe
            for (lane in res.passLanes) {
                val sx = roiRect.left + lane.fromX * roiRect.width()
                val sy = roiRect.top + lane.fromY * roiRect.height()
                val ex = roiRect.left + lane.toX * roiRect.width()
                val ey = roiRect.top + lane.toY * roiRect.height()
                drawArrow(canvas, sx, sy, ex, ey, passPaint)
            }
        }

        private fun drawArrow(canvas: Canvas, sx: Float, sy: Float, ex: Float, ey: Float, paint: Paint) {
            canvas.drawLine(sx, sy, ex, ey, paint)
            val angle = atan2((ey - sy).toDouble(), (ex - sx).toDouble())
            val headLen = dp(9).toFloat()
            val a1 = angle - Math.PI / 6
            val a2 = angle + Math.PI / 6
            canvas.drawLine(ex, ey, (ex - headLen * cos(a1)).toFloat(), (ey - headLen * sin(a1)).toFloat(), paint)
            canvas.drawLine(ex, ey, (ex - headLen * cos(a2)).toFloat(), (ey - headLen * sin(a2)).toFloat(), paint)
        }
    }
}
