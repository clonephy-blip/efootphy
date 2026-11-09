package com.example.minimap.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.TextView
import com.example.minimap.hud.HudPacket
import com.example.minimap.hud.HudServer
import com.example.minimap.ui.PcHudActivity
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.hypot

/** No MediaProjection, image reader, analyzer or inference worker in this service. */
class PcHudOverlayService : Service() {
    private val latest = AtomicReference<HudPacket?>(null)
    private val main = Handler(Looper.getMainLooper())
    private lateinit var windows: WindowManager
    private var server: HudServer? = null
    private var hud: HudView? = null
    private var bubble: TextView? = null
    private var visible = true
    @Volatile private var destroyed = false

    override fun onCreate() {
        super.onCreate()
        windows = requireNotNull(getSystemService(WindowManager::class.java))
        requireNotNull(getSystemService(NotificationManager::class.java)).createNotificationChannel(
            NotificationChannel("pc-hud", "eFootPhy PC HUD", NotificationManager.IMPORTANCE_LOW))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP" || !Settings.canDrawOverlays(this)) {
            stopSelf()
            return START_NOT_STICKY
        }
        val stop = PendingIntent.getService(this, 1,
            Intent(this, PcHudOverlayService::class.java).setAction("STOP"),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val open = PendingIntent.getActivity(this, 2, Intent(this, PcHudActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notification = Notification.Builder(this, "pc-hud")
            .setSmallIcon(android.R.drawable.ic_menu_view).setContentTitle("eFootPhy · PC HUD")
            .setContentText("Nhận kết quả từ PC qua USB. Chạm để mở.")
            .setContentIntent(open).setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Dừng", stop).build()).build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(41, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else startForeground(41, notification)
        if (hud == null) attachWindows()
        if (server == null) server = HudServer(
            onPacketReceived = { packet ->
                if (!destroyed) { latest.set(packet); hud?.postInvalidateOnAnimation() }
            },
            onConnectionChanged = { connected ->
                if (!connected) latest.set(null)
                main.post {
                    if (!destroyed) {
                        bubble?.text = if (connected) "HUD ✓" else "HUD …"
                        hud?.invalidate()
                    }
                }
            }
        ).apply { start() }
        return START_NOT_STICKY
    }

    private fun layout(width: Int, height: Int, flags: Int) = WindowManager.LayoutParams(
        width, height, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY, flags,
        PixelFormat.TRANSLUCENT).apply { gravity = Gravity.TOP or Gravity.START }

    private fun attachWindows() {
        val view = HudView()
        windows.addView(view, layout(-1, -1, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN)
            .apply { alpha = .75f })
        hud = view
        val control = TextView(this).apply {
            text = "HUD …"; textSize = 13f; setTextColor(Color.WHITE)
            setBackgroundColor(Color.rgb(15, 23, 42)); gravity = Gravity.CENTER
            setOnClickListener { visible = !visible; hud?.invalidate() }
        }
        val params = layout(dp(82), dp(44), WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE)
            .apply { x = dp(16); y = dp(70) }
        var downX = 0f; var downY = 0f; var startX = 0; var startY = 0; var moved = false
        val slop = ViewConfiguration.get(this).scaledTouchSlop
        control.setOnTouchListener { v, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY; startX = params.x; startY = params.y; moved = false
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX-downX; val dy = event.rawY-downY
                    if (hypot(dx, dy) > slop) moved = true
                    if (moved) {
                        params.x = (startX+dx.toInt()).coerceIn(0, (resources.displayMetrics.widthPixels-dp(82)).coerceAtLeast(0))
                        params.y = (startY+dy.toInt()).coerceIn(0, (resources.displayMetrics.heightPixels-dp(44)).coerceAtLeast(0))
                        windows.updateViewLayout(control, params)
                    }
                }
                MotionEvent.ACTION_UP -> if (!moved) v.performClick()
            }
            true
        }
        windows.addView(control, params)
        bubble = control
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        destroyed = true
        server?.stop(); server = null
        main.removeCallbacksAndMessages(null)
        hud?.let { it.removeCallbacks(it.expire); runCatching { windows.removeView(it) } }
        bubble?.let { runCatching { windows.removeView(it) } }
        latest.set(null); hud = null; bubble = null
        super.onDestroy()
    }

    private fun dp(value: Int) = (value*resources.displayMetrics.density).toInt()

    private inner class HudView : View(this@PcHudOverlayService) {
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val expire = Runnable { invalidate() }
        private val dashed = DashPathEffect(floatArrayOf(8f, 8f), 0f)

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            removeCallbacks(expire)
            if (!visible) return
            val packet = latest.get() ?: return
            if (SystemClock.elapsedRealtime()-packet.receivedAtMs+packet.sourceAgeMs >= 350f) return
            postDelayed(expire, 60L)
            val w = width.toFloat(); val h = height.toFloat()
            paint.style = Paint.Style.FILL; paint.pathEffect = null
            paint.color = Color.WHITE; paint.textSize = dp(13).toFloat()
            if (packet.sourceWidth <= 0 || packet.sourceHeight <= 0 ||
                abs(packet.sourceWidth.toFloat()/packet.sourceHeight-w/h)/(w/h) > .05f) {
                canvas.drawText("PC: cần khớp khung hình scrcpy với màn hình điện thoại", dp(12).toFloat(), dp(28).toFloat(), paint)
                return
            }
            canvas.drawText("AI PC: ${packet.fps.toInt()} Hz · xử lý ${packet.latencyMs.toInt()} ms", dp(12).toFloat(), dp(28).toFloat(), paint)
            paint.strokeWidth = dp(2).toFloat(); paint.style = Paint.Style.STROKE
            for (player in packet.players) {
                if (!listOf(player.x1, player.y1, player.x2, player.y2).all { it.isFinite() && it in 0f..1f } ||
                    player.x1 > player.x2 || player.y1 > player.y2) continue
                paint.color = when(player.team) {
                    0 -> Color.CYAN; 1 -> Color.rgb(255, 100, 60)
                    2 -> Color.YELLOW; else -> Color.LTGRAY
                }
                canvas.drawOval(player.feetX*w-dp(10), player.feetY*h-dp(4),
                    player.feetX*w+dp(10), player.feetY*h+dp(4), paint)
                if (player.isControlled) canvas.drawCircle(player.feetX*w, player.y1*h-dp(6), dp(4).toFloat(), paint)
            }
            for (shadow in packet.shadowCandidates) {
                val x = shadow.x*w; val y = shadow.y*h
                paint.color = Color.MAGENTA
                canvas.drawLine(x-dp(4), y, x+dp(4), y, paint)
                canvas.drawLine(x, y-dp(4), x, y+dp(4), paint)
            }
            packet.ball?.let { ball ->
                val predicted = ball.state != "OBSERVED"
                paint.color = if(predicted) Color.LTGRAY else Color.YELLOW
                paint.pathEffect = if(predicted) dashed else null
                val x = ball.x*w; val y = ball.y*h
                canvas.drawCircle(x, y, dp(7).toFloat(), paint)
                canvas.drawLine(x, y, x+ball.vx*w*.15f, y+ball.vy*h*.15f, paint)
                paint.pathEffect = null
            }
            // Pass hints are deliberately not rendered until possession/geometry QC passes.
        }
    }
}
