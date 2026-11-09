package com.example.minimap.ui

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.example.minimap.model.ModelRepository
import com.example.minimap.service.FloatingOverlayService
import kotlin.math.roundToInt

class MainActivity: ComponentActivity() {
    private lateinit var status: TextView
    private lateinit var pcStatus: TextView
    private var importButton: Button? = null

    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(screen())
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    private fun refreshStatus() {
        if (!::status.isInitialized) return
        val hasOverlay = Settings.canDrawOverlays(this)
        status.text = if (hasOverlay) "✅ Quyền cửa sổ nổi: ĐÃ SẴN SÀNG" else "⚠️ Quyền cửa sổ nổi: CHƯA CẤP (Chạm để cấp quyền)"
        status.setTextColor(if (hasOverlay) Color.rgb(16, 185, 129) else Color.rgb(245, 158, 11))
    }

    private fun importModel(uri: Uri) {
        importButton?.isEnabled = false
        Thread({
            val outcome = runCatching { ModelRepository(applicationContext).importBundle(uri) }
            runOnUiThread {
                if (!isDestroyed) {
                    importButton?.isEnabled = true
                    refreshStatus()
                    Toast.makeText(this, outcome.fold({ "Đã nạp mô hình kiểm tra" }, { it.message ?: "Lỗi nạp mô hình" }), Toast.LENGTH_LONG).show()
                }
            }
        }, "model-import").start()
    }

    private fun screen(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(24), dp(20), dp(24))
            setBackgroundColor(Color.rgb(15, 23, 42))
        }

        fun text(value: String, size: Float = 14f, bold: Boolean = false, color: Int = Color.WHITE) = TextView(this).apply {
            this.text = value
            textSize = size
            setTextColor(color)
            if (bold) typeface = Typeface.DEFAULT_BOLD
            setPadding(0, dp(6), 0, dp(6))
        }

        root.addView(text("⚽ eFootball AI Assistant", 22f, bold = true, color = Color.rgb(56, 189, 248)))
        root.addView(text("Mô hình Cách B: PC AI Co-Processor (Siêu mượt 60-120 FPS)", 12f, color = Color.rgb(148, 163, 184)))

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                cornerRadius = dp(10).toFloat()
                setColor(Color.rgb(30, 41, 59))
                setStroke(dp(1), Color.rgb(51, 65, 85))
            }
        }
        status = text("Kiểm tra quyền...", 13f)
        card.addView(status)
        pcStatus = text("⚡ Kênh USB: Port 8888 (adb forward tcp:8888 tcp:8888)", 13f, color = Color.rgb(56, 189, 248))
        card.addView(pcStatus)
        root.addView(card)

        fun btn(title: String, bgColor: Int, onClick: () -> Unit) = Button(this).apply {
            this.text = title
            setTextColor(Color.WHITE)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            background = GradientDrawable().apply {
                cornerRadius = dp(8).toFloat()
                setColor(bgColor)
            }
            setOnClickListener { onClick() }
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = dp(12)
            }
            layoutParams = lp
        }

        root.addView(btn("1. BẬT TRỢ LÝ (HUD MÀN HÌNH)", Color.rgb(16, 185, 129)) {
            if (!Settings.canDrawOverlays(this@MainActivity)) {
                startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
            } else {
                startForegroundService(Intent(this@MainActivity, FloatingOverlayService::class.java).setAction(FloatingOverlayService.ACTION_SHOW))
                Toast.makeText(this@MainActivity, "Trợ lý đã khởi động! Cắm cáp USB và chạy start_pc_brain.bat", Toast.LENGTH_SHORT).show()
            }
        })

        root.addView(btn("TẮT TRỢ LÝ", Color.rgb(239, 68, 68)) {
            stopService(Intent(this@MainActivity, FloatingOverlayService::class.java))
            Toast.makeText(this@MainActivity, "Đã dừng trợ lý", Toast.LENGTH_SHORT).show()
        })

        val guideCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                cornerRadius = dp(10).toFloat()
                setColor(Color.argb(120, 30, 41, 59))
            }
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = dp(16)
            }
            layoutParams = lp
        }
        guideCard.addView(text("📖 Hướng dẫn sử dụng cực nhanh:", 14f, bold = true, color = Color.rgb(226, 232, 240)))
        guideCard.addView(text("• Bước 1: Bấm 'BẬT TRỢ LÝ' ở trên (nút tròn nhỏ xuất hiện).", 12f, color = Color.rgb(203, 213, 225)))
        guideCard.addView(text("• Bước 2: Cắm cáp Type-C vào PC và chạy 'start_pc_brain.bat'.", 12f, color = Color.rgb(203, 213, 225)))
        guideCard.addView(text("• Bước 3: Vào game eFootball. Trợ lý PC tự nhận diện 2 đội + bóng + gợi ý chuyền!", 12f, color = Color.rgb(203, 213, 225)))
        root.addView(guideCard)

        val advTitle = text("⚙️ Tùy chọn nâng cao (Radar cục bộ)", 13f, bold = true, color = Color.rgb(148, 163, 184))
        root.addView(advTitle)

        val importBtn = Button(this).apply {
            this.text = "Nạp model radar on-device (tùy chọn)"
            textSize = 12f
            setOnClickListener {
                modelPicker.launch(arrayOf("application/zip", "application/octet-stream", "application/x-zip-compressed"))
            }
        }
        importButton = importBtn
        root.addView(importBtn)

        val prefs = getSharedPreferences("roi", Context.MODE_PRIVATE)
        FloatingOverlayService.loadRoi(this)
        fun saveRoi() {
            val region = FloatingOverlayService.roi
            prefs.edit().putFloat("left", region.left).putFloat("top", region.top)
                .putFloat("right", region.right).putFloat("bottom", region.bottom).apply()
        }
        fun slider(title: String, initial: Float, update: (Float) -> Float) {
            val label = text("$title: ${"%.1f".format(initial * 100)}%", 12f)
            root.addView(label)
            root.addView(SeekBar(this).apply {
                max = 1000
                progress = (initial * 1000).roundToInt()
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(bar: SeekBar?, value: Int, fromUser: Boolean) {
                        if (fromUser) {
                            val actual = update(value / 1000f)
                            label.text = "$title: ${"%.1f".format(actual * 100)}%"
                        }
                    }
                    override fun onStartTrackingTouch(bar: SeekBar?) = Unit
                    override fun onStopTrackingTouch(bar: SeekBar?) { saveRoi() }
                })
            })
        }
        slider("Vị trí ngang", FloatingOverlayService.roi.left) { value ->
            FloatingOverlayService.roi = FloatingOverlayService.roi.moveX(value)
            FloatingOverlayService.roi.left
        }
        slider("Vị trí dọc", FloatingOverlayService.roi.top) { value ->
            FloatingOverlayService.roi = FloatingOverlayService.roi.moveY(value)
            FloatingOverlayService.roi.top
        }

        return ScrollView(this).apply { addView(root) }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).roundToInt()
}
