package com.example.minimap.ui

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
import androidx.activity.ComponentActivity
import com.example.minimap.service.FloatingOverlayService
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private lateinit var statusLabel: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildMainScreen())
    }

    override fun onResume() {
        super.onResume()
        if (::statusLabel.isInitialized) {
            val granted = Settings.canDrawOverlays(this)
            statusLabel.text = if (granted) {
                "✅ Quyền Overlay đã sẵn sàng — Bấm 'BẬT ENGINE 60FPS' rồi vào game!"
            } else {
                "⚠ Cần cấp quyền Hiện trên ứng dụng khác ở lần chạy đầu tiên"
            }
            statusLabel.setTextColor(if (granted) Color.rgb(74, 222, 128) else Color.rgb(250, 204, 21))
        }
    }

    private fun buildMainScreen(): View {
        val scroll = ScrollView(this).apply {
            setBackgroundColor(Color.rgb(15, 23, 42))
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(24), dp(18), dp(32))
        }

        root.addView(TextView(this).apply {
            text = "⚡ eFootball Real-Time 60FPS Vision (v4.0)"
            textSize = 20f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.WHITE)
        })

        statusLabel = TextView(this).apply {
            textSize = 13f
            setPadding(0, dp(8), 0, dp(16))
        }
        root.addView(statusLabel)

        val startBtn = Button(this).apply {
            text = "🚀 BẬT ENGINE 60FPS (AUTO AR + SCAN)"
            setBackgroundColor(Color.rgb(16, 185, 129))
            setTextColor(Color.WHITE)
            setPadding(dp(12), dp(14), dp(12), dp(14))
            setOnClickListener {
                if (!Settings.canDrawOverlays(this@MainActivity)) {
                    startActivity(
                        Intent(
                            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:$packageName")
                        )
                    )
                } else {
                    startService(
                        Intent(this@MainActivity, FloatingOverlayService::class.java)
                            .setAction(FloatingOverlayService.ACTION_SHOW)
                    )
                }
            }
        }
        root.addView(startBtn)

        val stopBtn = Button(this).apply {
            text = "⏹ TẮT ENGINE"
            setBackgroundColor(Color.rgb(239, 68, 68))
            setTextColor(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(10) }
            setOnClickListener {
                startService(
                    Intent(this@MainActivity, FloatingOverlayService::class.java)
                        .setAction(FloatingOverlayService.ACTION_STOP)
                )
            }
        }
        root.addView(stopBtn)

        // Thanh tinh chỉnh nhanh khung Minimap nếu đổi tỷ lệ màn hình khác
        val calibCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
            background = GradientDrawable().apply {
                setColor(Color.rgb(30, 41, 59))
                cornerRadius = dp(12).toFloat()
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(18) }
        }

        val calibTitle = TextView(this).apply {
            text = "🔧 Tinh chỉnh độ rộng khung Minimap (Mặc định chuẩn 20:9): 13%"
            textSize = 13f
            setTextColor(Color.rgb(203, 213, 225))
        }
        calibCard.addView(calibTitle)

        val seek = SeekBar(this).apply {
            max = 20
            progress = 9
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    val halfW = 0.046f + (p * 0.002f)
                    FloatingOverlayService.roiLeftRatio = 0.50f - halfW
                    FloatingOverlayService.roiRightRatio = 0.50f + halfW
                    val pct = ((halfW * 2f) * 100f).roundToInt()
                    calibTitle.text = "🔧 Tinh chỉnh độ rộng khung Minimap: $pct%"
                }
                override fun onStartTrackingTouch(s: SeekBar?) = Unit
                override fun onStopTrackingTouch(s: SeekBar?) = Unit
            })
        }
        calibCard.addView(seek)
        root.addView(calibCard)

        val infoCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
            background = GradientDrawable().apply {
                setColor(Color.rgb(30, 41, 59))
                cornerRadius = dp(12).toFloat()
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(16) }
        }

        infoCard.addView(TextView(this).apply {
            text = """
                🎯 MỌI THỨ ĐÃ TỰ ĐỘNG HÓA 100% TRONG BẢN v4.0:

                1. Tự động nhận diện Màn hình Sơ đồ (Mảng 1):
                • Khi bạn mở màn hình Đội hình trước trận, app tự biết và nhắc bạn bấm sang xem Sơ đồ đối thủ 1 giây.
                • Ngay khi hiện sơ đồ đối thủ, AI tự đọc đội hình, khoanh tròn ĐỎ vào đúng lỗ hổng của địch và nạp chiến thuật vào Radar!

                2. Tự động khóa hướng tấn công (Hiệp 1 & Hiệp 2):
                • App tự nhìn vị trí Thủ môn và Con trỏ Xanh Ngọc để biết bạn đang công sang Phải (AUTO →) hay sang Trái (AUTO ←). Không cần bấm đổi tay nữa!

                3. Tầm nhìn 60/60 FPS trực tiếp trên mặt cỏ 3D (Mảng 3):
                • Vòng tròn Xanh Ngọc bám dưới bàn chân cầu thủ bạn đang cầm (như João Cancelo).
                • Tia Laser Xanh Lá phóng thẳng từ chân cầu thủ của bạn tới điểm rơi bóng trống trải nhất trên mặt cỏ 3D!
                • Nếu đồng đội chạy chỗ ngoài khung hình Camera, mép màn hình sẽ nháy '>>> CÁNH TRỐNG' để bạn phất bóng chuẩn xác!
            """.trimIndent()
            textSize = 13f
            setTextColor(Color.rgb(226, 232, 240))
        })
        root.addView(infoCard)

        scroll.addView(root)
        return scroll
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).roundToInt()
}
