package com.example.minimap.ui

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.example.minimap.service.FloatingOverlayService
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private lateinit var statusLabel: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildTacticalHubUi())
    }

    override fun onResume() {
        super.onResume()
        refreshOverlayStatus()
    }

    private fun refreshOverlayStatus() {
        if (!::statusLabel.isInitialized) return
        val granted = Settings.canDrawOverlays(this)
        statusLabel.text = if (granted) {
            "✅ Quyền Overlay đã sẵn sàng — Bấm 'BẬT RADAR HUD' trước khi vào trận"
        } else {
            "⚠ Chưa cấp quyền Hiện trên ứng dụng khác — Bấm nút bên dưới để mở khóa"
        }
        statusLabel.setTextColor(if (granted) Color.rgb(74, 222, 128) else Color.rgb(251, 191, 36))
    }

    private fun buildTacticalHubUi(): View {
        val scroll = ScrollView(this).apply {
            setBackgroundColor(Color.rgb(15, 23, 42))
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(20), dp(16), dp(32))
        }

        // Tiêu đề chính
        root.addView(TextView(this).apply {
            text = "⚽ eFootball Tactical AI Hub v2.0"
            textSize = 21f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.WHITE)
        })
        root.addView(TextView(this).apply {
            text = "Tích hợp Mảng 1 (Phân tích Chiến thuật & Đội hình) + Mảng 3 (Radar HUD Thời gian thực)"
            textSize = 13f
            setTextColor(Color.rgb(148, 163, 184))
            setPadding(0, dp(4), 0, dp(16))
        })

        // ==================== KHỐI 1: ĐIỀU KHIỂN MẢNG 3 (RADAR HUD & CĂN CHỈNH MINIMAP) ====================
        root.addView(createCardSection("📡 MẢNG 3: ĐIỀU KHIỂN RADAR HUD & KHUNG MINIMAP") { card ->
            statusLabel = TextView(this).apply {
                textSize = 13f
                setPadding(0, 0, 0, dp(10))
            }
            card.addView(statusLabel)

            val btnRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
            }
            val startBtn = Button(this).apply {
                text = "🚀 BẬT RADAR HUD"
                setBackgroundColor(Color.rgb(16, 185, 129))
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    rightMargin = dp(6)
                }
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
            val stopBtn = Button(this).apply {
                text = "⏹ TẮT HUD"
                setBackgroundColor(Color.rgb(239, 68, 68))
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    leftMargin = dp(6)
                }
                setOnClickListener {
                    startService(
                        Intent(this@MainActivity, FloatingOverlayService::class.java)
                            .setAction(FloatingOverlayService.ACTION_STOP)
                    )
                }
            }
            btnRow.addView(startBtn)
            btnRow.addView(stopBtn)
            card.addView(btnRow)

            // Thanh trượt tinh chỉnh độ rộng khung Minimap nếu người dùng đổi điện thoại khác
            val calibText = TextView(this).apply {
                text = "🔧 Tinh chỉnh độ rộng khung Minimap (Mặc định đã chuẩn màn 20:9): ${(FloatingOverlayService.roiRightRatio - FloatingOverlayService.roiLeftRatio).times(100).roundToInt()}%"
                textSize = 12f
                setTextColor(Color.rgb(203, 213, 225))
                setPadding(0, dp(12), 0, dp(4))
            }
            card.addView(calibText)

            val widthSeek = SeekBar(this).apply {
                max = 20
                progress = 6 // Tương ứng độ rộng ~11.6% màn hình (0.442..0.558)
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                        val halfWidth = 0.045f + (progress * 0.002f)
                        FloatingOverlayService.roiLeftRatio = 0.50f - halfWidth
                        FloatingOverlayService.roiRightRatio = 0.50f + halfWidth
                        val pct = ((halfWidth * 2f) * 100f).roundToInt()
                        calibText.text = "🔧 Tinh chỉnh độ rộng khung Minimap: $pct% chiều ngang màn hình"
                    }
                    override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                    override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                })
            }
            card.addView(widthSeek)
        })

        // ==================== KHỐI 2: MẢNG 1 — MÁY TÍNH KHẮC CHẾ LỐI CHƠI ĐỐI THỦ ====================
        root.addView(createCardSection("🧠 MẢNG 1A: KHẮC CHẾ CHIẾN THUẬT & PLAYSTYLE ĐỐI THỦ") { card ->
            card.addView(makeLabel("Chọn Lối Chơi (Team Playstyle) của Đối Thủ:"))
            val styles = listOf(
                "Quick Counter (Phản công nhanh - Dâng cao)",
                "Long Ball Counter (Phản công bóng dài - Lùi sâu)",
                "Possession Game (Kiểm soát bóng - Đan bật)",
                "Out Wide (Tấn công biên - Tạt cánh)",
                "Long Ball (Bóng dài trực diện)"
            )
            val styleSpinner = makeSpinner(styles)
            card.addView(styleSpinner)

            val counterOutput = TextView(this).apply {
                textSize = 13f
                setTextColor(Color.rgb(226, 232, 240))
                setPadding(dp(10), dp(10), dp(10), dp(10))
                background = GradientDrawable().apply {
                    setColor(Color.rgb(15, 23, 42))
                    cornerRadius = dp(8).toFloat()
                }
            }
            card.addView(counterOutput)

            styleSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                    counterOutput.text = getPlaystyleCounterGuide(pos)
                }
                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
        })

        // ==================== KHỐI 3: MẢNG 1 — CHẤM ĐIỂM GHÉP CẶP CẦU THỦ (MIMO ENGINE) ====================
        root.addView(createCardSection("⚡ MẢNG 1B: CHẤM ĐIỂM GHÉP CẶP ĐỘI HÌNH (MIMO SYNERGY)") { card ->
            card.addView(makeLabel("1. Cặp Tiền Đạo (CF + CF/SS trong 4-2-2-2 / 4-3-1-2):"))
            val cfPairs = listOf(
                "Goal Poacher + Deep-Lying Forward (Ví dụ: Suárez + Griezmann)",
                "Goal Poacher + Fox in the Box / Target Man (Ví dụ: Suárez + Budimir)",
                "2x Goal Poacher (2 Tiền đạo săn bàn giăng ngang)",
                "2x Deep-Lying Forward / Creative Playmaker (Không có mũi nhọn)"
            )
            val cfSpinner = makeSpinner(cfPairs)
            card.addView(cfSpinner)

            card.addView(makeLabel("2. Cặp Tiền Vệ Trụ (DMF + CMF/DMF):"))
            val dmfPairs = listOf(
                "Anchor Man + Box-to-Box / Orchestrator (Chuẩn Meta)",
                "Anchor Man + Destroyer (Bê tông 2 lớp)",
                "Destroyer + Box-to-Box (Dễ bỏ vị trí khi dâng cao)"
            )
            val dmfSpinner = makeSpinner(dmfPairs)
            card.addView(dmfSpinner)

            card.addView(makeLabel("3. Cặp Trung Vệ (CB + CB):"))
            val cbPairs = listOf(
                "Build Up + Destroyer (Ví dụ: Van de Ven / Van Dijk + Ferdinand)",
                "2x Build Up (An toàn chống chọc khe)",
                "2x Destroyer (Rất dễ bị hút lên tạo lỗ hổng sau lưng)"
            )
            val cbSpinner = makeSpinner(cbPairs)
            card.addView(cbSpinner)

            val synergyOutput = TextView(this).apply {
                textSize = 13f
                setTextColor(Color.rgb(167, 243, 208))
                setPadding(dp(10), dp(10), dp(10), dp(10))
                background = GradientDrawable().apply {
                    setColor(Color.rgb(15, 23, 42))
                    cornerRadius = dp(8).toFloat()
                }
            }
            card.addView(synergyOutput)

            val updateSynergy = {
                synergyOutput.text = evaluateSquadSynergy(
                    cfSpinner.selectedItemPosition,
                    dmfSpinner.selectedItemPosition,
                    cbSpinner.selectedItemPosition
                )
            }

            val listener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) = updateSynergy()
                override fun onNothingSelected(p0: AdapterView<*>?) = Unit
            }
            cfSpinner.onItemSelectedListener = listener
            dmfSpinner.onItemSelectedListener = listener
            cbSpinner.onItemSelectedListener = listener
        })

        scroll.addView(root)
        refreshOverlayStatus()
        return scroll
    }

    private fun getPlaystyleCounterGuide(index: Int): String = when (index) {
        0 -> """
            🎯 ĐỐI THỦ ĐÁ: QUICK COUNTER (HLV Klopp / Mourinho)
            • Tử huyệt của địch: Hàng thủ dâng cực cao lên gần vạch giữa sân, khoảng trống sau lưng 2 hậu vệ cánh (LB/RB) rất lớn.
            • Lối chơi khắc chế cứng: Long Ball Counter (Beckenbauer) hoặc Quick Counter.
            • Sơ đồ & Phối hợp: Dùng 4-2-2-2 (2 AMF/LMF-RMF). Khi cướp được bóng, bật 1 chạm cho AMF rồi nhìn đèn xanh của Radar HUD phất chọc khe bổng (Through Pass) ra sau lưng hậu vệ biên.
            • Individual Instructions: Đặt 'Counter Target' lên 1 tiền đạo Goal Poacher để luôn rình ở vạch việt vị.
        """.trimIndent()

        1 -> """
            🎯 ĐỐI THỦ ĐÁ: LONG BALL COUNTER (Lùi sâu đổ bê tông)
            • Tử huyệt của địch: Hàng thủ lùi rất sâu về sát vòng cấm, nhường hoàn toàn khu vực 16m50 và trung tuyến.
            • Lối chơi khắc chế cứng: Possession Game hoặc Out Wide / Quick Counter.
            • Sơ đồ & Phối hợp: Đừng phất bóng dài vô ích! Dùng 1 Deep-Lying Forward / Hole Player kéo hậu vệ địch ra rồi sút xa (Stunning Shot) ngoài vòng cấm hoặc chồng biên tạt cho Target Man.
            • Individual Instructions: Đặt 'Defensive' cho 1 DMF (Anchor Man) để chống các pha phất dài phản công của địch.
        """.trimIndent()

        2 -> """
            🎯 ĐỐI THỦ ĐÁ: POSSESSION GAME (Kiểm soát bóng)
            • Tử huyệt của địch: Cự ly đội hình hẹp, hậu vệ dâng cao và giữ bóng lâu ở giữa sân.
            • Lối chơi khắc chế cứng: Quick Counter (Áp sát tầm cao).
            • Sơ đồ & Phối hợp: Không giữ nút Dash & Pressure đuổi theo bóng! Dùng nút Match-up che đường chuyền về DMF của địch, ngay khi cắt được bóng ở giữa sân là chọc khe thẳng vào nách trung vệ.
        """.trimIndent()

        3 -> """
            🎯 ĐỐI THỦ ĐÁ: OUT WIDE (Bám biên tạt cánh)
            • Tử huyệt của địch: Tuyến giữa rất mỏng do dồn người ra 2 biên.
            • Lối chơi khắc chế cứng: Quick Counter / Long Ball Counter đánh thẳng vào trung lộ.
            • Individual Instructions: Bắt buộc đặt 'Defensive' cho cả LB và RB hoặc dùng 'Tight Marking' khóa chết tiền đạo đánh đầu (Target Man) trong vòng cấm.
        """.trimIndent()

        else -> """
            🎯 ĐỐI THỦ ĐÁ: LONG BALL (Bóng dài trực diện)
            • Tử huyệt của địch: Bỏ qua hàng tiền vệ, thường xuyên mất bóng hai (Second ball) ở giữa sân.
            • Cách khắc chế: Dùng cặp CB cao to có Aerial Superiority (như Van Dijk / Ferdinand) để tranh chấp bóng một, và đặt 1 Anchor Man thu hồi bóng hai ngay trước vòng cấm.
        """.trimIndent()
    }

    private fun evaluateSquadSynergy(cfIdx: Int, dmfIdx: Int, cbIdx: Int): String {
        var score = 100
        val notes = mutableListOf<String>()

        when (cfIdx) {
            0 -> notes.add("✔ Hàng công (10/10): 1 người lùi làm tường (DLF) kéo CB địch lên + 1 người bứt tốc phá bẫy việt vị (Goal Poacher). Cực hợp với mắt thần Radar HUD!")
            1 -> {
                score -= 4
                notes.add("✔ Hàng công (9.5/10): 1 làm tường tỳ đè (Target Man/FITB) + 1 chạy chỗ. Rất mạnh cả chọc khe lẫn tạt cánh.")
            }
            2 -> {
                score -= 10
                notes.add("⚠ Hàng công (8.5/10): 2 Goal Poacher cùng cắm đầu chạy thẳng, thiếu người lùi xuống nối bóng với hàng tiền vệ.")
            }
            else -> {
                score -= 22
                notes.add("❌ Hàng công (7/10): Thiếu hẳn mũi nhọn đâm sau lưng hàng thủ đối phương!")
            }
        }

        when (dmfIdx) {
            0 -> notes.add("✔ Tuyến giữa (10/10): Anchor Man giữ vị trí + Box-to-Box/Orchestrator lên công về thủ. Cân bằng tuyệt đối.")
            1 -> {
                score -= 5
                notes.add("✔ Tuyến giữa (9/10): Thủ cực chắc nhưng hơi thiếu người dâng lên hỗ trợ sút tuyến 2.")
            }
            else -> {
                score -= 18
                notes.add("❌ Tuyến giữa (7.5/10): Cả Destroyer và B2B đều hay lao lên cướp bóng, rất dễ để trống khoảng trước mặt 2 CB!")
            }
        }

        when (cbIdx) {
            0 -> notes.add("✔ Hàng thủ (10/10): Destroyer lao ra cắt bóng sớm + Build Up bọc lót phía sau. Công thức chuẩn nhất eFootball.")
            1 -> {
                score -= 3
                notes.add("✔ Hàng thủ (9.5/10): 2 Build Up giữ vạch phòng ngự cực chuẩn, chống chọc khe tốt.")
            }
            else -> {
                score -= 25
                notes.add("❌ Hàng thủ (6.5/10): 2 Destroyer cùng lao lên dập tiền đạo sẽ bị đối phương bật tường 1-2 qua lưng ngay lập tức!")
            }
        }

        return "🏆 ĐIỂM TƯƠNG THÍCH ĐỘI HÌNH: $score / 100\n\n" + notes.joinToString("\n\n")
    }

    private fun createCardSection(title: String, builder: (LinearLayout) -> Unit): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
            background = GradientDrawable().apply {
                setColor(Color.rgb(30, 41, 59))
                cornerRadius = dp(12).toFloat()
                setStroke(dp(1), Color.rgb(51, 65, 85))
            }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(16) }
        }
        card.addView(TextView(this).apply {
            text = title
            textSize = 15f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.rgb(56, 189, 248))
            setPadding(0, 0, 0, dp(10))
        })
        builder(card)
        return card
    }

    private fun makeLabel(text: String) = TextView(this).apply {
        this.text = text
        textSize = 13f
        setTextColor(Color.rgb(203, 213, 225))
        setPadding(0, dp(8), 0, dp(4))
    }

    private fun makeSpinner(items: List<String>) = Spinner(this).apply {
        adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, items)
        setBackgroundColor(Color.rgb(226, 232, 240))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(8) }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).roundToInt()
}
