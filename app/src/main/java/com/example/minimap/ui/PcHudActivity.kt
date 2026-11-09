package com.example.minimap.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.example.minimap.service.PcHudOverlayService

/** Phone role: show the PC's results. This entry point never loads a model. */
class PcHudActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 40, 32, 32)
        }
        root.addView(TextView(this).apply {
            text = "eFootPhy · PC + USB\n\nPC nhận diện và tracking. Điện thoại chỉ vẽ HUD.\n\n1. Cắm USB và cho phép USB debugging.\n2. Bật HUD bên dưới.\n3. Chạy start_pc_brain.bat trên PC.\n\nChạm hoặc kéo nút HUD để ẩn/hiện và di chuyển. Dừng từ ứng dụng hoặc thông báo."
            textSize = 17f
        })
        root.addView(Button(this).apply {
            text = "BẬT HUD"
            setOnClickListener {
                if (!Settings.canDrawOverlays(this@PcHudActivity)) {
                    startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName")))
                } else {
                    startForegroundService(Intent(this@PcHudActivity, PcHudOverlayService::class.java))
                }
            }
        })
        root.addView(Button(this).apply {
            text = "DỪNG HUD"
            setOnClickListener { stopService(Intent(this@PcHudActivity, PcHudOverlayService::class.java)) }
        })
        setContentView(root)
    }
}
