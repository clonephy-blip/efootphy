package com.example.minimap.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.example.minimap.service.FloatingOverlayService

class MainActivity : ComponentActivity() {
    private val overlayPermission = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { if (Settings.canDrawOverlays(this)) requestNotificationsAndStart() }

    private val notificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { startBubble() } // Denial does not prohibit a foreground service.

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 80, 32, 32)
            addView(TextView(context).apply {
                text = "Bật bubble → mở game → chạm bubble để cấp quyền và quét."
            })
            addView(Button(context).apply {
                text = "Bật bubble"
                setOnClickListener {
                    if (Settings.canDrawOverlays(this@MainActivity)) {
                        requestNotificationsAndStart()
                    } else {
                        overlayPermission.launch(Intent(
                            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:$packageName")
                        ))
                    }
                }
            })
            addView(Button(context).apply {
                text = "Dừng ứng dụng nổi"
                setOnClickListener {
                    stopService(Intent(this@MainActivity, FloatingOverlayService::class.java))
                }
            })
        }
        setContentView(layout)
    }

    private fun requestNotificationsAndStart() {
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        else startBubble()
    }

    private fun startBubble() {
        startForegroundService(Intent(this, FloatingOverlayService::class.java).apply {
            action = FloatingOverlayService.ACTION_SHOW
        })
    }
}
