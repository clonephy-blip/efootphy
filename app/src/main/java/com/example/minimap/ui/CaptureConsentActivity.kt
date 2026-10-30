package com.example.minimap.ui

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionConfig
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.example.minimap.service.FloatingOverlayService

/** A visible Activity owns consent; a Service cannot use an ActivityResultLauncher. */
class CaptureConsentActivity : ComponentActivity() {
    private val consent = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data
        if (result.resultCode == Activity.RESULT_OK && data != null) {
            startForegroundService(Intent(this, FloatingOverlayService::class.java).apply {
                action = FloatingOverlayService.ACTION_CAPTURE
                putExtra(FloatingOverlayService.EXTRA_RESULT_CODE, result.resultCode)
                putExtra(FloatingOverlayService.EXTRA_RESULT_DATA, data)
            })
        }
        finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState != null) return
        val manager = requireNotNull(getSystemService(MediaProjectionManager::class.java))
        val request = if (Build.VERSION.SDK_INT >= 34) {
            manager.createScreenCaptureIntent(MediaProjectionConfig.createConfigForDefaultDisplay())
        } else manager.createScreenCaptureIntent()
        consent.launch(request)
    }
}
