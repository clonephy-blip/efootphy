package com.example.minimap.ui

import android.content.Context
import android.content.Intent
import android.graphics.Color
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
    private lateinit var modelStatus: TextView
    private lateinit var importButton: Button
    private val modelPicker=registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if(uri!=null) importModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(screen())
    }
    override fun onResume() { super.onResume();refreshStatus() }

    private fun refreshStatus() {
        if(!::status.isInitialized) return
        status.text=if(Settings.canDrawOverlays(this)) "Quyền cửa sổ nổi đã sẵn sàng" else "Cấp quyền cửa sổ nổi khi bật trợ lý lần đầu"
        modelStatus.text=if(runCatching { ModelRepository(this).current() }.getOrNull()!=null)
            "Đã nhập mô hình radar · chế độ kiểm tra" else "Chưa nhập mô hình radar"
    }

    private fun importModel(uri: Uri) {
        importButton.isEnabled=false;modelStatus.text="Đang kiểm tra gói mô hình…"
        Thread({
            val outcome=runCatching { ModelRepository(applicationContext).importBundle(uri) }
            runOnUiThread {
                if(!isDestroyed) {
                    importButton.isEnabled=true;refreshStatus()
                    Toast.makeText(this,outcome.fold({ "Đã nhập mô hình radar" },{ it.message ?: "Không nhập được mô hình" }),Toast.LENGTH_LONG).show()
                }
            }
        },"model-import").start()
    }

    private fun screen(): View {
        val root=LinearLayout(this).apply {
            orientation=LinearLayout.VERTICAL;setPadding(dp(18),dp(24),dp(18),dp(24))
            setBackgroundColor(Color.rgb(15,23,42))
        }
        fun text(value: String,size: Float=14f)=TextView(this).apply {
            text=value;textSize=size;setTextColor(Color.WHITE);setPadding(0,dp(9),0,dp(9))
        }
        root.addView(text("eFootball Assistant · bản nghiên cứu",21f))
        status=text("");root.addView(status)
        modelStatus=text("");root.addView(modelStatus)
        importButton=Button(this).apply {
            text="1. Nhập gói mô hình radar"
            setOnClickListener { modelPicker.launch(arrayOf("application/zip","application/octet-stream","application/x-zip-compressed")) }
        }
        root.addView(importButton)
        root.addView(Button(this).apply {
            text="2. Bật trợ lý"
            setOnClickListener {
                if(!Settings.canDrawOverlays(this@MainActivity)) {
                    startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,Uri.parse("package:$packageName")))
                } else startForegroundService(Intent(this@MainActivity,FloatingOverlayService::class.java).setAction(FloatingOverlayService.ACTION_SHOW))
            }
        })
        root.addView(Button(this).apply {
            text="Dừng trợ lý"
            setOnClickListener { stopService(Intent(this@MainActivity,FloatingOverlayService::class.java)) }
        })
        root.addView(text("3. Chỉnh vùng radar theo khung viền trên màn hình"))
        val prefs=getSharedPreferences("roi",Context.MODE_PRIVATE)
        FloatingOverlayService.loadRoi(this)
        fun saveRoi() {
            val region=FloatingOverlayService.roi
            prefs.edit().putFloat("left",region.left).putFloat("top",region.top)
                .putFloat("right",region.right).putFloat("bottom",region.bottom).apply()
        }
        fun slider(title: String,initial: Float,update: (Float)->Float) {
            val label=text("$title: ${"%.1f".format(initial*100)}%")
            root.addView(label)
            root.addView(SeekBar(this).apply {
                max=1000;progress=(initial*1000).roundToInt()
                setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(bar: SeekBar?,value: Int,fromUser: Boolean) {
                        if(fromUser) { val actual=update(value/1000f);label.text="$title: ${"%.1f".format(actual*100)}%" }
                    }
                    override fun onStartTrackingTouch(bar: SeekBar?)=Unit
                    override fun onStopTrackingTouch(bar: SeekBar?) { saveRoi() }
                })
            })
        }
        slider("Vị trí ngang",FloatingOverlayService.roi.left) { value ->
            FloatingOverlayService.roi=FloatingOverlayService.roi.moveX(value)
            FloatingOverlayService.roi.left
        }
        slider("Vị trí dọc",FloatingOverlayService.roi.top) { value ->
            FloatingOverlayService.roi=FloatingOverlayService.roi.moveY(value)
            FloatingOverlayService.roi.top
        }
        slider("Chiều rộng",FloatingOverlayService.roi.width) { value ->
            FloatingOverlayService.roi=FloatingOverlayService.roi.resizeWidth(value)
            FloatingOverlayService.roi.width
        }
        slider("Chiều cao",FloatingOverlayService.roi.height) { value ->
            FloatingOverlayService.roi=FloatingOverlayService.roi.resizeHeight(value)
            FloatingOverlayService.roi.height
        }
        root.addView(text("Bản này dùng để kiểm tra nhận diện và theo dõi. Dữ liệu được xử lý trên máy. Gợi ý chuyền và mũi tên trên sân sẽ mở sau khi kiểm chứng chất lượng."))
        return ScrollView(this).apply { addView(root) }
    }

    private fun dp(value: Int)=(value*resources.displayMetrics.density).roundToInt()
}
