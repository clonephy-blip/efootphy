package com.example.minimap.processor

import android.content.Context
import com.efootphy.vision.Detection
import com.efootphy.vision.DetectorPostprocess
import com.efootphy.vision.FusionEngine
import com.efootphy.vision.FusionInput
import com.efootphy.vision.Kind
import com.efootphy.vision.Observation
import com.efootphy.vision.Scene
import com.efootphy.vision.Source
import com.efootphy.vision.Team
import com.efootphy.vision.Vec2
import com.efootphy.vision.WorldState
import com.example.minimap.model.ModelRepository
import com.example.minimap.model.OnnxMinimapDetector

/** Research pipeline. A detection score is not tactical confidence. */
data class TacticalResult(
    val timestampNs: Long,
    val screenMode: Int,
    val attackLeftToRight: Boolean,
    val world: WorldState?,
    val detections: List<Detection>,
    val status: String
)

class TacticalAnalyzer(context: Context): AutoCloseable {
    private val repository=ModelRepository(context.applicationContext)
    private val fusion=FusionEngine()
    private var detector: OnnxMinimapDetector?=null
    private var activeHash: String?=null
    private var loadError: String?=null
    private var roiKey: List<Int>?=null
    private var epoch=0L
    @Volatile var attackLeftToRight=true
    @Volatile var manualDirectionLock=false

    fun toggleManualDirection() { manualDirectionLock=true;attackLeftToRight=!attackLeftToRight }

    /** Called only on the capture worker after the previous callback has finished. */
    fun reset() {
        fusion.reset();epoch++;roiKey=null
        val old=detector;detector=null;activeHash=null;loadError=null
        runCatching { old?.close() }
    }

    fun analyze(frame: RgbaFrame): TacticalResult {
        fun result(message: String,world: WorldState?=null,detections: List<Detection> = emptyList()) =
            TacticalResult(frame.timestampNs,frame.screenMode,attackLeftToRight,world,detections,message)
        val key=listOf(frame.width,frame.height,frame.roiLeft,frame.roiTop,frame.captureWidth,frame.captureHeight)
        if(roiKey!=key) { fusion.reset();epoch++;roiKey=key }
        if(frame.screenMode!=ScreenMode.MODE_LIVE_MATCH) {
            fusion.update(FusionInput(frame.timestampNs,epoch,if(frame.screenMode==ScreenMode.MODE_GAMEPLAN_FORMATION) Scene.FORMATION else Scene.UNKNOWN))
            return result(if(frame.screenMode==ScreenMode.MODE_GAMEPLAN_FORMATION) "Đang ở màn hình đội hình" else "Chưa thấy cảnh thi đấu")
        }
        val model=repository.current()
        if(model==null) { fusion.reset();return result("Nhập mô hình để đọc radar") }
        if(activeHash!=model.sha256) {
            val old=detector;detector=null;runCatching { old?.close() };fusion.reset();epoch++
            activeHash=model.sha256
            loadError=null
            try { detector=OnnxMinimapDetector(model) }
            catch(_: Exception) { loadError="Không mở được mô hình; hãy nhập lại" }
        }
        val engine=detector ?: return result(loadError ?: "Chưa có mô hình")
        val detections=try { engine.detect(frame) } catch(_: Exception) {
            fusion.reset();return result("Không đọc được ảnh radar")
        }
        // dot/controlled may overlap on the same player. Marker roles are retained separately.
        val playerBoxes=mutableListOf<Detection>()
        for(box in detections.filter { it.label==1 || it.label==2 }.sortedByDescending { it.score }) {
            if(playerBoxes.none { DetectorPostprocess.iou(it.box,box.box)>.55 }) playerBoxes.add(box)
        }
        val observations=playerBoxes.map { d -> Observation(
            Vec2((d.box.left+d.box.right)/2,(d.box.top+d.box.bottom)/2),Team.UNKNOWN,d.score,
            frame.timestampNs,epoch,Source.MINIMAP,Kind.PLAYER)
        }+detections.filter { it.label==0 }.map { d -> Observation(
            Vec2((d.box.left+d.box.right)/2,(d.box.top+d.box.bottom)/2),Team.UNKNOWN,d.score,
            frame.timestampNs,epoch,Source.MINIMAP,Kind.BALL_MARKER)
        }
        val world=fusion.update(FusionInput(frame.timestampNs,epoch,Scene.LIVE,observations,
            modelsValidated=false))
        // Full-pitch/landmark models and independent device QC are required before enabling AR advice.
        return result(if(playerBoxes.isEmpty()) "Kiểm tra vùng radar" else "${playerBoxes.size} ứng viên · kiểm tra",world,detections)
    }

    override fun close() { reset() }
}
