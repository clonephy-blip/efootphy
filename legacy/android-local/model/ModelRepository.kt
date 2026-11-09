package com.example.minimap.model

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream

data class LocalModel(val sha256: String,val model: File,val manifest: JSONObject)

class ModelRepository(private val context: Context) {
    private val preferences=context.getSharedPreferences("vision_models",Context.MODE_PRIVATE)
    private val root=File(context.filesDir,"models")
    private var cached: LocalModel?=null

    fun current(): LocalModel? {
        val hash=preferences.getString("active_minimap",null) ?: return null
        if(!hash.matches(Regex("[a-f0-9]{64}"))) return null
        cached?.takeIf { it.sha256==hash }?.let { return it }
        val folder=File(root,hash)
        val manifest=File(folder,"manifest.json")
        val model=File(folder,"model.onnx")
        if(!manifest.isFile || !model.isFile) return null
        return runCatching {
            val metadata=JSONObject(manifest.readText())
            require(metadata.getString("model_sha256")==hash)
            LocalModel(hash,model,metadata).also { cached=it }
        }.getOrNull()
    }

    fun importBundle(uri: Uri): LocalModel {
        val staging=File.createTempFile("model-import-","",context.cacheDir)
        check(staging.delete() && staging.mkdir())
        try {
            val input=requireNotNull(context.contentResolver.openInputStream(uri))
            val seen=mutableSetOf<String>()
            ZipInputStream(input).use { zip ->
                while(true) {
                    val entry=zip.nextEntry ?: break
                    require(entry.name in setOf("model.onnx","manifest.json") && !entry.isDirectory && seen.add(entry.name)) {
                        "Gói cần đúng model.onnx và manifest.json ở thư mục gốc"
                    }
                    val limit=if(entry.name=="manifest.json") 128*1024L else 40*1024*1024L
                    var total=0L
                    File(staging,entry.name).outputStream().use { output ->
                        val buffer=ByteArray(32*1024)
                        while(true) {
                            val count=zip.read(buffer)
                            if(count<0) break
                            total+=count
                            require(total<=limit) { "Gói mô hình vượt kích thước cho phép" }
                            output.write(buffer,0,count)
                        }
                    }
                    zip.closeEntry()
                }
            }
            require(seen==setOf("model.onnx","manifest.json")) { "Gói thiếu mô hình hoặc cấu hình" }
            val manifest=JSONObject(File(staging,"manifest.json").readText())
            require(manifest.getInt("schema_version")==1 && manifest.getString("profile")=="minimap" &&
                manifest.getString("task")=="detect" && manifest.getInt("input_size")==320 &&
                manifest.getString("input_layout")=="NCHW" && manifest.getString("output_layout")=="channels_first" &&
                manifest.getString("normalization")=="RGB/255") { "Cấu hình mô hình chưa được hỗ trợ" }
            val classes=manifest.getJSONArray("classes")
            require(classes.length()==3 && (0..2).map { classes.getString(it) }==listOf("ball","controlled","dot")) {
                "Thứ tự nhãn mô hình không khớp"
            }
            val model=File(staging,"model.onnx")
            val hash=model.inputStream().use { inputStream ->
                val digest=MessageDigest.getInstance("SHA-256")
                val buffer=ByteArray(32*1024)
                while(true) { val n=inputStream.read(buffer);if(n<0)break;digest.update(buffer,0,n) }
                digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
            }
            require(hash==manifest.getString("model_sha256")) { "Mã kiểm tra mô hình không khớp" }
            createSession(model).use { validateShape(it) }
            root.mkdirs()
            val destination=File(root,hash)
            if(!destination.exists()) check(staging.renameTo(destination)) { "Không lưu được mô hình" }
            check(preferences.edit().putString("active_minimap",hash).commit())
            cached=null
            return requireNotNull(current())
        } finally { if(staging.exists()) staging.deleteRecursively() }
    }

    companion object {
        fun createSession(model: File): OrtSession = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(1)
            options.setInterOpNumThreads(1)
            OrtEnvironment.getEnvironment().createSession(model.absolutePath,options)
        }
        fun validateShape(session: OrtSession) {
            require(session.inputInfo.size==1 && session.outputInfo.size==1)
            val input=session.inputInfo.values.single().info as? TensorInfo
            val output=session.outputInfo.values.single().info as? TensorInfo
            require(input!=null && input.type==OnnxJavaType.FLOAT && input.shape.contentEquals(longArrayOf(1,3,320,320)))
            require(output!=null && output.type==OnnxJavaType.FLOAT && output.shape.contentEquals(longArrayOf(1,7,2100))) {
                "Dạng đầu ra mô hình chưa được hỗ trợ"
            }
        }
    }
}
