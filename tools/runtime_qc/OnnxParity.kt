import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Desktop Java ORT parity against little-endian FP32 reference buffers.
 * This verifies Java runtime compatibility, not Android Bitmap preprocessing or device speed.
 * Args: model.onnx CASE_DIRECTORY; cases use N-input.bin and N-reference.bin starting at 0.
 */
fun main(args: Array<String>) {
    require(args.size==2)
    val directory=File(args[1])
    val environment=OrtEnvironment.getEnvironment()
    var count=0;var worst=0.0
    OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(1);options.setInterOpNumThreads(1)
        environment.createSession(args[0],options).use { session ->
            while(true) {
                val source=File(directory,"$count-input.bin")
                if(!source.isFile) break
                val data=source.readBytes();require(data.size==3*320*320*4)
                val reference=File(directory,"$count-reference.bin").readBytes()
                require(reference.size==7*2100*4)
                val input=ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val expected=ByteBuffer.wrap(reference).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                OnnxTensor.createTensor(environment,input,longArrayOf(1,3,320,320)).use { tensor ->
                    session.run(mapOf(session.inputNames.single() to tensor)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val actual=result[0].value as Array<Array<FloatArray>>
                        require(actual.size==1 && actual[0].size==7 && actual[0].all { it.size==2100 })
                        for(channel in actual[0]) for(value in channel) {
                            val target=expected.get()
                            require(value.isFinite() && target.isFinite())
                            worst=maxOf(worst,abs(value.toDouble()-target))
                        }
                    }
                }
                count++
            }
        }
    }
    require(count>0 && worst<=.05) { "Parity failed: cases=$count, maximum absolute error=$worst" }
    println("{\"cases\":$count,\"max_absolute_raw_output_error\":$worst,\"passed\":true,\"scope\":\"desktop Java ORT; Android preprocessing and device performance untested\"}")
    environment.close()
}
