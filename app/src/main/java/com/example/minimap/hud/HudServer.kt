package com.example.minimap.hud

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

class HudServer(
    private val port: Int = 8888,
    private val onPacketReceived: (HudPacket) -> Unit,
    private val onConnectionChanged: (Boolean) -> Unit = {}
) {
    private var serverSocket: ServerSocket? = null
    private var serverThread: Thread? = null
    private val isRunning = AtomicBoolean(false)

    fun start() {
        if (isRunning.getAndSet(true)) return
        serverThread = Thread({
            try {
                serverSocket = ServerSocket(port).apply {
                    reuseAddress = true
                }
                Log.d("HudServer", "HUD Server listening on port $port")
                while (isRunning.get()) {
                    val client = try {
                        serverSocket?.accept() ?: break
                    } catch (e: Exception) {
                        break
                    }
                    handleClient(client)
                }
            } catch (e: Exception) {
                Log.e("HudServer", "Server exception: ${e.message}")
            } finally {
                stop()
            }
        }, "HudServer-Thread").apply {
            isDaemon = true
            start()
        }
    }

    private fun handleClient(socket: Socket) {
        onConnectionChanged(true)
        Log.d("HudServer", "PC Client connected from ${socket.remoteSocketAddress}")
        try {
            socket.tcpNoDelay = true
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), "UTF-8"))
            while (isRunning.get()) {
                val line = reader.readLine() ?: break
                if (line.isBlank()) continue
                val packet = HudPacket.fromJson(line)
                if (packet != null) {
                    onPacketReceived(packet)
                }
            }
        } catch (e: Exception) {
            Log.d("HudServer", "Client disconnected: ${e.message}")
        } finally {
            onConnectionChanged(false)
            runCatching { socket.close() }
        }
    }

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        runCatching { serverSocket?.close() }
        serverSocket = null
        serverThread?.interrupt()
        serverThread = null
        onConnectionChanged(false)
    }
}
