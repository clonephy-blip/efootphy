const net = require('net');

// Create bridge between WSL (port 8889) and ADB Windows (port 8888)
const server = net.createServer(wslSocket => {
    const adbSocket = net.connect(8888, '127.0.0.1');
    wslSocket.pipe(adbSocket).pipe(wslSocket);
    adbSocket.on('error', err => {});
    wslSocket.on('error', err => {});
});

server.listen(8889, '0.0.0.0', () => {
    console.log('[Bridge] Relay listening on 0.0.0.0:8889 -> 127.0.0.1:8888');
});
