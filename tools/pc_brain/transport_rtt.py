"""Transport round-trip telemetry for eFootPhy PC→Android HUD.

Only times within the PC monotonic clock are compared. RTT is not camera latency,
end-to-end video latency, or Android render-to-photon time.
"""
import json
import time
from collections import OrderedDict


class AckRttMeter:
    def __init__(self, max_pending=128, max_buffer=8192):
        self.max_pending = max_pending
        self.max_buffer = max_buffer
        self.pending = OrderedDict()
        self.buffer = bytearray()
        self.last_rtt_ms = None
        self.ack_count = 0

    def record_sent(self, sequence, now=None):
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be nonnegative integer")
        self.pending[sequence] = time.perf_counter() if now is None else now
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)

    def feed(self, chunk, now=None):
        """Consume zero or more newline-delimited JSON acknowledgements."""
        if not isinstance(chunk, bytes):
            raise TypeError("ack chunk must be bytes")
        if len(self.buffer) + len(chunk) > self.max_buffer:
            self.buffer.clear()
            return []
        self.buffer.extend(chunk)
        observed = time.perf_counter() if now is None else now
        results = []
        while b"\n" in self.buffer:
            line, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, TypeError):
                continue
            if not isinstance(message, dict):
                continue
            sequence = message.get("ack")
            if type(sequence) is not int or sequence < 0:
                continue
            sent_time = self.pending.pop(sequence, None)
            if sent_time is None or observed < sent_time:
                continue
            rtt_ms = (observed - sent_time) * 1000.0
            self.last_rtt_ms = rtt_ms
            self.ack_count += 1
            results.append(rtt_ms)
        return results

    def reset(self):
        self.pending.clear()
        self.buffer.clear()
        self.last_rtt_ms = None
        self.ack_count = 0
