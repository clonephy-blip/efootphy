"""PC↔Android RTT protocol tests without USB, Windows, or ML dependencies."""
import unittest

from tools.pc_brain.transport_rtt import AckRttMeter


class TransportRttTests(unittest.TestCase):
    def test_roundtrip_same_pc_clock(self):
        meter = AckRttMeter()
        meter.record_sent(7, now=5.0)
        self.assertEqual(len(meter.feed(b'{"ack":7}\n', now=5.042)), 1)
        self.assertAlmostEqual(meter.last_rtt_ms, 42.0, places=5)
        self.assertEqual(meter.ack_count, 1)

    def test_fragmented_and_batched_acks(self):
        meter = AckRttMeter()
        meter.record_sent(1, now=1.0)
        meter.record_sent(2, now=1.0)
        self.assertEqual(meter.feed(b'{"ac', now=1.01), [])
        result = meter.feed(b'k":1}\n{"ack":2}\n', now=1.02)
        self.assertEqual(len(result), 2)
        self.assertEqual(meter.ack_count, 2)

    def test_does_not_accept_unmatched_invalid_or_replayed_ids(self):
        meter = AckRttMeter()
        meter.record_sent(3, now=2.0)
        invalid = b'{"ack":true}\n{"ack":"3"}\n{"ack":4}\nnot-json\n'
        self.assertEqual(meter.feed(invalid, now=2.1), [])
        self.assertEqual(len(meter.feed(b'{"ack":3}\n{"ack":3}\n', now=2.2)), 1)

    def test_never_compare_remote_clock(self):
        meter = AckRttMeter()
        meter.record_sent(8, now=10)
        meter.feed(b'{"ack":8,"phone_elapsed_ms":9000000}\n', now=10.01)
        self.assertAlmostEqual(meter.last_rtt_ms, 10.0, places=5)

    def test_bounds_and_reset(self):
        meter = AckRttMeter(max_pending=2, max_buffer=10)
        for sequence in (1, 2, 3):
            meter.record_sent(sequence, now=1.0)
        self.assertNotIn(1, meter.pending)
        self.assertEqual(meter.feed(b"not a complete frame", now=2.0), [])
        self.assertEqual(len(meter.buffer), 0)
        meter.reset()
        self.assertEqual(len(meter.pending), 0)
        self.assertIsNone(meter.last_rtt_ms)

    def test_future_timestamp_is_rejected(self):
        meter = AckRttMeter()
        meter.record_sent(1, now=30.0)
        self.assertEqual(meter.feed(b'{"ack":1}\n', now=29.0), [])
        self.assertIsNone(meter.last_rtt_ms)


if __name__ == "__main__":
    unittest.main()
