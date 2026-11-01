import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from .run_control import RunControl


class RunControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.clock_value=0
        self.trainer=types.SimpleNamespace(epoch=0,last=self.root/'last.pt',best=self.root/'best.pt',
                                           metrics={'pose':.1},stop=False)
        self.trainer.ema=types.SimpleNamespace(updates=31)
        self.trainer.accumulate=16;self.trainer.args=types.SimpleNamespace(nbs=64)
        self.patch=patch.dict('sys.modules',{'torch':types.SimpleNamespace(get_num_threads=lambda:4)})
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_budget_stops_before_estimated_next_epoch_and_records_saved_checkpoint(self):
        c=RunControl(self.root,1,clock=lambda:self.clock_value);c.start()
        self.clock_value=20;c.epoch_end(self.trainer)
        self.assertFalse(self.trainer.stop)
        self.trainer.epoch=1;self.trainer.last.write_bytes(b'saved')
        self.clock_value=40;c.epoch_end(self.trainer)
        self.assertTrue(self.trainer.stop)
        data=json.loads((self.root/'progress.json').read_text())
        self.assertEqual(data['epochs_completed'],2);self.assertTrue(data['last_checkpoint_present'])
        self.assertEqual(data['optimizer_updates'],31)
        self.assertEqual(data['gradient_accumulation'],16)
        self.assertEqual(data['nominal_batch_size'],64)
        self.assertEqual(data['stop_reason'],'budget_next_epoch_estimate')
        self.assertFalse((self.root/'progress.json.tmp').exists())

    def test_stop_file_works_without_time_budget(self):
        c=RunControl(self.root,clock=lambda:0);c.start()
        (self.root/'STOP_AFTER_EPOCH').touch();c.epoch_end(self.trainer)
        self.assertTrue(self.trainer.stop);self.assertEqual(c.reason,'stop_file')

    def test_invalid_budgets_fail(self):
        for n in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):RunControl(self.root,n)

    def test_unstarted_control_does_not_emit_misleading_elapsed_time(self):
        with self.assertRaises(RuntimeError):RunControl(self.root).epoch_end(self.trainer)
