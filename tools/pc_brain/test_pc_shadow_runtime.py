"""Real postprocessing with a fake model: no Windows capture/ORT initialization."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np

from tools.pc_brain.runtime_safety import select_execution_providers


class ProviderSelectionTest(unittest.TestCase):
    def test_auto_prefers_installed_gpu(self):
        self.assertEqual(select_execution_providers('auto',['CPUExecutionProvider','CUDAExecutionProvider']),
                         ['CUDAExecutionProvider','CPUExecutionProvider'])

    def test_explicit_gpu_does_not_silently_fall_back(self):
        with self.assertRaises(RuntimeError): select_execution_providers('cuda',['CPUExecutionProvider'])

    def test_cpu_is_explicit(self):
        self.assertEqual(select_execution_providers('cpu',['DmlExecutionProvider','CPUExecutionProvider']),['CPUExecutionProvider'])

    def test_invalid_provider_is_rejected(self):
        with self.assertRaises(ValueError): select_execution_providers('guess',[])


class RuntimeBallStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('efootball_pc_brain.py')
        spec=importlib.util.spec_from_file_location('pc_brain_isolated_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'onnxruntime':types.ModuleType('onnxruntime'),'mss':types.ModuleType('mss')}):
            spec.loader.exec_module(module)
        cls.module=module

    def setUp(self):
        self.brain=self.module.EFootballPcBrain.__new__(self.module.EFootballPcBrain)
        b=self.brain
        b.input_size=320;b.input_name='images';b.kit_classifier=self.module.TeamKitClassifier()
        b.prev_ball=None;b.prev_ball_time=0.;b.last_ball_observed_s=float('-inf')
        b.ball_persistence=0;b.last_known_ball=None;b.last_ball_v=(0.,0.)
        b.shadow_enabled=False;b.latest_shadow_candidates=[]
        self.output=np.zeros((1,11,2100),np.float32)
        b.sess=types.SimpleNamespace(run=lambda *_: [self.output])
        self.frame=np.empty((320,320,3),np.uint8);self.frame[:]=(40,140,40)

    def run_frame(self,now,ball):
        self.output[:]=0
        if ball is not None:
            self.output[0,:4,0]=[*ball,8,8];self.output[0,4,0]=.9
        with patch.object(self.module.time,'perf_counter',side_effect=[now-.02,now-.01,now]):
            return self.brain.run_detection(self.frame)[0]

    def test_observed_predicted_then_lost_are_different_states(self):
        self.assertEqual(self.run_frame(1.,(100,100))['state'],'OBSERVED')
        self.assertEqual(self.run_frame(1.1,None)['state'],'PREDICTED')
        self.assertIsNone(self.run_frame(1.3,None))

    def test_reacquisition_does_not_use_prediction_timestamp_for_velocity(self):
        self.run_frame(1.,(100,100))
        self.run_frame(1.05,None)
        ball=self.run_frame(1.1,(110,100))
        self.assertAlmostEqual(ball['vx'],100/320,places=5)

    def test_long_gap_resets_velocity(self):
        self.run_frame(1.,(100,100))
        ball=self.run_frame(2.,(200,100))
        self.assertEqual(ball['vx'],0.)

    def test_shadow_proposal_does_not_change_observed_ball(self):
        first=self.run_frame(1.,(100,100))
        self.brain.shadow_enabled=True
        second=self.run_frame(1.1,(100,100))
        self.assertEqual((first['x'],first['y']),(second['x'],second['y']))
        self.assertEqual(second['state'],'OBSERVED')

    def test_shadow_cannot_create_ball_without_detection_history(self):
        self.brain.shadow_enabled=True
        self.assertIsNone(self.run_frame(1.,None))
        self.assertEqual(self.brain.latest_shadow_candidates,[])

    def test_unverified_tactics_never_run_in_live_path(self):
        with patch.object(self.module.TacticalPassAnalyzer, 'analyze', side_effect=AssertionError('not gated')):
            self.run_frame(1., (100,100))

    def test_missing_or_minimized_scrcpy_does_not_capture_other_windows(self):
        capture=self.module.WindowCapture.__new__(self.module.WindowCapture)
        capture.window_title='eFootball-Screen'
        for handle,visible,minimized in [(0,True,False),(7,False,False),(7,True,True)]:
            capture.user32=types.SimpleNamespace(
                FindWindowW=lambda *args:handle,IsWindow=lambda *args:True,
                IsWindowVisible=lambda *args:visible,IsIconic=lambda *args:minimized)
            self.assertIsNone(capture.find_target_window())

    def test_visible_exact_scrcpy_window_is_selected(self):
        capture=self.module.WindowCapture.__new__(self.module.WindowCapture)
        capture.window_title='eFootball-Screen'
        def exact_title(_,title):
            self.assertEqual(title,'eFootball-Screen')
            return 7
        capture.user32=types.SimpleNamespace(
            FindWindowW=exact_title,IsWindow=lambda *args:True,
            IsWindowVisible=lambda *args:True,IsIconic=lambda *args:False)
        self.assertEqual(capture.find_target_window(),7)


if __name__=='__main__':unittest.main()
