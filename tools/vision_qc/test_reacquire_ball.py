import builtins
import unittest
from unittest.mock import patch
import numpy as np
from .ball_video import NativeBallDetector
from .reacquire_ball import CoarseToFineBallDetector


class ConfirmationTests(unittest.TestCase):
    def detector(self,coarse,fine):
        d=object.__new__(CoarseToFineBallDetector)
        d.acquire=.25;d.proposal=.03;d.max_refinements=2;d.max_shift_px=24.;d.source_stretch=False
        calls=[]
        def detect(frame,rect,threshold=.15):
            calls.append(rect)
            return coarse if rect[2]-rect[0]==640 else fine
        d.detect_crop=detect
        return d,calls

    def test_weak_coarse_proposal_requires_nearby_native_confirmation(self):
        coarse={'xy_px':[700,300],'score':.1,'box_wh_px':[12,12]}
        fine={'xy_px':[702,299],'score':.8,'box_wh_px':[12,12]}
        d,calls=self.detector([coarse],[fine]);out,tiles=d.scan(np.zeros((1080,2340,3),np.uint8))
        self.assertEqual(tiles,9);self.assertEqual(len(calls),9)
        self.assertEqual(out[0]['xy_px'],[702,299]);self.assertEqual(out[0]['reacquisition_evidence'],'same_frame_coarse_to_native')

    def test_distant_native_object_is_not_confirmation(self):
        d,_=self.detector([{'xy_px':[700,300],'score':.1}], [{'xy_px':[750,300],'score':.9}])
        out,_=d.scan(np.zeros((1080,2340,3),np.uint8));self.assertEqual(out,[])

    def test_strong_coarse_detection_has_no_extra_forward_pass(self):
        d,calls=self.detector([{'xy_px':[700,300],'score':.8}],[])
        _,tiles=d.scan(np.zeros((1080,2340,3),np.uint8));self.assertEqual(tiles,8);self.assertEqual(len(calls),8)

    def test_low_native_score_cannot_start_a_track(self):
        d,_=self.detector([{'xy_px':[700,300],'score':.1}],[{'xy_px':[700,300],'score':.24}])
        out,_=d.scan(np.zeros((1080,2340,3),np.uint8));self.assertEqual(out,[])

    def test_masked_proposals_do_not_spend_native_budget(self):
        d,calls=self.detector([{'xy_px':[700,100],'score':.1}],[])
        out,tiles=d.scan(np.zeros((1080,2340,3),np.uint8),None,[(0,0,2340,140)])
        self.assertEqual(out,[]);self.assertEqual(tiles,8)

    def test_active_track_keeps_one_native_roi(self):
        d,calls=self.detector([],[]);_,tiles=d.scan(np.zeros((1080,2340,3),np.uint8),[700,300])
        self.assertEqual(tiles,1);self.assertEqual(calls,[(540,140,860,460)])

    def test_native_confirmation_budget_is_bounded(self):
        coarse=[{'xy_px':[100+i*100,300],'score':.1} for i in range(12)]
        d,calls=self.detector(coarse,[]);_,tiles=d.scan(np.zeros((1080,2340,3),np.uint8))
        self.assertEqual(tiles,10);self.assertEqual(len(calls),10)


class LocalBackendTests(unittest.TestCase):
    def test_opencv_path_does_not_import_onnxruntime(self):
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='onnxruntime':raise AssertionError('Unexpected runtime import')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded),patch('cv2.dnn.readNetFromONNX') as read:
            read.return_value.forward.return_value=np.zeros((1,11,2100),np.float32)
            model=NativeBallDetector('local.onnx',backend='opencv')
            self.assertEqual(model.backend,'opencv')

    def test_opencv_output_shape_is_checked(self):
        with patch('cv2.dnn.readNetFromONNX') as read:
            read.return_value.forward.return_value=np.zeros((1,5,2100),np.float32)
            with self.assertRaises(ValueError):NativeBallDetector('local.onnx',backend='opencv')
