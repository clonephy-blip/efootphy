import unittest

import numpy as np

from .mainview_video import decode_field,prepare_frame


class MainviewVideoTests(unittest.TestCase):
    def test_explicit_stretch_source_adapter_restores_both_axes(self):
        frame=np.zeros((360,780,3),np.uint8)
        tensor,t=prepare_frame(frame,source_stretch=True)
        self.assertEqual(tensor.shape,(1,3,640,640))
        raw=np.zeros((1,11,8400),np.float32)
        raw[0,:4,0]=[200*640/780,150*640/360,40*640/780,80*640/360];raw[0,10,0]=.9
        result=decode_field(raw,t)
        np.testing.assert_allclose(result[0]['box_xyxy_px'],[180,110,220,190],atol=1e-4)
        self.assertEqual(result[0]['class_name'],'referee')

    def test_letterbox_restores_original_pixel_boxes(self):
        frame=np.full((360,780,3),(35,110,40),np.uint8)
        tensor,t=prepare_frame(frame);self.assertEqual(tensor.shape,(1,3,640,640))
        left,top,r,_,_=t;raw=np.zeros((1,11,8400),np.float32)
        raw[0,:4,0]=[200*r+left,150*r+top,40*r,80*r];raw[0,8,0]=.9
        result=decode_field(raw,t)
        np.testing.assert_allclose(result[0]['box_xyxy_px'],[180,110,220,190],atol=1e-4)
        self.assertEqual(result[0]['class_name'],'player');self.assertIsNone(result[0]['player_identity'])

    def test_nms_does_not_suppress_different_semantic_classes(self):
        raw=np.zeros((1,11,8400),np.float32)
        raw[0,:4,:3]=np.array([[100,100,100],[100,100,100],[20,20,20],[40,40,40]])
        raw[0,8,0]=.9;raw[0,8,1]=.8;raw[0,4,2]=.7
        result=decode_field(raw,(0,0,1,640,640))
        self.assertEqual(sorted(d['class_name'] for d in result),['ball','player'])

    def test_empty_or_bad_model_output(self):
        raw=np.zeros((1,11,8400),np.float32)
        self.assertEqual(decode_field(raw,(0,0,1,640,640)),[])
        raw[0,0,0]=np.nan
        with self.assertRaises(ValueError):decode_field(raw,(0,0,1,640,640))
