import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .teacher_proposals import gate_points,JUNCTION_IDS


class TeacherGateTests(unittest.TestCase):
    def inputs(self):
        world=np.array([[i%8,i//8] for i in range(32)],float)
        xy=world*[60,70]+[80,120];known=np.c_[xy,np.full(32,2.)]
        frame=np.full((640,640,3),(35,110,40),np.uint8)
        for x,y in xy.astype(int):
            cv2.line(frame,(x-4,y),(x+4,y),(230,240,230),2)
            cv2.line(frame,(x,y-4),(x,y+4),(230,240,230),2)
        prediction=np.c_[xy,np.full(32,.9)]
        return frame,known,prediction,prediction.copy(),world

    def test_consensus_can_propose_an_unknown_junction_but_not_circle_ids(self):
        frame,known,a,b,world=self.inputs();known[1]=0
        result,gate=gate_points(frame,known,a,b,world,JUNCTION_IDS)
        self.assertEqual(result[1,2],2);self.assertTrue(next(x for x in gate['points'] if x['id']==1)['new_point'])
        for i in [8,10,11,14,15,18,19,21,30,31]:self.assertEqual(result[i,2],0)
        self.assertEqual(known[1,2],0)

    def test_two_view_disagreement_is_not_pseudo_ground_truth(self):
        frame,known,a,b,world=self.inputs();b[0,0]+=4
        result,gate=gate_points(frame,known,a,b,world,JUNCTION_IDS)
        self.assertEqual(result[0,2],0);self.assertEqual(gate['points'][0]['reason'],'two_view_disagreement')

    def test_geometry_and_white_line_are_both_required(self):
        frame,known,a,b,world=self.inputs();frame[:]=(35,110,40)
        result,gate=gate_points(frame,known,a,b,world,JUNCTION_IDS)
        self.assertEqual(np.count_nonzero(result[:,2]),0)
        self.assertIn('no_white_line_support',{q['reason'] for q in gate['points']})
        known[:]=0;result,gate=gate_points(frame,known,a,b,world,JUNCTION_IDS)
        self.assertEqual(gate['reason'],'too_few_reference_points')

    def test_confidence_rejection_and_fixed_coordinate_scale(self):
        frame,known,a,b,world=self.inputs();a[0,2]=.59
        result,_=gate_points(frame,known,a,b,world,JUNCTION_IDS);self.assertEqual(result[0,2],0)
        with self.assertRaises(ValueError):gate_points(frame[:320],known,a,b,world,JUNCTION_IDS)

    def test_reference_fit_is_reproducible(self):
        args=self.inputs();a=gate_points(*args,JUNCTION_IDS);b=gate_points(*args,JUNCTION_IDS)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        try:import torch
        except ImportError:self.skipTest('Optional training dependencies are not installed')
        self.torch=torch

    def test_cache_is_bit_exact_and_input_encoder_runtime_bound(self):
        from .feature_cache import FeatureCache
        torch=self.torch
        with tempfile.TemporaryDirectory() as directory:
            cache=FeatureCache(directory,'encoder-a','runtime-a');tensor=torch.ones(1,3,4,4)
            def compute():return [torch.arange(8).float().reshape(1,2,2,2)+i for i in range(3)]
            first,hit=cache.get(tensor,compute);self.assertFalse(hit)
            second,hit=cache.get(tensor,lambda:self.fail('Cache hit must not recompute'));self.assertTrue(hit)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(first,second)))
            other=FeatureCache(directory,'encoder-b','runtime-a');self.assertFalse(other.get(tensor,compute)[1])
            other=FeatureCache(directory,'encoder-a','runtime-b');self.assertFalse(other.get(tensor,compute)[1])
            self.assertFalse(cache.get(tensor+1,compute)[1])
            with self.assertRaisesRegex(ValueError,'Trainable'):
                cache.get(tensor+2,lambda:[torch.ones(1,2,2,2,requires_grad=True) for _ in range(3)])

    def test_content_corruption_is_rejected(self):
        from .feature_cache import FeatureCache
        torch=self.torch
        with tempfile.TemporaryDirectory() as directory:
            cache=FeatureCache(directory,'encoder','runtime');tensor=torch.ones(1,3,4,4)
            cache.get(tensor,lambda:[torch.ones(1,2,2,2) for _ in range(3)])
            path=cache.path(cache.metadata(tensor));payload=torch.load(path,weights_only=True)
            payload['features'][0][0,0,0,0]=7;torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'checksum'):cache.get(tensor,lambda:None)

    def test_no_negative_teacher_gradient_for_unlabeled_points(self):
        from .heatmap_pose import supervised_loss
        torch=self.torch;logits=torch.randn(2,32,4,4,requires_grad=True);vis=torch.zeros(2,32,requires_grad=True)
        points=torch.zeros(2,32,3);loss=supervised_loss(logits,vis,points,[True,True]);loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertEqual(float(logits.grad.abs().sum()),0);self.assertEqual(float(vis.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
