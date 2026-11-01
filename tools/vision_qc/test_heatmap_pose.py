import unittest
import tempfile
from pathlib import Path

import numpy as np

from tools.vision_qc.heatmap_pose import transform_points


class TransformTests(unittest.TestCase):
    def test_source_edge_point_is_kept_for_consistent_denominators(self):
        result=transform_points([[320,320,2]],np.eye(3),320)
        np.testing.assert_array_equal(result,[[320,320,2]])

    def test_projective_labels_follow_pixels_and_missing_stays_missing(self):
        points=np.array([[10,20,2],[0,0,0],[39,30,2]],np.float32)
        matrix=np.array([[1,0,5],[0,1,-3],[0,0,1]],np.float32)
        result=transform_points(points,matrix,40)
        np.testing.assert_allclose(result[0],[15,17,2])
        np.testing.assert_array_equal(result[1:],0)
        np.testing.assert_array_equal(points[0],[10,20,2])

    def test_projective_denominator_and_border_are_guarded(self):
        matrix=np.array([[1,0,0],[0,1,0],[.1,0,-1]],np.float32)
        points=np.array([[10,10,2],[20,10,2]],np.float32)
        result=transform_points(points,matrix,40)
        np.testing.assert_array_equal(result[0],0)
        np.testing.assert_allclose(result[1],[20,10,2])


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        try:import torch
        except ImportError:self.skipTest('Optional training dependencies are not installed')
        self.torch=torch;torch.set_num_threads(1)

    def test_dense_heatmap_round_trip_in_original_pixels(self):
        from tools.vision_qc.heatmap_pose import gaussian_targets,decode
        torch=self.torch
        points=torch.tensor([[[13.2,19.7,2],[45.1,54.3,2],[5,5,0]]])
        targets,mask=gaussian_targets(points,80,.7)
        result=decode(targets.clamp_min(1e-30).log(),torch.ones(1,3)*10)
        expected=(points[...,:2]+.5)*8
        self.assertLess(float((result[:,:2,:2]-expected[:,:2]).abs().max()),.2)
        self.assertEqual(mask.tolist(),[[True,True,False]])
        self.assertTrue(torch.allclose(targets.sum((-2,-1)),torch.ones(1,3)))

    def test_missing_coordinate_has_no_spatial_gradient(self):
        from tools.vision_qc.heatmap_pose import gaussian_targets
        torch=self.torch
        points=torch.tensor([[[10.,10.,2.],[0.,0.,0.]]])
        targets,mask=gaussian_targets(points,20)
        logits=torch.randn_like(targets,requires_grad=True)
        spatial=-(targets*logits.flatten(2).log_softmax(-1).reshape_as(logits)).sum((-2,-1))
        (spatial*mask).sum().backward()
        self.assertEqual(float(logits.grad[0,1].abs().sum()),0.)
        self.assertGreater(float(logits.grad[0,0].abs().sum()),0.)

    def test_unknown_transferred_point_has_no_negative_visibility_gradient(self):
        from tools.vision_qc.heatmap_pose import supervised_loss
        torch=self.torch
        logits=torch.randn((1,2,20,20),requires_grad=True)
        visibility=torch.zeros((1,2),requires_grad=True)
        points=torch.tensor([[[10.,10.,2.],[0.,0.,0.]]])
        supervised_loss(logits,visibility,points,[True]).backward()
        self.assertEqual(float(visibility.grad[0,1]),0.)
        self.assertLess(float(visibility.grad[0,0]),0.)
        self.assertEqual(float(logits.grad[0,1].abs().sum()),0.)

    def test_checkpoint_reload_preserves_predictions_and_checks_encoder(self):
        from tools.vision_qc.heatmap_pose import make_head
        from tools.vision_qc.infer_heatmap import load_head
        torch=self.torch;torch.manual_seed(14);head=make_head().eval()
        features=[torch.randn(1,c,40//(2**i),40//(2**i)) for i,c in enumerate([64,128,256])]
        rgb=torch.rand(1,3,320,320)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.pt';torch.save({'head':head.state_dict(),'encoder_sha256':'test-encoder'},path)
            restored=load_head(path,'test-encoder')
            with torch.no_grad():
                a=head(features,rgb);b=restored(features,rgb)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
            with self.assertRaises(ValueError):load_head(path,'wrong-encoder')


if __name__=='__main__':unittest.main()
