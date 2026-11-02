import unittest

import cv2
import numpy as np

from .pitch_tracking import project
from .semantic_lines import SemanticLineConfig,_solve_lines,calibrate_semantic_lines,observe_line_junctions,semantic_field_mask


class SemanticLineTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.frame=np.full((360,640,3),(35,110,40),np.uint8)
        self.segments=[{'id':str(i),'world_xy':line} for i,line in enumerate([
            [[40,50],[600,50]],[[40,310],[600,310]],[[70,35],[70,330]],
            [[570,35],[570,330]],[[320,35],[320,330]]])]
        for s in self.segments:cv2.polylines(self.frame,[np.asarray(s['world_xy'],np.int32)],False,(235,240,235),3)

    def test_semantic_fit_recovers_known_translation_amid_white_clutter(self):
        for x,y in [(175,130),(450,200),(210,265)]:cv2.rectangle(self.frame,(x,y),(x+12,y+32),(240,240,240),-1)
        seed=np.array([[1.,.001,7.],[-.001,1.,-5.],[0.,0.,1.]])
        result,m=calibrate_semantic_lines(self.frame,seed,self.segments)
        self.assertTrue(m['accepted'],m)
        points=np.array([[70,50],[570,310],[320,180]])
        self.assertLess(np.median(np.linalg.norm(project(result,points)-points,axis=1)),1.)
        self.assertFalse(m['models_validated']);self.assertFalse(m['automatic_training_import_allowed'])

    def test_line_solver_does_not_pair_unrelated_segment_endpoints(self):
        observations=[]
        true=np.array([[2.,.03,40.],[.01,1.5,30.],[.0001,.0002,1.]])
        for segment in self.segments:
            world=np.asarray(segment['world_xy']);xy=project(true,world)
            line=np.cross(np.r_[xy[0],1.],np.r_[xy[1],1.]);line/=np.linalg.norm(line[:2])
            observations.append({'world_xy':world.tolist(),'line_abc':line.tolist()})
        H,rank=_solve_lines(observations,(1280,720))
        np.testing.assert_allclose(project(H,[[70,90],[500,270]]),project(true,[[70,90],[500,270]]),atol=1e-6)
        self.assertGreater(rank,1e-5)

    def test_parallel_lines_cannot_determine_full_calibration(self):
        observations=[{'world_xy':[[0,y],[600,y]],'line_abc':[0,1,-y]} for y in [40,100,180,250]]
        with self.assertRaises(ValueError):_solve_lines(observations,(640,360))

    def test_hud_paint_and_blank_input_do_not_create_calibration(self):
        for frame,hud in [(self.frame,{'rects':[[0,0,640,360]]}),(np.full_like(self.frame,(35,110,40)),{})]:
            result,m=calibrate_semantic_lines(frame,np.eye(3),self.segments,hud_spec=hud)
            np.testing.assert_array_equal(result,np.eye(3));self.assertFalse(m['accepted'])

    def test_proposals_cannot_replace_missing_pixel_evidence(self):
        frame=np.full_like(self.frame,(35,110,40))
        proposal={'world_xy':[70,50],'image_xy':[100,100],'confidence':1.}
        _,m=calibrate_semantic_lines(frame,np.eye(3),self.segments,[proposal])
        self.assertFalse(m['accepted'])

    def test_inputs_are_immutable_and_invalid_seed_or_budgets_fail(self):
        original=self.frame.copy();seed=np.eye(3)
        calibrate_semantic_lines(self.frame,seed,self.segments)
        np.testing.assert_array_equal(self.frame,original);np.testing.assert_array_equal(seed,np.eye(3))
        with self.assertRaises(ValueError):calibrate_semantic_lines(self.frame,np.zeros((3,3)),self.segments)
        with self.assertRaises(ValueError):SemanticLineConfig(band_px=float('nan'))
        with self.assertRaises(ValueError):SemanticLineConfig(ransac_trials=1.5)

    def test_junction_observation_requires_two_supported_lines_and_unmasked_location(self):
        segments=[{'id':'vertical','world_xy':[[70,0],[70,310]]},
                  {'id':'horizontal','world_xy':[[0,50],[600,50]]}]
        evidence=[{'id':'vertical','accepted':True,'line_abc':[1,0,-72],
                   'paint_endpoints_px':[[72,30],[72,300]],'paint_residual_p95_px':1.2},
                  {'id':'horizontal','accepted':True,'line_abc':[0,1,-51],
                   'paint_endpoints_px':[[50,51],[500,51]],'paint_residual_p95_px':1.1}]
        points=[{'id':0,'world_xy':[70,50]},{'id':1,'world_xy':[70,180]}]
        result=observe_line_junctions(points,segments,evidence,(640,360))
        self.assertEqual(len(result),1);np.testing.assert_allclose(result[0]['image_xy'],[72,51])
        self.assertFalse(result[0]['global_calibration_usable'])
        self.assertFalse(result[0]['automatic_training_import_allowed'])
        self.assertEqual(observe_line_junctions(points,segments,evidence,(640,360),{'rects':[[60,40,30,30]]}),[])
        self.assertEqual(observe_line_junctions(points,segments,evidence[:1],(640,360)),[])

    def test_junction_rejects_long_extrapolation_past_visible_paint(self):
        segments=[{'id':'v','world_xy':[[70,0],[70,310]]},{'id':'h','world_xy':[[0,50],[600,50]]}]
        evidence=[{'id':'v','accepted':True,'line_abc':[1,0,-70],
                   'paint_endpoints_px':[[70,150],[70,300]],'paint_residual_p95_px':1.},
                  {'id':'h','accepted':True,'line_abc':[0,1,-50],
                   'paint_endpoints_px':[[50,50],[500,50]],'paint_residual_p95_px':1.}]
        self.assertEqual(observe_line_junctions([{'id':0,'world_xy':[70,50]}],segments,evidence,(640,360)),[])

    def test_yellow_advertising_with_white_letters_is_not_pitch_paint(self):
        frame=np.full((240,640,3),(10,240,240),np.uint8)
        frame[80:]=(35,110,40)
        cv2.line(frame,(0,50),(639,50),(240,240,240),3)
        cv2.line(frame,(0,80),(639,80),(240,240,240),2)
        mask=semantic_field_mask(frame)
        self.assertEqual(int(mask[50,320]),0)
        self.assertEqual(int(mask[80,320]),255)


if __name__=='__main__':unittest.main()
