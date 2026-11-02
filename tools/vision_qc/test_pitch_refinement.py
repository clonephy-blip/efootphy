import unittest

import cv2
import numpy as np

from .pitch_refinement import RefinementConfig, refine_calibration
from .pitch_tracking import project


class PaintRefinementTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.frame=np.full((320,640,3),(35,110,40),np.uint8)
        self.lines=[[[35,45],[600,45]],[[55,275],[600,275]],
                    [[70,45],[70,275]],[[580,45],[580,275]],[[320,45],[320,275]]]
        for line in self.lines:
            cv2.polylines(self.frame,[np.asarray(line,np.int32)],False,(235,240,235),2)

    def test_known_small_camera_error_is_reduced_without_reference_input(self):
        H=np.array([[1.,.001,5.],[-.001,1.,-4.],[0,0,1.]])
        refined,m=refine_calibration(self.frame,H,self.lines)
        points=[[70,45],[580,275],[320,170]]
        old=np.mean(np.linalg.norm(project(H,points)-points,axis=1))
        new=np.mean(np.linalg.norm(project(refined,points)-points,axis=1))
        self.assertTrue(m['accepted'],m)
        self.assertLess(new,old*.5)
        self.assertFalse(m['models_validated'])
        self.assertFalse(m['automatic_training_import_allowed'])

    def test_no_paint_retains_seed_and_inputs_are_immutable(self):
        blank=np.full_like(self.frame,(35,110,40));H=np.eye(3);original=blank.copy()
        refined,m=refine_calibration(blank,H,self.lines)
        np.testing.assert_array_equal(refined,H);np.testing.assert_array_equal(blank,original)
        self.assertEqual(m['reason'],'insufficient_paint')

    def test_masked_hud_paint_cannot_be_used_as_evidence(self):
        H=np.eye(3)
        refined,m=refine_calibration(self.frame,H,self.lines,{'rects':[[0,0,640,320]]})
        np.testing.assert_array_equal(refined,H)
        self.assertFalse(m['accepted']);self.assertEqual(m['reason'],'insufficient_paint')

    def test_a_single_line_does_not_identify_a_homography(self):
        refined,m=refine_calibration(self.frame,np.eye(3),self.lines[:1])
        np.testing.assert_array_equal(refined,np.eye(3))
        self.assertEqual(m['reason'],'insufficient_distributed_evidence')

    def test_invalid_inputs_and_budgets_are_rejected(self):
        for H in [np.zeros((3,3)),np.full((3,3),np.nan),np.ones((2,2))]:
            with self.assertRaises(ValueError):refine_calibration(self.frame,H,self.lines)
        with self.assertRaises(ValueError):refine_calibration(self.frame[:,:,0],np.eye(3),self.lines)
        for kw in [{'max_width':0},{'spacing_px':float('nan')},{'max_evaluations':1.5}]:
            with self.assertRaises(ValueError):RefinementConfig(**kw)


if __name__=='__main__':unittest.main()
