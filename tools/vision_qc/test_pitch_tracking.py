import unittest

import cv2
import numpy as np

from .pitch_tracking import Observation,PitchTracker,SiftMatcher,XFeatMatcher,estimate_motion,field_mask,project
from .pnl_adapter import SOURCE_TO_PNL,map_source_points
from .benchmark_tracking import draw_panel
from .evaluate_teacher import score_calibrations


def texture():
    rng=np.random.default_rng(42);frame=np.full((320,640,3),(35,110,40),np.uint8)
    for x,y in rng.integers([20,20],[620,300],(200,2)):
        cv2.rectangle(frame,(int(x),int(y)),(int(x+4),int(y+4)),(220,240,225),-1)
    return frame


class MotionTest(unittest.TestCase):
    def setUp(self):cv2.setNumThreads(1)

    def test_ransac_recovers_distributed_motion_with_outliers(self):
        rng=np.random.default_rng(5);a=rng.uniform([20,20],[620,300],(100,2))
        true=np.array([[1.01,.003,4],[-.002,.99,2],[.00001,-.00002,1]])
        b=project(true,a);b[:20]=rng.uniform([0,0],[640,320],(20,2))
        H,metrics=estimate_motion(a,b,(640,320));self.assertIsNotNone(H)
        self.assertLess(np.median(np.linalg.norm(project(H,a[20:])-project(true,a[20:]),axis=1)),.01)
        self.assertGreaterEqual(metrics['inliers'],79)

    def test_rejects_clustered_and_collinear_points(self):
        rng=np.random.default_rng(2);a=rng.uniform(100,105,(30,2))
        self.assertIsNone(estimate_motion(a,a+1,(640,320))[0])
        a=np.c_[np.linspace(0,600,30),np.full(30,100)]
        self.assertIsNone(estimate_motion(a,a+1,(640,320))[0])

    def test_rejects_flip_and_invalid_matches(self):
        a=np.float32([[x,y] for x in np.linspace(20,620,8) for y in np.linspace(20,300,5)])
        b=a.copy();b[:,0]=640-b[:,0]
        self.assertIsNone(estimate_motion(a,b,(640,320))[0])
        a[0]=np.nan;self.assertIsNone(estimate_motion(a,b,(640,320))[0])

    def test_hud_is_removed_from_pitch_features(self):
        frame=texture();hud=np.full(frame.shape[:2],255,np.uint8);hud[120:220,200:350]=0
        mask=field_mask(frame,hud)
        self.assertEqual(mask[150,220],0);self.assertGreater(np.mean(mask>0),.5)

    def test_blank_and_unsupported_seeds_are_withheld(self):
        tracker=PitchTracker();self.assertEqual(tracker.seed(np.full((320,640,3),(35,110,40),np.uint8),np.eye(3),0).status,'LOST')
        tracker=PitchTracker(world_lines=[[[10,10],[600,10]],[[10,300],[600,300]]])
        self.assertEqual(tracker.seed(texture(),np.eye(3),0).status,'LOST')

    def test_lk_tracks_camera_translation(self):
        frame=texture();H=np.array([[1,0,3],[0,1,2],[0,0,1.]])
        tracker=PitchTracker();self.assertEqual(tracker.seed(frame,np.eye(3),0).status,'TRACKING')
        shifted=cv2.warpPerspective(frame,H,(640,320));obs=tracker.update(shifted,33)
        self.assertEqual(obs.status,'TRACKING');self.assertLess(np.linalg.norm(project(obs.homography,[[320,160]])-[[323,162]]),.5)

    def test_loss_returns_no_stale_homography_and_sift_relocalizes(self):
        frame=texture();tracker=PitchTracker(matcher=SiftMatcher());tracker.seed(frame,np.eye(3),0)
        obs=tracker.update(np.zeros_like(frame),33);self.assertEqual(obs.status,'LOST');self.assertIsNone(obs.homography)
        obs=tracker.update(frame,66);self.assertEqual(obs.status,'TRACKING');self.assertEqual(obs.source,'sift')

    def test_timestamp_gap_drops_lk_pose(self):
        frame=texture();tracker=PitchTracker();tracker.seed(frame,np.eye(3),0)
        obs=tracker.update(frame,251);self.assertEqual(obs.status,'LOST');self.assertIsNone(obs.homography)

    def test_bad_time_and_resolution_need_new_calibration(self):
        frame=texture();tracker=PitchTracker();tracker.seed(frame,np.eye(3),0)
        self.assertEqual(tracker.update(frame,0).metrics['reason'],'timestamp_not_monotonic')
        tracker.seed(frame,np.eye(3),0);obs=tracker.update(frame[:200],33)
        self.assertEqual(obs.metrics['reason'],'resolution_changed_needs_seed');self.assertIsNone(obs.homography)
        self.assertEqual(tracker.update(frame,66).metrics['reason'],'needs_seed')

    def test_invalid_seed_horizon_and_input_raise(self):
        tracker=PitchTracker()
        with self.assertRaises(ValueError):tracker.seed(texture(),np.zeros((3,3)),0)
        with self.assertRaises(ValueError):tracker.seed(np.zeros((32,32)),np.eye(3),0)
        with self.assertRaises(ValueError):project(np.zeros((3,3)),[[1,2]])

    def test_semantic_map_does_not_use_elevated_goal_posts(self):
        self.assertEqual(len(set(SOURCE_TO_PNL)),32)
        self.assertFalse(set(SOURCE_TO_PNL)&set(range(12,20)))
        raw=np.array([[i,i+1,.7] for i in range(1,58)])
        mapped=map_source_points(raw,1920,1080)
        np.testing.assert_allclose(mapped[0],[2,4,.7]);np.testing.assert_allclose(mapped[21],[114,116,.7])

    def test_mutual_matching_rejects_ambiguous_repeated_descriptors(self):
        matcher=XFeatMatcher.__new__(XFeatMatcher);xy=np.array([[i,i] for i in range(20)])
        descriptors=np.eye(20,dtype=np.float32);a,b=matcher.match((xy,descriptors),(xy+2,descriptors))
        self.assertEqual(len(a),20);np.testing.assert_array_equal(b,a+2)
        repeated=np.ones((20,20),np.float32)/np.sqrt(20)
        self.assertEqual(len(matcher.match((xy,repeated),(xy,repeated))[0]),0)

    def test_qc_overlay_preserves_control_pixels(self):
        frame=texture();obs=Observation('TRACKING',np.eye(3),'seed',0,{})
        panel=draw_panel(frame,obs,[[[50,80],[50,280]]],'test',None,width=640,hud_spec={'rects':[[40,100,30,100]]})
        np.testing.assert_array_equal(panel[100:200,40:70],frame[100:200,40:70])

    def test_geometry_scoring_does_not_count_inferred_points(self):
        geometry={'frames':[{'source_file':'frame.png','points':[
            {'status':'manual_reference','world_xy_m_assumed':[2,3],'manual_xy_px':[2,3]},
            {'status':'occluded_inferred','world_xy_m_assumed':[10,20],'manual_xy_px':[999,999]}]}]}
        result=score_calibrations(geometry,{'records':[{'image':'frame.png','H_world_to_image':np.eye(3).tolist()}]})
        self.assertEqual(result['manual_reference_count'],1);self.assertEqual(result['correct_within_5px'],1)
        result=score_calibrations(geometry,{'records':[]});self.assertEqual(result['correct_within_5px'],0)
        self.assertEqual(result['manual_reference_count'],1)


if __name__=='__main__':unittest.main()
