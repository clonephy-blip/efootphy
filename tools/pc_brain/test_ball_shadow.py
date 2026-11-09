import unittest
import numpy as np

from tools.pc_brain.ball_shadow import CameraCalibration, find_shadow_candidates, triangulate_ball_shadow


class BallShadowGeometryTest(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[900.,0,320],[0,900,240],[0,0,1]])
        self.center=np.array([0.,-20.,15.])
        forward=-self.center/np.linalg.norm(self.center)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        self.r=np.vstack([right,down,forward]);self.t=-self.r@self.center
        self.camera=CameraCalibration(self.k,self.r,self.t,1.,7)

    def project(self,p):
        q=self.k@(self.r@np.asarray(p)+self.t)
        return q[:2]/q[2]

    def case(self,light=(.4,.2,-1.),point=(4.,8.,3.)):
        point=np.asarray(point);light=np.asarray(light)
        shadow=point-point[2]/light[2]*light
        return point,self.project(point),self.project(shadow),light

    def test_recovers_known_3d_and_does_not_assume_shadow_is_ground_projection(self):
        point,ball,shadow,light=self.case()
        result=triangulate_ball_shadow(ball,shadow,self.camera,light,1.1,7)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.xyz_m,point,atol=1e-8)
        self.assertLess(result.ray_gap_m,1e-8)

    def test_vertical_light(self):
        point,ball,shadow,light=self.case(light=(0,0,-1))
        result=triangulate_ball_shadow(ball,shadow,self.camera,light,1.1,7)
        np.testing.assert_allclose(result.xyz_m,point,atol=1e-8)

    def test_unknown_camera_or_light_yields_no_height(self):
        _,ball,shadow,light=self.case()
        self.assertIsNone(triangulate_ball_shadow(ball,shadow,None,light,1.1,7))
        self.assertIsNone(triangulate_ball_shadow(ball,shadow,self.camera,None,1.1,7))

    def test_stale_or_changed_camera_yields_no_height(self):
        _,ball,shadow,light=self.case()
        for now,epoch in [(1.5,7),(.9,7),(1.1,8)]:
            self.assertIsNone(triangulate_ball_shadow(ball,shadow,self.camera,light,now,epoch))

    def test_upward_or_zero_light_is_rejected(self):
        _,ball,shadow,_=self.case()
        for light in [(0,0,0),(0,0,1),(1,0,0)]:
            self.assertIsNone(triangulate_ball_shadow(ball,shadow,self.camera,light,1.1,7))

    def test_negative_height_is_rejected(self):
        _,ball,shadow,light=self.case(point=(4,8,-3))
        self.assertIsNone(triangulate_ball_shadow(ball,shadow,self.camera,light,1.1,7))

    def test_nonfinite_input_is_rejected(self):
        _,_,shadow,light=self.case()
        self.assertIsNone(triangulate_ball_shadow([np.nan,10],shadow,self.camera,light,1.1,7))

    def test_singular_intrinsics_are_rejected(self):
        _,ball,shadow,light=self.case()
        cam=CameraCalibration(np.zeros((3,3)),self.r,self.t,1.,7)
        self.assertIsNone(triangulate_ball_shadow(ball,shadow,cam,light,1.1,7))


class ShadowProposalTest(unittest.TestCase):
    def frame(self):
        import cv2
        frame=np.empty((220,400,3),np.uint8);frame[:]=(40,140,40)
        cv2.ellipse(frame,(220,140),(7,4),0,0,360,(8,30,8),-1)
        return frame

    def test_finds_local_synthetic_shadow(self):
        proposals=find_shadow_candidates(self.frame(),(200,90),4)
        self.assertTrue(any(np.linalg.norm(np.array(p.xy_px)-[220,140])<3 for p in proposals))

    def test_no_ball_hypothesis_does_not_search_or_create_ball(self):
        self.assertEqual(find_shadow_candidates(self.frame(),None),[])

    def test_player_body_region_is_excluded(self):
        proposals=find_shadow_candidates(self.frame(),(200,90),4,player_boxes=[(200,120,240,160)])
        self.assertFalse(any(np.linalg.norm(np.array(p.xy_px)-[220,140])<3 for p in proposals))

    def test_black_transition_has_no_grass_shadow(self):
        self.assertEqual(find_shadow_candidates(np.zeros((220,400,3),np.uint8),(200,90)),[])

    def test_bright_text_or_control_marker_near_dark_spot_is_not_shadow_evidence(self):
        for color in ((255,255,255),(255,255,0)):
            with self.subTest(color=color):
                frame=self.frame()
                frame[140:142,219:222]=color
                proposals=find_shadow_candidates(frame,(200,90),4)
                self.assertFalse(any(np.linalg.norm(np.array(p.xy_px)-[220,140])<5 for p in proposals))

    def test_search_is_bounded(self):
        with self.assertRaises(ValueError):
            find_shadow_candidates(self.frame(),(200,90),search_radius_px=10000)

    def test_nonfinite_position_is_ignored(self):
        self.assertEqual(find_shadow_candidates(self.frame(),(np.inf,90)),[])


if __name__=='__main__':unittest.main()
