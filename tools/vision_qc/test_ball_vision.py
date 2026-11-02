import unittest

import cv2
import numpy as np

from .ball_crops import crop_labels, grid_origins
from .ball_video import decode_ball
from .ball_vision import (BallObservation, CausalBallTrack, associate_control_markers,
                          ball_flow_observation,control_marker_candidates, patch_batch)


class BallTrackingTests(unittest.TestCase):
    def test_occlusion_is_prediction_and_expires_by_pts(self):
        track=CausalBallTrack()
        self.assertEqual(track.update(0,[BallObservation(10,20,.95)])['state'],'OBSERVED')
        self.assertEqual(track.update(17,[])['state'],'PREDICTED')
        state=track.update(101,[])
        self.assertEqual(state['state'],'LOST');self.assertIsNone(state['xy_px'])

    def test_low_confidence_cannot_create_a_track(self):
        track=CausalBallTrack()
        self.assertEqual(track.update(1,[BallObservation(10,20,.7)])['state'],'LOST')

    def test_flow_cannot_keep_an_unconfirmed_identity_alive_forever(self):
        track=CausalBallTrack(max_flow_without_detector_ms=80)
        track.update(0,[BallObservation(10,20,.95)])
        self.assertEqual(track.update(40,[BallObservation(10,20,.85,'optical_flow')])['state'],'OBSERVED')
        self.assertEqual(track.update(90,[BallObservation(10,20,.85,'optical_flow')])['state'],'PREDICTED')
        self.assertEqual(track.update(141,[BallObservation(10,20,.85,'optical_flow')])['state'],'LOST')

    def test_distant_bright_object_cannot_teleport_active_ball(self):
        track=CausalBallTrack()
        track.update(0,[BallObservation(10,20,.95)])
        state=track.update(16,[BallObservation(900,600,.999)])
        self.assertEqual(state['state'],'PREDICTED')
        self.assertEqual(state['xy_px'],[10,20])

    def test_scene_cut_does_not_extrapolate_previous_shot(self):
        track=CausalBallTrack()
        track.update(0,[BallObservation(10,20,.95)])
        state=track.update(20,[],scene_cut=True)
        self.assertEqual(state['state'],'LOST')

    def test_duplicate_and_invalid_times_or_observations_are_rejected(self):
        track=CausalBallTrack();track.update(17,[])
        with self.assertRaises(ValueError):track.update(17,[])
        with self.assertRaises(ValueError):track.update(float('nan'),[])
        with self.assertRaises(ValueError):track.update(30,[BallObservation(0,0,1.2)])

    def test_variable_frame_spacing_uses_seconds_for_velocity(self):
        track=CausalBallTrack()
        track.update(0,[BallObservation(100,100,.95)])
        track.update(40,[BallObservation(120,100,.95)])
        # 500 px/s measured, damped to 250 px/s, then 20 ms prediction.
        np.testing.assert_allclose(track.update(60,[])['xy_px'],[125,100])


class BallDecodeTests(unittest.TestCase):
    def test_ball_channel_cannot_be_replaced_by_player_confidence(self):
        raw=np.zeros((1,11,2100),np.float32)
        raw[0,:4,0]=[160,160,10,10];raw[0,8,0]=.99
        self.assertEqual(decode_ball(raw,(0,0,1,320,320)),[])

    def test_native_crop_coordinates_restore_into_full_frame(self):
        raw=np.zeros((1,11,2100),np.float32)
        raw[0,:4,:2]=np.array([[100,101],[50,50],[12,12],[12,12]])
        raw[0,4,:2]=[.8,.6]
        result=decode_ball(raw,(0,0,1,320,320),(600,200))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['xy_px'],[700,250])

    def test_padding_and_nonfinite_outputs_are_rejected(self):
        raw=np.zeros((1,11,2100),np.float32)
        raw[0,:4,0]=[10,5,10,10];raw[0,4,0]=.8
        self.assertEqual(decode_ball(raw,(0,80,1,320,160)),[])
        raw[0,0,0]=np.nan
        with self.assertRaises(ValueError):decode_ball(raw,(0,0,1,320,320))

    def test_anisotropic_source_adapter_restores_both_pixel_axes(self):
        raw=np.zeros((1,11,2100),np.float32)
        raw[0,:4,0]=[100,80,10,20];raw[0,4,0]=.8
        d=decode_ball(raw,(0,0,.5,1.,640,320),(600,200))[0]
        self.assertEqual(d['xy_px'],[800,280]);self.assertEqual(d['box_wh_px'],[20,20])

    def test_visual_flow_tracks_translation_and_rejects_blank_patch(self):
        frame=np.zeros((160,200),np.uint8)
        cv2.circle(frame,(70,80),8,255,-1);cv2.line(frame,(66,76),(73,84),80,2)
        moved=cv2.warpAffine(frame,np.array([[1,0,6],[0,1,3]],np.float32),(200,160))
        observation=ball_flow_observation(frame,moved,[70,80])
        self.assertIsNotNone(observation);self.assertEqual(observation.source,'optical_flow')
        np.testing.assert_allclose([observation.x,observation.y],[76,83],atol=.5)
        self.assertIsNone(ball_flow_observation(np.zeros_like(frame),np.zeros_like(frame),[70,80]))


class CropAndControlTests(unittest.TestCase):
    def test_validation_grid_covers_last_pixels_without_label_input(self):
        self.assertEqual(grid_origins(700,320),[0,320,380])
        with self.assertRaises(ValueError):grid_origins(200,320)

    def test_crop_clips_boxes_and_excludes_outside_objects(self):
        result=crop_labels([[0,100,100,20,20],[4,800,800,30,80]],0,0,320)
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(list(map(float,result[0].split()[1:])),[.3125,.3125,.0625,.0625])

    def test_cyan_triangle_is_separate_from_roster_identity_and_possession(self):
        image=np.zeros((1080,1920,3),np.uint8)
        cv2.fillPoly(image,[np.array([[80,20],[100,20],[90,40]])],(255,255,0))
        cv2.rectangle(image,(20,100),(150,104),(255,255,0),-1)
        markers=control_marker_candidates(image)
        self.assertEqual(len(markers),1)
        self.assertIsNone(markers[0]['player_identity']);self.assertIsNone(markers[0]['possession'])
        body={'class_name':'player','box_xyxy_px':[80,41,100,80]}
        matched=associate_control_markers(markers,[body],1080)
        self.assertEqual(matched[0]['state'],'ASSOCIATED_UI_CONTROL_CANDIDATE')
        self.assertEqual(associate_control_markers(markers,[body,body],1080)[0]['state'],'AMBIGUOUS')

    def test_patch_rgb_order_and_edge_padding(self):
        frame=np.full((60,60,3),(0,0,255),np.uint8)
        patch=patch_batch(frame,[(0,0)])
        self.assertEqual(patch.shape,(1,3,48,48))
        self.assertEqual(float(patch[0,0].mean()),1.)
        self.assertEqual(float(patch[0,2].mean()),0.)
