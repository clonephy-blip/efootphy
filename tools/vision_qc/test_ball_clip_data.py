import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from .ball_clip_data import frame_span, validate_ball_review, validate_clip_plan
from .score_ball_frames import score_predictions, score_tracking_records
from .video_ball_crops import build_video_crops


def example():
    data=b'reviewed image bytes'
    plan={'sources':[{'video_id':'a','sha256':'a'*64,'split':'train'},
                     {'video_id':'b','sha256':'b'*64,'split':'val'}],
          'clips':[{'clip_id':'train','video_id':'a','start_ms':0,'end_ms':100},
                   {'clip_id':'val','video_id':'b','start_ms':0,'end_ms':100}]}
    row={'id':'a-2','video_id':'a','clip_id':'train','source_video_sha256':'a'*64,
         'source_frame_index':2,'pts_ms':30.,'image_size_wh':[800,700],
         'image_sha256':hashlib.sha256(data).hexdigest(),'reviewed':True,
         'reviewed_by':'assistant','geometry_reviewed':True,'visibility':'visible',
         'ball_box_xyxy_px':[390,340,410,360],
         'causal_context_frame_indices':[0,1,2],'causal_context_pts_ms':[10.,20.,30.]}
    return plan,row,data


class SourceReviewTests(unittest.TestCase):
    def test_vfr_span_is_half_open_without_rounding_to_fps(self):
        self.assertEqual(frame_span([0.,10.,26.5,49.],10.,49.),(1,3))

    def test_duplicate_pts_are_invalid(self):
        with self.assertRaises(ValueError):frame_span([0.,0.],0,1)

    def test_same_source_bytes_cannot_cross_splits(self):
        p,_,_=example();p['sources'][1]['sha256']='a'*64
        with self.assertRaises(ValueError):validate_clip_plan(p)

    def test_overlapping_clips_are_invalid(self):
        p,_,_=example();p['clips'].append({'clip_id':'overlap','video_id':'a','start_ms':50,'end_ms':150})
        with self.assertRaises(ValueError):validate_clip_plan(p)

    def test_reviewed_visible_geometry_and_bytes_are_required(self):
        p,r,data=example();self.assertTrue(validate_ball_review(r,data,p))
        with self.assertRaises(ValueError):validate_ball_review(r,data+b'changed',p)
        r['geometry_reviewed']=False
        with self.assertRaises(ValueError):validate_ball_review(r,data,p)

    def test_automatic_pseudolabel_is_rejected(self):
        p,r,data=example();r['reviewed']=False
        with self.assertRaises(ValueError):validate_ball_review(r,data,p)

    def test_unknown_ball_is_not_a_negative_training_label(self):
        p,r,data=example();r.update(visibility='occluded',ball_box_xyxy_px=None)
        self.assertFalse(validate_ball_review(r,data,p))

    def test_absent_ball_has_no_invented_coordinate(self):
        p,r,data=example();r['visibility']='absent'
        with self.assertRaises(ValueError):validate_ball_review(r,data,p)
        r['ball_box_xyxy_px']=None;self.assertTrue(validate_ball_review(r,data,p))

    def test_future_context_is_rejected(self):
        p,r,data=example();r['causal_context_frame_indices']=[1,2,3]
        with self.assertRaises(ValueError):validate_ball_review(r,data,p)

    def test_context_cannot_cross_clip_boundary(self):
        p,r,data=example();r['causal_context_pts_ms'][0]=-1
        with self.assertRaises(ValueError):validate_ball_review(r,data,p)


class FrozenScoringTests(unittest.TestCase):
    def test_missing_ball_stays_in_recall_denominator(self):
        rows=[{'visibility':'visible','reference_xy_px':[10,10],
               'detections':[{'xy_px':[11,10],'score':.8}]},
              {'visibility':'visible','reference_xy_px':[10,10],'detections':[]}]
        s=score_predictions(rows);self.assertEqual(s['top1_recall_on_reviewed_visible_frames'],.5)

    def test_lower_rank_correct_ball_does_not_fix_top1_failure(self):
        s=score_predictions([{'visibility':'visible','reference_xy_px':[0,0],
            'detections':[{'xy_px':[100,100],'score':.9},{'xy_px':[0,0],'score':.8}]}])
        self.assertEqual(s['top1_correct_within_tolerance'],0)
        self.assertEqual(s['any_candidate_correct_within_tolerance'],1)

    def test_absent_false_positive_and_unknown_are_separate(self):
        ds=[{'xy_px':[5,5],'score':.7}]
        s=score_predictions([{'visibility':'absent','detections':ds},
                             {'visibility':'uncertain','detections':ds}])
        self.assertEqual(s['negative_frames_with_false_detection'],1)
        self.assertEqual(s['ignored_frames'],1);self.assertIsNone(s['top1_recall_on_reviewed_visible_frames'])

    def test_last_missed_frame_contributes_its_real_duration(self):
        r={'pts_ms':100.,'frame_end_pts_ms':126.5,'visibility':'visible',
           'reference_xy_px':[5,5],'ball':{'state':'LOST','xy_px':None}}
        self.assertEqual(score_tracking_records([r])['longest_visible_missing_position_interval_ms'],26.5)

    def test_prediction_is_not_counted_as_observation(self):
        r={'pts_ms':0.,'frame_end_pts_ms':10.,'visibility':'visible',
           'reference_xy_px':[5,5],'ball':{'state':'PREDICTED','xy_px':[5,5]}}
        s=score_tracking_records([r]);self.assertEqual(s['correct_observed_within_tolerance'],0)
        self.assertEqual(s['correct_position_including_short_prediction'],1)

    def test_unknown_reference_breaks_a_missing_interval(self):
        rows=[{'pts_ms':t,'frame_end_pts_ms':t+10.,'visibility':v,
               'reference_xy_px':[5,5] if v=='visible' else None,
               'ball':{'state':'LOST','xy_px':None}}
              for t,v in [(0.,'visible'),(10.,'uncertain'),(20.,'visible')]]
        self.assertEqual(score_tracking_records(rows)['longest_visible_missing_position_interval_ms'],10.)

    def test_missing_source_interval_is_not_assumed_contiguous(self):
        rows=[{'pts_ms':t,'frame_end_pts_ms':t+10.,'visibility':'visible',
               'reference_xy_px':[5,5],'ball':{'state':'LOST','xy_px':None}} for t in [0.,100.]]
        self.assertEqual(score_tracking_records(rows)['longest_visible_missing_position_interval_ms'],10.)

    def test_invalid_reference_intervals_are_rejected(self):
        r={'pts_ms':10.,'frame_end_pts_ms':10.}
        with self.assertRaises(ValueError):score_tracking_records([r])


class CropExportTests(unittest.TestCase):
    def inputs(self,root):
        p,r,_=example();rows=[]
        for i,(identity,clip) in enumerate([('a','train'),('b','val')]):
            frame=np.full((700,800,3),50+i*20,np.uint8);path=root/(identity+'.png')
            cv2.imwrite(str(path),frame);row=copy.deepcopy(r)
            row.update(id=identity+'-2',video_id=identity,clip_id=clip,
                       source_video_sha256=identity*64,image_path=str(path),
                       image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
        plan=root/'plan.json';review=root/'review.json'
        plan.write_text(json.dumps(p));review.write_text(json.dumps(rows))
        return plan,review,rows

    def test_validation_crop_origins_do_not_depend_on_ball_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,review,rows=self.inputs(root)
            a=build_video_crops(plan,review,root/'one')
            rows[1]['ball_box_xyxy_px']=[20,20,40,40];review.write_text(json.dumps(rows))
            b=build_video_crops(plan,review,root/'two')
            for report in [a,b]:self.assertFalse(report['validation_crop_selection_uses_labels'])
            self.assertEqual([c['origin_xy'] for c in a['crops'] if c['split']=='valid'],
                             [c['origin_xy'] for c in b['crops'] if c['split']=='valid'])

    def test_duplicate_source_frame_is_not_imported_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,review,rows=self.inputs(root)
            duplicate=copy.deepcopy(rows[0]);duplicate['id']='different-id'
            review.write_text(json.dumps(rows+[duplicate]))
            with self.assertRaises(ValueError):build_video_crops(plan,review,root/'out')
