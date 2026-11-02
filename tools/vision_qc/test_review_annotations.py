import copy
import unittest

from .review_annotations import COORDINATE_SPACE,review,validate


def fixtures():
    p={'physical_name_vi':'junction','candidate_id':1,'visibility':'visible','xy_px':[20,30],'estimated_uncertainty_px':2}
    frame={'frame_id':'mobile-00','width_px':100,'height_px':80,'points':[p]}
    payload={'schema_version':1,'status':'automatic_review_proposals_not_ground_truth','coordinate_space':COORDINATE_SPACE,'frames':[frame]}
    manifest={'frames':[{'frame_id':'mobile-00','width_px':100,'height_px':80,'sha256':'reference','file':'frame.png'}]}
    reference={'frames':[{'index':0,'source_sha256':'reference','image_size':[100,80],'points':[
        {'id':1,'status':'manual_reference','manual_xy_px':[20,30]},
        {'id':2,'status':'manual_reference','manual_xy_px':[50,40]},
        {'id':3,'status':'geometry_inferred','manual_xy_px':[80,50]}]}]}
    return payload,manifest,reference


class ExternalReviewTests(unittest.TestCase):
    def test_missing_points_stay_in_denominator_and_inferred_are_not_gt(self):
        p,m,r=fixtures();s=review(p,m,r)['summary']
        self.assertEqual(s['reviewed_reference_points'],2);self.assertEqual(s['missing_or_unobserved'],1);self.assertEqual(s['pck']['5'],.5)

    def test_dimension_and_reference_hash_mismatch_fail_closed(self):
        p,m,r=fixtures();p['frames'][0]['width_px']=50
        with self.assertRaises(ValueError):review(p,m,r)
        p,m,r=fixtures();r['frames'][0]['source_sha256']='other'
        with self.assertRaises(ValueError):review(p,m,r)
        p,m,r=fixtures();p['schema_version']=True
        with self.assertRaises(ValueError):review(p,m,r)

    def test_duplicate_frames_and_candidate_ids_are_rejected(self):
        p,m,r=fixtures();p['frames']*=2
        with self.assertRaises(ValueError):review(p,m,r)
        p,m,r=fixtures();p['frames'][0]['points']*=2
        with self.assertRaises(ValueError):review(p,m,r)

    def test_unobserved_nonfinite_and_outside_coordinates_are_rejected(self):
        for visibility,xy in [('outside',[20,30]),('visible',[float('nan'),30]),('visible',[100,30])]:
            p,m,r=fixtures();p['frames'][0]['points'][0].update(visibility=visibility,xy_px=xy)
            with self.assertRaises(ValueError):review(p,m,r)

    def test_unmapped_auxiliary_point_is_not_assigned_an_arbitrary_model_id(self):
        p,m,r=fixtures();q=copy.deepcopy(p['frames'][0]['points'][0]);q['candidate_id']=None;p['frames'][0]['points'].append(q)
        result=review(p,m,r);self.assertEqual(result['summary']['reviewed_reference_points'],2);self.assertFalse(result['automatic_training_import_allowed'])

    def test_self_reported_uncertainty_is_diagnostic_not_acceptance(self):
        p,m,r=fixtures();p['frames'][0]['points'][0]['xy_px']=[40,30]
        result=review(p,m,r);self.assertEqual(result['summary']['correct_within_px']['5'],0)
        self.assertEqual(result['declared_uncertainty_diagnostic']['within_declared_radius'],0)

    def test_input_proposals_are_never_mutated(self):
        p,m,r=fixtures();before=copy.deepcopy(p);review(p,m,r);self.assertEqual(p,before)

    def test_missing_frame_is_rejected(self):
        p,m,r=fixtures();p['frames']=[]
        with self.assertRaises(ValueError):review(p,m,r)

    def test_duplicate_reference_cannot_inflate_denominator(self):
        p,m,r=fixtures();r['frames'][0]['points'].append(copy.deepcopy(r['frames'][0]['points'][0]))
        with self.assertRaises(ValueError):review(p,m,r)


if __name__=='__main__':unittest.main()
