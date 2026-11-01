import copy
import unittest

from .group_candidates import infer_group
from .split_candidates import plan_split


def fixture():
    images=[]
    for i in range(1,7):
        name=f'train/images/m{i}_480p_f000001_jpg.rf.a.jpg'
        images.append({'image':name,'split':'train','pixel_sha256':f'pixel{i}',
                       'sha256':f'image{i}','label_sha256':f'label{i}'})
    audit={'task':'detect','data_valid':True,'images':images,'dataset_fingerprint':'fixture'}
    groups={'records':[{'image':x['image'],'inferred_group':infer_group(x['image'].split('/')[-1].removesuffix('.jpg'))} for x in images]}
    return audit,groups


class SplitCandidateTest(unittest.TestCase):
    def test_input_order_does_not_change_assignments_and_inputs_are_unchanged(self):
        audit,groups=fixture();before=copy.deepcopy(audit)
        one=plan_split(audit,groups);audit['images'].reverse();groups['records'].reverse()
        two=plan_split(audit,groups)
        self.assertEqual(one['plan_fingerprint'],two['plan_fingerprint'])
        self.assertEqual(sorted(before['images'],key=lambda x:x['image']),sorted(audit['images'],key=lambda x:x['image']))

    def test_exact_pixels_merge_different_filename_groups(self):
        audit,groups=fixture();audit['images'][1]['pixel_sha256']=audit['images'][0]['pixel_sha256']
        plan=plan_split(audit,groups)
        records={x['image']:x['candidate_split'] for x in plan['records']}
        self.assertEqual(records[audit['images'][0]['image']],records[audit['images'][1]['image']])
        self.assertTrue(any(x['groups']==['m1','m2'] for x in plan['clusters']))

    def test_unknown_pixel_duplicate_quarantines_the_whole_known_group(self):
        audit,groups=fixture();known=audit['images'][0]
        second={**known,'image':'train/images/m1_480p_f000002_jpg.rf.b.jpg','pixel_sha256':'other-pixel'}
        unknown={**known,'image':'test/images/screenshot_png.rf.c.jpg','split':'test'}
        audit['images'] += [second,unknown]
        groups['records'] += [{'image':second['image'],'inferred_group':'m1'},{'image':unknown['image'],'inferred_group':None}]
        plan=plan_split(audit,groups);self.assertEqual(plan['quarantine_images'],3)
        records={x['image']:x['candidate_split'] for x in plan['records']}
        for x in [known,second,unknown]:self.assertEqual(records[x['image']],'quarantine')

    def test_candidate_plan_never_certifies_evaluation_or_reuses_seen_weights(self):
        plan=plan_split(*fixture(),forbidden_weights=['a'*64])
        self.assertFalse(plan['ready_for_training']);self.assertFalse(plan['eligible_for_independent_evaluation'])
        self.assertTrue(plan['checkpoint_policy']['requires_fresh_official_pretrained'])
        self.assertEqual(plan['checkpoint_policy']['forbidden_checkpoint_sha256'],['a'*64])

    def test_mismatched_metadata_is_rejected(self):
        audit,groups=fixture();groups['records'][0]['inferred_group']='different'
        with self.assertRaises(ValueError):plan_split(audit,groups)
