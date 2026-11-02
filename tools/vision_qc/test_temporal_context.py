import unittest

from .temporal_context import causal_window,validate_match_splits


class TemporalContextTests(unittest.TestCase):
    def test_vfr_window_never_reads_a_future_frame(self):
        times=[0.,18.,36.,61.,83.,101.,127.,151.,177.,200.]
        result=causal_window(times,9,count=4,step_ms=50,max_gap_ms=60)
        self.assertEqual(result,[2,4,6,9])
        for index,target in zip(result,[50,100,150,200]):self.assertLessEqual(times[index],target)

    def test_warmup_and_missing_interval_are_not_silently_padded(self):
        self.assertIsNone(causal_window([0,50,100],1,count=4))
        self.assertIsNone(causal_window([0,50,400,450],3,count=4,max_gap_ms=100))

    def test_window_cannot_bridge_a_scene_cut(self):
        times=list(range(0,1000,50))
        self.assertIsNone(causal_window(times,10,count=8,cut_indices=[7]))
        self.assertEqual(causal_window(times,15,count=8,cut_indices=[7]),list(range(8,16)))

    def test_bad_timestamps_or_budget_fail(self):
        for times in [[0,0,50],[0,float('nan'),100],[100,50,150]]:
            with self.assertRaises(ValueError):causal_window(times,2)
        with self.assertRaises(ValueError):causal_window([0,50],1,count=0)

    def test_match_and_video_hash_leakage_fail(self):
        videos=[{'video_id':str(i),'split':s,'match_id':str(i),'match_id_verified':True,'sha256':str(i)*64}
                for i,s in enumerate(['train','val','test'])]
        self.assertTrue(validate_match_splits(videos)['independent_evaluation_ready'])
        videos[1]['match_id']='0';self.assertIn('match_leakage',validate_match_splits(videos)['errors'])
        videos[1]['sha256']='0'*64;self.assertIn('duplicate_video_across_splits',validate_match_splits(videos)['errors'])

    def test_a_filename_does_not_verify_match_identity(self):
        r=validate_match_splits([{'video_id':'match1_part1','split':'unassigned','match_id':'match1',
            'match_id_verified':False,'sha256':'a'*64}])
        self.assertFalse(r['independent_evaluation_ready']);self.assertFalse(r['automatic_training_import_allowed'])
        self.assertIn('match_identity_unverified',r['errors'])
