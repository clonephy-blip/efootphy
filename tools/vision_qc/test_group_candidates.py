import tempfile
import unittest
from pathlib import Path

import yaml

from .group_candidates import infer_group,inspect


class GroupCandidateTest(unittest.TestCase):
    def test_roboflow_variants_keep_one_original_source_group(self):
        self.assertEqual(infer_group('m1418_480p_f014592_jpg.rf.abc'),'m1418')
        self.assertEqual(infer_group('m1418_480p_f099999_jpg.rf.xyz'),'m1418')

    def test_video_prefix_preserves_leading_underscores_in_source_id(self):
        self.assertEqual(infer_group('yt__RyaqwxUGPI_01085s_jpg.rf.abc'),'yt__RyaqwxUGPI')
        self.assertIsNone(infer_group('Captura-de-pantalla-2026-01-29-080209_png.rf.abc'))

    def test_disjoint_filename_hypotheses_never_certify_an_independent_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,source in [('train','m1'),('valid','m2'),('test','m3')]:
                folder=root/split/'images';folder.mkdir(parents=True)
                (folder/(source+'_480p_f000001_jpg.rf.abc.jpg')).touch()
            data=root/'data.yaml';data.write_text(yaml.safe_dump({'train':'train/images','val':'valid/images','test':'test/images'}))
            report=inspect(data)
            self.assertEqual(report['recognized_group_count'],3)
            self.assertEqual(report['groups_crossing_splits'],[])
            self.assertFalse(report['verified']);self.assertFalse(report['eligible_for_independent_evaluation'])
