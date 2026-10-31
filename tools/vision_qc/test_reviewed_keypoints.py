import hashlib
import tempfile
import unittest
from pathlib import Path

from .reviewed_keypoints import apply_reviewed


class ReviewedKeypointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.source=Path(self.tmp.name)/'source';self.source.mkdir()
        self.label=self.source/'train/labels/frame.txt';self.label.parent.mkdir(parents=True)
        row=[0,.5,.5,1,1]+[v for i in range(32) for v in [i/32,.5,2]]
        self.label.write_text(' '.join(map(str,row))+'\n')
        self.before=self.label.read_bytes();self.dest=Path(self.tmp.name)/'reviewed'

    def review(self,kind,ids):
        return {'corrections':[{'label':'train/labels/frame.txt',
            'sha256':hashlib.sha256(self.before).hexdigest(),
            'operations':[{'kind':kind,'ids':ids}],'reason':'human source-image review'}]}

    def test_swap_preserves_source_and_all_other_ids(self):
        report=apply_reviewed(self.source,self.dest,self.review('swap',[15,31]))
        row=list(map(float,(self.dest/'train/labels/frame.txt').read_text().split()))
        self.assertEqual(row[50:53],[31/32,.5,2]);self.assertEqual(row[98:101],[15/32,.5,2])
        self.assertEqual(row[5:50],list(map(float,self.before.split()))[5:50])
        self.assertEqual(self.label.read_bytes(),self.before)
        self.assertFalse(report['model_predictions_used'])

    def test_missing_point_is_zeroed(self):
        apply_reviewed(self.source,self.dest,self.review('missing',[5]))
        self.assertEqual(list(map(float,(self.dest/'train/labels/frame.txt').read_text().split()))[20:23],[0,0,0])

    def test_stale_review_rejected_before_creating_copy(self):
        r=self.review('missing',[5]);self.label.write_text('changed')
        with self.assertRaises(ValueError):apply_reviewed(self.source,self.dest,r)
        self.assertFalse(self.dest.exists())

    def test_external_paths_rejected(self):
        r=self.review('missing',[5]);r['corrections'][0]['label']='../../elsewhere.txt'
        with self.assertRaises(ValueError):apply_reviewed(self.source,self.dest,r)
        self.assertFalse(self.dest.exists())

    def test_bad_point_rejected(self):
        with self.assertRaises(ValueError):apply_reviewed(self.source,self.dest,self.review('missing',[32]))
        self.assertFalse(self.dest.exists())

    def test_output_cannot_replace_or_recurse_into_source(self):
        for destination in [self.source,self.source/'reviewed']:
            with self.assertRaises(ValueError):apply_reviewed(self.source,destination,self.review('missing',[5]))
