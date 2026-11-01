import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from .pose_sanity import checked_samples


class PoseSanityInputTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.relative='train/images/frame.jpg'
        self.image=self.root/self.relative;self.image.parent.mkdir(parents=True)
        Image.new('RGB',(32,32),(50,100,10)).save(self.image)
        self.label=self.root/'train/labels/frame.txt';self.label.parent.mkdir()
        self.row=[0,.5,.5,1,1]+[v for i in range(32) for v in [.2+i*.01,.4,2 if i<4 else 0]]
        self.label.write_text(' '.join(map(str,self.row)))
        self.audit=self.root/'qc.json';self.write_audit()

    def write_audit(self,split='train'):
        self.audit.write_text(json.dumps({'task':'pose','kpt_shape':[32,3],'data_valid':True,
            'images':[{'image':self.relative,'split':split,'sha256':hashlib.sha256(self.image.read_bytes()).hexdigest(),
                       'label_sha256':hashlib.sha256(self.label.read_bytes()).hexdigest()}]}))

    def test_selected_source_bytes_are_preserved(self):
        before=self.label.read_bytes();items=checked_samples(self.root,self.audit,[self.relative])
        self.assertEqual(len(items),1);self.assertEqual(self.label.read_bytes(),before)

    def test_validation_images_cannot_be_used_for_sanity_training(self):
        self.write_audit('valid')
        with self.assertRaises(ValueError):checked_samples(self.root,self.audit,[self.relative])

    def test_changed_label_is_not_silently_used(self):
        self.label.write_text('modified')
        with self.assertRaises(ValueError):checked_samples(self.root,self.audit,[self.relative])

    def test_invalid_points_fail_even_when_report_claims_valid(self):
        self.row[5]=2;self.label.write_text(' '.join(map(str,self.row)));self.write_audit()
        with self.assertRaises(ValueError):checked_samples(self.root,self.audit,[self.relative])

    def test_duplicate_or_external_selections_fail(self):
        for selection in [[self.relative,self.relative],['../external.jpg'],[]]:
            with self.assertRaises(ValueError):checked_samples(self.root,self.audit,selection)
