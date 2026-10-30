import hashlib
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from .dataset_qc import audit
from .normalize import normalize_dataset
from .train import preflight


class DatasetQcTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'source'
        self.root.mkdir()

    def fixture(self, pose=False, test=True):
        config = {'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
                  'nc': 1 if pose else 3, 'names': ['pitch'] if pose else ['ball','controlled','dot']}
        if pose: config['kpt_shape'] = [32, 3]
        (self.root/'data.yaml').write_text(yaml.safe_dump(config))
        groups = {}
        for i, split in enumerate(['train','valid','test'] if test else ['train','valid']):
            images = self.root/split/'images'; images.mkdir(parents=True)
            labels = self.root/split/'labels'; labels.mkdir()
            rng = random.Random(i + 2026)
            image = Image.frombytes('RGB',(32,32),rng.randbytes(32*32*3))
            image.save(images/'frame.png')
            text = '0 .2 .2 .1 .1\n1 .5 .5 .1 .1\n2 .8 .8 .1 .1\n'
            if pose: text = '0 .5 .5 1 1 ' + ' '.join(f'{.1+j*.02} .4 2' for j in range(32))+'\n'
            (labels/'frame.txt').write_text(text)
            groups[f'{split}/images/frame.png'] = f'match-{i}'
        (self.root/'groups.json').write_text(json.dumps(groups))
        return self.root/'data.yaml'

    def codes(self, report): return {x['code'] for x in report['data_errors']}

    def test_full_fixture_checks_all_splits(self):
        data=self.fixture()
        result=audit(data,3,self.root/'groups.json')
        self.assertTrue(result['data_valid'])
        self.assertTrue(result['eligible_for_independent_evaluation'])
        self.assertEqual(result['split_counts'],{'train':1,'valid':1,'test':1})

    def test_incomplete_export_is_a_hard_failure(self):
        self.assertIn('incomplete_export',self.codes(audit(self.fixture(),256)))

    def test_missing_label_is_not_assumed_to_be_a_negative(self):
        data=self.fixture();(self.root/'train/labels/frame.txt').unlink()
        self.assertIn('missing_label',self.codes(audit(data)))

    def test_explicit_empty_label_is_allowed(self):
        data=self.fixture();(self.root/'train/labels/frame.txt').write_text('')
        self.assertTrue(audit(data)['data_valid'])

    def test_cross_split_duplicate_pixels_fail(self):
        data=self.fixture();shutil.copy2(self.root/'train/images/frame.png',self.root/'valid/images/frame.png')
        self.assertIn('duplicate_pixels_across_splits',self.codes(audit(data)))

    def test_video_group_leakage_fails_even_for_different_images(self):
        data=self.fixture();groups=json.loads((self.root/'groups.json').read_text())
        groups['valid/images/frame.png']=groups['train/images/frame.png']
        path=self.root/'groups.json';path.write_text(json.dumps(groups))
        self.assertIn('video_group_leakage',self.codes(audit(data,groups_file=path)))

    def test_nan_and_bad_class_are_rejected(self):
        data=self.fixture();label=self.root/'train/labels/frame.txt'
        label.write_text('0 nan .5 .1 .1\n10 .5 .5 .1 .1\n')
        self.assertTrue({'non_finite','class_out_of_range'} <= self.codes(audit(data)))

    def test_polygon_is_reported_and_converted_without_changing_source(self):
        data=self.fixture();label=self.root/'train/labels/frame.txt'
        label.write_text('2 .1 .2 .2 .2 .2 .3 .1 .3\n')
        before=label.read_bytes();result=audit(data)
        self.assertEqual(result['polygon_rows'],1)
        self.assertFalse(result['data_valid'])
        output=self.root.parent/'normalized';converted=normalize_dataset(self.root,output)
        self.assertEqual(converted['polygon_rows_converted'],1)
        self.assertEqual(label.read_bytes(),before)
        row=[float(x) for x in (output/'train/labels/frame.txt').read_text().split()]
        self.assertEqual(row,[2,.15,.25,.1,.1])
        self.assertTrue(audit(output/'data.yaml',3)['data_valid'])

    def test_normalize_refuses_to_silently_drop_unlabeled_images(self):
        self.fixture();(self.root/'train/labels/frame.txt').unlink()
        output=self.root.parent/'normalized'
        with self.assertRaises(ValueError):normalize_dataset(self.root,output)
        self.assertFalse(output.exists())

    def test_normalize_refuses_same_stem_with_different_image_extensions(self):
        self.fixture()
        with Image.open(self.root/'train/images/frame.png') as image:
            image.save(self.root/'train/images/frame.jpg')
        with self.assertRaises(ValueError):normalize_dataset(self.root,self.root.parent/'normalized')

    def test_pose_has_101_values_and_is_never_converted_as_polygon(self):
        data=self.fixture(pose=True)
        self.assertTrue(audit(data)['data_valid'])
        with self.assertRaises(ValueError):normalize_dataset(self.root,self.root.parent/'normalized')

    def test_invalid_pose_visibility_fails(self):
        data=self.fixture(pose=True);label=self.root/'train/labels/frame.txt'
        row=label.read_text().split();row[-1]='1.5';label.write_text(' '.join(row))
        self.assertIn('invalid_keypoint',self.codes(audit(data)))

    def test_landmark_profile_rejects_other_keypoint_schemas(self):
        data=self.fixture(pose=True)
        config=yaml.safe_load(data.read_text());config['kpt_shape']=[17,3]
        data.write_text(yaml.safe_dump(config))
        for label in self.root.glob('*/labels/*.txt'):
            label.write_text(' '.join(label.read_text().split()[:56]))
        self.assertTrue(audit(data)['data_valid'])
        with self.assertRaises(ValueError):preflight(data,3,None,'landmarks',True)

    def test_missing_test_and_video_groups_do_not_claim_independent_evaluation(self):
        result=audit(self.fixture(test=False))
        self.assertTrue(result['data_valid'])
        self.assertFalse(result['eligible_for_independent_evaluation'])

    def test_profile_does_not_silently_change_class_order(self):
        data=self.fixture();config=yaml.safe_load(data.read_text());config['names'].reverse();data.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):preflight(data,3,None,'minimap',True)

    def test_research_flag_cannot_bypass_bad_data(self):
        with self.assertRaises(ValueError):preflight(self.fixture(),256,None,'minimap',True)

    def test_nonresearch_training_requires_independent_test(self):
        data=self.fixture(test=False)
        with self.assertRaises(ValueError):preflight(data,2,None,'minimap',False)

    def test_orphan_labels_fail(self):
        data=self.fixture();(self.root/'train/labels/extra.txt').write_text('')
        self.assertIn('orphan_label',self.codes(audit(data)))

    def test_export_path_cannot_escape_dataset_root(self):
        data=self.fixture();config=yaml.safe_load(data.read_text());config['train']='../../outside/images';data.write_text(yaml.safe_dump(config))
        self.assertIn('missing_split',self.codes(audit(data)))


if __name__=='__main__':unittest.main()
