"""Exercise the notebook's actual packaging cell with real synthetic ONNX graphs."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import nbformat
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


NOTEBOOK = Path(__file__).resolve().parents[2] / 'notebooks' / 'efootball_ai_colab_training.ipynb'
DOCUMENT = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
CELLS = [''.join(cell['source']) for cell in DOCUMENT['cells']]
HELPERS = next(''.join(c['source']) for c in DOCUMENT['cells']
               if 'minimap-bundle-helpers' in c.get('metadata', {}).get('tags', []))
TRAIN = next(c for c in CELLS if 'minimap_best_pt =' in c)
PACKAGE = next(c for c in CELLS if 'stamp = datetime.now' in c)


def graph_bytes(names=None, input_shape=(1, 3, 320, 320), output_shape=(1, 7, 2100),
                dtype=TensorProto.FLOAT, task='detect', metadata=True, external=False,
                extra_output=False, broken=False):
    names = ['ball', 'controlled', 'dot'] if names is None else names
    input_info = helper.make_tensor_value_info('images', TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info('output0', dtype, output_shape)
    array = np.zeros(output_shape, dtype=np.float16 if dtype == TensorProto.FLOAT16 else np.float32)
    node = helper.make_node('Constant', [], ['output0'], value=numpy_helper.from_array(array))
    graph = helper.make_graph([node], 'synthetic-contract-only', [input_info], [output_info])
    if extra_output:
        graph.output.append(helper.make_tensor_value_info('images', TensorProto.FLOAT, input_shape))
    if broken:
        graph.node[0].output[0] = 'different_output'
    if external:
        tensor = TensorProto(name='external_weight', data_type=TensorProto.FLOAT, dims=[1])
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key='location', value='missing.bin')
        graph.initializer.append(tensor)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
    model.ir_version = 10
    if metadata:
        helper.set_model_props(model, {'names': repr(dict(enumerate(names))), 'task': task})
    return model.SerializeToString()


class ColabMinimapBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ns = {}
        exec(compile(HELPERS, str(NOTEBOOK), 'exec'), self.ns)
        self.validate = self.ns['validate_minimap_bytes']
        self.build = self.ns['build_minimap_bundle']

    def test_notebook_schema_and_code_syntax(self):
        nbformat.validate(nbformat.from_dict(DOCUMENT))
        for index, cell in enumerate(DOCUMENT['cells']):
            if cell['cell_type'] == 'code':
                # Shell/IPython magics belong to Colab; compile the surrounding Python.
                source = ''.join(cell['source'])
                clean = '\n'.join(line[:len(line)-len(line.lstrip())] + 'pass'
                                  if line.lstrip().startswith(('!', '%')) else line
                                  for line in source.splitlines())
                compile(clean, f'cell-{index}', 'exec')

    def test_valid_graph_and_manifest_match_apk_contract(self):
        raw = graph_bytes()
        manifest = self.validate(raw)
        self.assertEqual(manifest, {
            'schema_version': 1, 'profile': 'minimap', 'task': 'detect',
            'input_size': 320, 'input_layout': 'NCHW', 'output_layout': 'channels_first',
            'normalization': 'RGB/255', 'classes': ['ball', 'controlled', 'dot'],
            'model_sha256': hashlib.sha256(raw).hexdigest(),
        })

    def test_exact_zip_entries_and_roundtrip_hash(self):
        raw = graph_bytes()
        model = self.root / 'actual-export.onnx'; model.write_bytes(raw)
        output = self.root / 'exports' / 'bundle.zip'
        self.build(model, output)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.namelist(), ['model.onnx', 'manifest.json'])
            self.assertEqual(archive.read('model.onnx'), raw)
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['model_sha256'], hashlib.sha256(raw).hexdigest())

    def test_missing_model_is_not_replaced_with_field_model(self):
        (self.root / 'field.onnx').write_bytes(graph_bytes(names=['ball', 'player']))
        with self.assertRaises(FileNotFoundError):
            self.build(self.root / 'missing-minimap.onnx', self.root / 'bundle.zip')
        self.assertFalse((self.root / 'bundle.zip').exists())

    def test_field_seven_class_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(names=['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee'], output_shape=(1, 11, 2100)))

    def test_wrong_names_even_with_right_tensor_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(names=['ball', 'player', 'referee']))

    def test_wrong_class_order_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(names=['ball', 'dot', 'controlled']))

    def test_missing_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(metadata=False))

    def test_pose_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(task='pose'))

    def test_dynamic_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(input_shape=(1, 3, 'height', 320)))

    def test_wrong_input_size_and_batch_are_rejected(self):
        for shape in [(1,3,640,640), (2,3,320,320)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.validate(graph_bytes(input_shape=shape))

    def test_wrong_output_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(output_shape=(1, 7, 8400)))

    def test_fp16_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(dtype=TensorProto.FLOAT16))

    def test_multiple_outputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(extra_output=True))

    def test_external_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(graph_bytes(external=True))

    def test_invalid_graph_is_rejected_by_onnx_checker(self):
        with self.assertRaises(onnx.checker.ValidationError):
            self.validate(graph_bytes(broken=True))

    def test_empty_and_oversize_models_are_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(b'')
        self.ns['MAX_MODEL_BYTES'] = 10
        with self.assertRaises(ValueError):
            self.validate(graph_bytes())

    def test_untrusted_metadata_is_not_executed(self):
        model = onnx.load_model_from_string(graph_bytes())
        marker = self.root / 'should-not-exist'
        helper.set_model_props(model, {'task':'detect', 'names':f'__import__("pathlib").Path({str(marker)!r}).touch()'})
        with self.assertRaises(ValueError):
            self.validate(model.SerializeToString())
        self.assertFalse(marker.exists())

    def test_failed_validation_keeps_existing_zip_and_leaves_no_temporary_zip(self):
        model = self.root / 'invalid.onnx'; model.write_bytes(graph_bytes(names=['player']))
        output = self.root / 'existing.zip'; output.write_bytes(b'keep-existing')
        with self.assertRaises(ValueError):
            self.build(model, output)
        self.assertEqual(output.read_bytes(), b'keep-existing')
        self.assertEqual(list(self.root.glob('.minimap-*.zip')), [])

    def test_output_cannot_overwrite_source_model(self):
        model = self.root / 'model.onnx'; raw = graph_bytes(); model.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.build(model, model)
        self.assertEqual(model.read_bytes(), raw)

    def test_dataset_wrong_order_fails_before_training(self):
        data = self.root / 'data.yaml'
        data.write_text('nc: 3\nnames: [ball, dot, controlled]\n')
        with self.assertRaises(ValueError):
            self.ns['check_minimap_dataset'](data)

    def test_dataset_integer_and_string_class_ids(self):
        expected = ['ball','controlled','dot']
        for names in [expected, dict(enumerate(expected)), {str(i):v for i,v in enumerate(expected)}]:
            self.assertEqual(self.ns['require_minimap_names'](names), expected)
        for invalid in [{1:'ball',2:'controlled',3:'dot'}, {'00':'ball','1':'controlled','2':'dot'}]:
            with self.assertRaises(ValueError):
                self.ns['require_minimap_names'](invalid)

    def _prepare_training(self, fails=False):
        (self.root / 'data.yaml').write_text('nc: 3\nnames: [ball, controlled, dot]\n')
        best = self.root / 'run-renamed' / 'weights' / 'best.pt'
        best.parent.mkdir(parents=True); best.write_bytes(b'fake-pt-no-load')
        exported = self.root / 'returned-export-path.onnx'; exported.write_bytes(graph_bytes())
        train = Mock()
        if fails: train.train.side_effect = RuntimeError('training failed')
        else: train.train.return_value = types.SimpleNamespace(save_dir=best.parent.parent)
        exporter = Mock(names={0:'ball',1:'controlled',2:'dot'})
        exporter.export.return_value = str(exported)
        module = types.ModuleType('ultralytics'); module.YOLO = Mock(side_effect=[train, exporter])
        self.ns.update(minimap_dir=self.root, minimap_onnx_path=self.root/'stale.onnx')
        return module, exported

    def test_actual_export_path_is_used_after_training(self):
        module, exported = self._prepare_training()
        with patch.dict('sys.modules', {'ultralytics':module}):
            exec(TRAIN, self.ns)
        self.assertEqual(self.ns['minimap_onnx_path'], exported.resolve())

    def test_failed_training_clears_stale_export_reference(self):
        module, _ = self._prepare_training(fails=True)
        with patch.dict('sys.modules', {'ultralytics':module}), self.assertRaises(RuntimeError):
            exec(TRAIN, self.ns)
        self.assertIsNone(self.ns['minimap_onnx_path'])

    def test_restart_clears_old_export_even_when_drive_mount_fails(self):
        colab = types.ModuleType('google.colab'); colab.drive = Mock()
        colab.drive.mount.side_effect = RuntimeError('mount failed')
        self.ns['minimap_onnx_path'] = self.root / 'previous-run.onnx'
        self.ns['minimap_bundle_path'] = self.root / 'previous-run.zip'
        start_cell = next(c for c in CELLS if "drive.mount('/content/drive')" in c)
        with patch.dict('sys.modules', {'google.colab':colab}), self.assertRaises(RuntimeError):
            exec(start_cell, self.ns)
        self.assertIsNone(self.ns['minimap_onnx_path'])
        self.assertIsNone(self.ns['minimap_bundle_path'])

    def test_packaging_without_this_run_export_does_not_download(self):
        colab = types.ModuleType('google.colab'); colab.files = Mock()
        with patch.dict('sys.modules', {'google.colab':colab}), self.assertRaises(RuntimeError):
            exec(PACKAGE, self.ns)
        colab.files.download.assert_not_called()
        self.assertIsNone(self.ns['minimap_bundle_path'])


if __name__ == '__main__':
    unittest.main()
