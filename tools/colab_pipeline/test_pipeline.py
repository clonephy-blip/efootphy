"""Synthetic lifecycle and data-contract tests. No cloud account or GPU is used."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import __main__ as controller
from . import worker


class SyntheticColab(controller.Controller):
    def __init__(self, *args, mode='success', **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = []
        self.mode = mode
        self.remote_archive = None

    def command(self, *args, input_text=None, interactive=False, check=True, timeout=None):
        self.commands.append(args)
        if args[0] == 'exec' and self.mode == 'probe_failure':
            raise RuntimeError('synthetic Drive probe failure')
        if args[0] == 'install' and self.mode == 'install_failure':
            raise RuntimeError('synthetic installation failure')
        if args[0] == 'exec' and '__EFOOTPHY_DRIVE__' in (input_text or ''):
            mounted = self.mode != 'drive_missing'
            return SimpleNamespace(stdout='__EFOOTPHY_DRIVE__=' + str(mounted), returncode=0)
        if args[0] == 'exec':
            compile(input_text, '<bootstrap>', 'exec')
            remote = self.folder / 'synthetic-remote'
            remote.mkdir(exist_ok=True)
            status = 'failed' if self.mode == 'kernel_error' else 'success'
            worker.write_json(remote / 'job.json', {'run_id': self.run_id,
                'config_sha256': worker.digest(self.config_file), 'status': status,
                'steps': [{'name': 'inventory', 'status': status}], 'runtime': {'gpu': 'synthetic'}})
            worker.write_json(remote / 'inventory' / 'result.json', {'status': status})
            worker.archive(remote, 1)
            self.remote_archive = remote / 'artifacts.zip'
            # A zero CLI exit status does not imply successful remote Python.
            return SimpleNamespace(stdout='kernel simulated', returncode=0)
        if args[0] == 'download':
            if self.mode == 'download_failure':
                raise RuntimeError('synthetic transfer failure')
            Path(args[4]).write_bytes(self.remote_archive.read_bytes())
        return SimpleNamespace(stdout='', returncode=0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'schema_version': 1, 'auth': 'oauth2', 'gpu': 'T4',
            'project_root': str(self.root / 'project'), 'output_root': str(self.root / 'output'),
            'max_minutes': 1, 'requirements': ['ultralytics'],
            'tasks': [{'name': 'inventory', 'kind': 'inventory'}]}
        Path(self.config['project_root']).mkdir()
        self.file = self.root / 'config.json'
        self.save_config()

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self):
        worker.write_json(self.file, self.config)

    def make_controller(self, mode='success'):
        folder = self.root / 'local'
        folder.mkdir(exist_ok=True)
        return SyntheticColab(self.config, folder, 'test-01', sys.executable, mode=mode)

    def test_plan_runs_without_cli_or_cloud(self):
        args = ['colab_pipeline', 'plan', '--config', str(self.file), '--colab', '/missing/colab']
        with patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(controller.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['cloud_calls'], 0)

    def test_supported_configuration(self):
        self.assertEqual(controller.read_config(self.file, None)['tasks'], self.config['tasks'])

    def test_unknown_gpu_is_rejected(self):
        self.config['gpu'] = 'typo'
        self.save_config()
        with self.assertRaises(ValueError):
            controller.read_config(self.file, None)

    def test_invalid_time_budget(self):
        for value in [True, 0, -1, float('nan')]:
            self.config['max_minutes'] = value
            # Exercise parser behavior for JSON NaN as well.
            self.file.write_text(json.dumps(self.config))
            with self.assertRaises(ValueError):
                controller.read_config(self.file, None)

    def test_placeholder_dataset_is_rejected_before_allocating(self):
        self.config['tasks'] = [{'name': 'pitch', 'kind': 'audit', 'data': 'EDIT_ME/data.yaml'}]
        self.save_config()
        with self.assertRaises(ValueError):
            controller.read_config(self.file, None)

    def test_dependency_must_precede_consumer(self):
        self.config['tasks'] = [{'name': 'export', 'kind': 'export', 'weights': '@ball', 'imgsz': 320},
                                {'name': 'ball', 'kind': 'train', 'prepare': 'ball_v2', 'weights': 'm.pt', 'imgsz': 320}]
        self.save_config()
        with self.assertRaises(ValueError):
            controller.read_config(self.file, None)

    def test_duplicate_task_selection(self):
        with self.assertRaises(ValueError):
            controller.read_config(self.file, 'inventory,inventory')

    def test_run_id_traversal_is_rejected(self):
        args = ['job', 'plan', '--run-id', '../../outside', '--config', str(self.file)]
        with patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(controller.main(), 2)

    def test_controller_collects_and_stops_owned_session(self):
        value = self.make_controller()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(value.run(), 0)
        self.assertEqual(value.state['phase'], 'stopped')
        self.assertTrue((value.folder / 'artifacts.zip').is_file())
        self.assertEqual(sum(args[0] == 'new' for args in value.commands), 1)
        self.assertEqual(sum(args[0] == 'stop' for args in value.commands), 1)

    def test_zero_cli_exit_does_not_hide_kernel_failure(self):
        value = self.make_controller('kernel_error')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(value.run(), 1)
        self.assertEqual(value.state['job_status'], 'failed')
        self.assertEqual(value.state['phase'], 'stopped')

    def test_install_failure_still_stops(self):
        value = self.make_controller('install_failure')
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            value.run()
        self.assertEqual(value.state['phase'], 'stopped')

    def test_probe_failure_still_stops_new_session(self):
        value = self.make_controller('probe_failure')
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            value.run()
        self.assertEqual(value.state['phase'], 'stopped')

    def test_transfer_failure_still_stops(self):
        value = self.make_controller('download_failure')
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            value.run()
        self.assertEqual(value.state['phase'], 'stopped')

    def test_awaiting_drive_does_not_allocate_twice(self):
        value = self.make_controller('drive_missing')
        with patch.object(sys.stdin, 'isatty', return_value=False), redirect_stdout(io.StringIO()):
            self.assertEqual(value.run(), 3)
            self.assertEqual(value.run(), 3)
        self.assertEqual(sum(args[0] == 'new' for args in value.commands), 1)
        self.assertEqual(value.state['phase'], 'awaiting_drive')

    def test_resume_completed_job_uses_verified_local_copy(self):
        value = self.make_controller()
        with redirect_stdout(io.StringIO()):
            value.run()
            before = len(value.commands)
            self.assertEqual(value.run(resume=True), 0)
        self.assertEqual(len(value.commands), before)

    def test_run_id_cannot_change_configuration(self):
        self.make_controller()
        changed = copy.deepcopy(self.config)
        changed['gpu'] = 'A100'
        with self.assertRaises(ValueError):
            SyntheticColab(changed, self.root / 'local', 'test-01', sys.executable)

    def test_lock_prevents_concurrent_mutation(self):
        with controller.run_lock(self.root / 'locks'):
            with self.assertRaises(RuntimeError):
                with controller.run_lock(self.root / 'locks'):
                    self.fail('Acquired a second operation lock')
        self.assertFalse((self.root / 'locks' / 'operation.lock').exists())

    def test_archive_oversized_files_remain_explicit(self):
        run = self.root / 'archive'
        run.mkdir()
        worker.write_json(run / 'job.json', {'status': 'failed'})
        (run / 'large.mp4').write_bytes(b'x' * (2 * 1024 * 1024))
        manifest = worker.archive(run, 1)
        self.assertEqual(controller.verify_archive(run / 'artifacts.zip')[0]['status'], 'failed')
        row = next(r for r in manifest['files'] if r['path'] == 'large.mp4')
        self.assertFalse(row['included'])

    def test_archive_digest_detects_corruption(self):
        run = self.root / 'bad'
        run.mkdir()
        worker.write_json(run / 'job.json', {'status': 'success'})
        worker.archive(run, 1)
        source = run / 'artifacts.zip'
        with zipfile.ZipFile(source) as original:
            payload = {name: original.read(name) for name in original.namelist()}
        payload['job.json'] = b'{}'
        with zipfile.ZipFile(source, 'w') as modified:
            for name, data in payload.items():
                modified.writestr(name, data)
        with self.assertRaises(ValueError):
            controller.verify_archive(source)

    def test_completed_step_cannot_be_reused_after_modification(self):
        run = self.root / 'step'
        worker.write_json(run / 'result.json', {'status': 'success'})
        (run / 'best.pt').write_bytes(b'original')
        worker.seal_step(run)
        worker.verify_step(run)
        (run / 'best.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            worker.verify_step(run)

    def test_gaps_are_self_reported_and_include_longest_run(self):
        csv = self.root / 'frames.csv'
        csv.write_text('frame,source\n1,ROI_DETECTED\n2,NO_DETECTION\n3,NO_DETECTION\n4,ROI_DETECTED\n')
        result = worker.summarize_csv(csv, 50)
        self.assertEqual(result['reported_observation_fraction'], 0.5)
        self.assertEqual(result['longest_gaps'][0]['duration_ms'], 40)
        self.assertIsNone(result['ground_truth_accuracy'])

    def test_nontracking_csv_is_ignored(self):
        csv = self.root / 'training.csv'
        csv.write_text('epoch,loss\n1,0.4\n')
        self.assertIsNone(worker.summarize_csv(csv))

    def test_inventory_works_without_loading_models(self):
        folder = Path(self.config['project_root']) / '02_MODELS'
        folder.mkdir()
        (folder / 'sample.pt').write_bytes(b'not a pickle')
        result = worker.inventory(Path(self.config['project_root']))
        self.assertEqual(result['files'][0]['sha256'], worker.digest(folder / 'sample.pt'))

    def test_worker_failure_writes_receipt(self):
        task = {'name': 'bad-audit', 'kind': 'audit', 'data': 'missing/data.yaml'}
        run = self.root / 'failure'
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = worker.run_step(self.config, run, task, False)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((run / task['name'] / 'result.json').read_text())['status'], 'failed')

    def test_staging_preserves_source_and_normalizes_roboflow_paths(self):
        from PIL import Image
        import yaml
        source = Path(self.config['project_root']) / 'dataset'
        for split, color in [('train', (20, 90, 160)), ('valid', (170, 40, 70))]:
            (source / split / 'images').mkdir(parents=True)
            (source / split / 'labels').mkdir()
            Image.new('RGB', (20, 20), color).save(source / split / 'images' / 'example.png')
            (source / split / 'labels' / 'example.txt').write_text('0 0.5 0.5 0.3 0.3\n')
        data = source / 'data.yaml'
        data.write_text(yaml.safe_dump({'names': ['ball'], 'nc': 1, 'train': '../train/images', 'val': '../valid/images'}))
        original = data.read_bytes()
        task = {'name': 'audit', 'kind': 'audit', 'data': 'dataset/data.yaml', 'expected_names': ['ball']}
        staged = worker.prepare_dataset(Path(self.config['project_root']), self.root / 'scratch', task)
        result = worker.audit_dataset(staged, self.root / 'result', task)
        self.assertTrue(result['data_valid'])
        self.assertEqual(data.read_bytes(), original)
        self.assertNotEqual(staged, data)

    def test_training_retry_after_validation_failure_reuses_finished_training(self):
        weights = Path(self.config['project_root']) / 'base.pt'
        weights.write_bytes(b'initial')
        output = self.root / 'training'
        output.mkdir()
        task = {'weights': 'base.pt', 'imgsz': 320, 'epochs': 1}
        train_calls = []

        class Model:
            task = 'detect'

            def __init__(self, path):
                self.trainer = SimpleNamespace(best=output / 'train' / 'weights' / 'best.pt')

            def add_callback(self, *args):
                pass

            def train(self, **options):
                train_calls.append(options)
                self.trainer.best.parent.mkdir(parents=True)
                self.trainer.best.write_bytes(b'candidate')

            def val(self, **options):
                return SimpleNamespace(results_dict={'precision': 0.9})

        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Model)}), \
             patch.object(worker, 'prepare_dataset', return_value=self.root / 'staged.yaml'), \
             patch.object(worker, 'audit_dataset', return_value={'dataset_fingerprint': 'same'}):
            first = worker.train(Path(self.config['project_root']), self.root, output, task, False)
            second = worker.train(Path(self.config['project_root']), self.root, output, task, True)
        self.assertEqual(len(train_calls), 1)
        self.assertEqual(first['best_sha256'], second['best_sha256'])
        self.assertEqual(weights.read_bytes(), b'initial')

    def test_status_remains_available_while_mutating_job_is_locked(self):
        folder = self.root / 'states' / 'test-01'
        folder.mkdir(parents=True)
        saved = SyntheticColab(self.config, folder, 'test-01', sys.executable)
        saved.state.update(phase='running', session='synthetic-owned')
        saved.save()
        args = ['job', 'status', '--config', str(self.file), '--state-root', str(folder.parent),
                '--run-id', 'test-01', '--colab', sys.executable]
        with controller.run_lock(folder), patch.object(sys, 'argv', args), \
             patch.object(controller, 'Controller', SyntheticColab):
            self.assertEqual(controller.main(), 0)

    def worker_process(self, resume=False):
        stub = self.root / 'stub'
        stub.mkdir(exist_ok=True)
        (stub / 'torch.py').write_text('__version__="synthetic"\n'
            'class cuda:\n @staticmethod\n def is_available(): return False\n'
            'class version: cuda=None\n')
        env = dict(os.environ)
        repo = Path(__file__).resolve().parents[2]
        env['PYTHONPATH'] = str(stub) + os.pathsep + str(repo)
        args = [sys.executable, str(Path(worker.__file__)), '--config', str(self.file), '--run-id', 'subprocess']
        if resume:
            args.append('--resume')
        return subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

    def test_remote_entrypoint_and_resume_across_processes(self):
        first = self.worker_process()
        self.assertEqual(first.returncode, 0, first.stderr)
        run = Path(self.config['output_root']) / 'subprocess'
        report, _ = controller.verify_archive(run / 'artifacts.zip')
        self.assertEqual(report['status'], 'success')
        resumed = self.worker_process(resume=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        report, _ = controller.verify_archive(run / 'artifacts.zip')
        self.assertEqual(report['steps'][0]['status'], 'reused')

    def test_remote_failure_preserves_results_and_stops_following_steps(self):
        self.config['tasks'].extend([{'name': 'bad', 'kind': 'audit', 'data': 'missing/data.yaml'},
                                     {'name': 'never', 'kind': 'inventory'}])
        self.save_config()
        result = self.worker_process()
        self.assertEqual(result.returncode, 1, result.stderr)
        run = Path(self.config['output_root']) / 'subprocess'
        report, _ = controller.verify_archive(run / 'artifacts.zip')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([s['name'] for s in report['steps']], ['inventory', 'bad'])
        self.assertTrue((run / 'bad' / 'execution.log').exists())
        self.assertFalse((run / 'never').exists())

    def test_remote_time_budget_terminates_child_and_keeps_receipt(self):
        self.config['max_minutes'] = 0.000001
        self.save_config()
        result = self.worker_process()
        self.assertEqual(result.returncode, 1, result.stderr)
        run = Path(self.config['output_root']) / 'subprocess'
        receipt = json.loads((run / 'inventory' / 'result.json').read_text())
        self.assertEqual(receipt['error'], 'job_time_budget_exceeded')
        self.assertTrue((run / 'artifacts.zip').is_file())


if __name__ == '__main__':
    unittest.main()
