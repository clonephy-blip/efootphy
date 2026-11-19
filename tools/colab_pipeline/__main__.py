"""Provision, execute, collect and release an owned Colab session."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from .worker import digest, write_json

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}')
KINDS = {'inventory', 'audit', 'train', 'benchmark', 'export'}


def read_config(path, selected):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema')
    if config.get('gpu') not in {'T4', 'L4', 'G4', 'A100', 'H100'}:
        raise ValueError('Choose an explicitly supported Colab GPU')
    if config.get('auth', 'oauth2') not in {'oauth2', 'adc'}:
        raise ValueError('Authentication must be oauth2 or adc')
    for field in ('project_root', 'output_root'):
        if not isinstance(config.get(field), str) or not PurePosixPath(config[field]).is_absolute():
            raise ValueError(f'{field} must be an absolute Linux VM path')
    minutes = config.get('max_minutes')
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError('max_minutes must be positive and finite')
    limit = config.get('download_limit_mb', 200)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError('download_limit_mb must be an integer from 1 to 200')
    fps = config.get('source_fps', 59.7)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('source_fps must be positive and finite')
    if not isinstance(config.get('requirements'), list) or not config['requirements'] or not all(isinstance(v, str) and v for v in config['requirements']):
        raise ValueError('requirements must be a nonempty package list')
    tasks = config.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('At least one task is required')
    known = {}
    for task in tasks:
        name = task.get('name')
        if not isinstance(name, str) or NAME.fullmatch(name) is None or name in known:
            raise ValueError('Every task needs a unique simple name')
        if task.get('kind') not in KINDS:
            raise ValueError(f'Unsupported task kind: {name}')
        known[name] = task
    names = selected.split(',') if selected else config.get('default_tasks', list(known))
    if not names or len(set(names)) != len(names) or any(name not in known for name in names):
        raise ValueError('Unknown or repeated task selection')
    config['tasks'] = [known[name] for name in names]
    completed = set()
    for task in config['tasks']:
        kind = task['kind']
        required = {'train': ['weights', 'imgsz'], 'export': ['weights', 'imgsz'],
                    'benchmark': ['video', 'field_model', 'ball_model', 'pitch_model']}.get(kind, [])
        for key in required:
            if key not in task:
                raise ValueError(f'{task["name"]} requires {key}')
        if kind in {'train', 'audit'} and task.get('prepare') not in {'ball_v2', 'actors_v2'} and not task.get('data'):
            raise ValueError(f'{task["name"]} requires a dataset YAML or V2 preparation')
        for key, value in task.items():
            if isinstance(value, str) and 'EDIT_ME' in value:
                raise ValueError(f'Fill in {task["name"]}.{key} before selecting this task')
            if key in {'weights', 'field_model', 'ball_model', 'pitch_model'} and isinstance(value, str) and value.startswith('@'):
                if value[1:] not in completed:
                    raise ValueError(f'Training dependency {value} must precede {task["name"]}')
                if known[value[1:]]['kind'] != 'train':
                    raise ValueError(f'{value} must reference a training task')
        if kind == 'train':
            for key in ('epochs', 'batch', 'imgsz'):
                value = task.get(key, 1 if key == 'epochs' else 8)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{task["name"]}.{key} must be a positive integer')
        completed.add(task['name'])
    config.pop('default_tasks', None)
    return config


def verify_archive(path):
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate members in artifact archive')
        for name in names:
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
                raise ValueError('Invalid archive member path')
        manifest = json.loads(bundle.read('artifact-manifest.json'))
        included = {row['path']: row for row in manifest['files'] if row['included']}
        if set(names) != set(included) | {'artifact-manifest.json'}:
            raise ValueError('Archive membership differs from manifest')
        for name, row in included.items():
            with bundle.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != row['sha256'] or bundle.getinfo(name).file_size != row['bytes']:
                raise ValueError(f'Artifact digest mismatch: {name}')
        report = json.loads(bundle.read('job.json'))
    return report, manifest


@contextlib.contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'operation.lock'
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'Another operation owns {path}. Check that process before removing a stale lock.') from error
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class Controller:
    def __init__(self, config, folder, run_id, binary):
        self.config, self.folder, self.run_id = config, folder, run_id
        self.binary = str(Path(binary).resolve()) if Path(binary).is_file() else shutil.which(binary)
        if self.binary is None:
            raise FileNotFoundError('Install google-colab-cli in Linux/macOS and select its executable with --colab')
        self.state_file = folder / 'state.json'
        self.config_file = folder / 'config.json'
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            if json.loads(self.config_file.read_text()) != config:
                raise ValueError('Run ID belongs to another configuration. Use its original task selection or a new ID.')
        else:
            self.state = {'run_id': run_id, 'phase': 'new', 'session': None, 'generation': 0}
            write_json(self.config_file, config)
            self.save()

    def save(self):
        write_json(self.state_file, self.state)

    def command(self, *args, input_text=None, interactive=False, check=True, timeout=None):
        command = [self.binary, '--config', str(self.folder / 'colab-sessions.json'),
                   '--auth', self.config.get('auth', 'oauth2'), *map(str, args)]
        print('$ ' + shlex.join(command), flush=True)
        if interactive:
            result = subprocess.run(command, timeout=timeout)
        else:
            result = subprocess.run(command, input=input_text or '', text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            with (self.folder / 'controller.log').open('a', encoding='utf-8') as handle:
                handle.write(result.stdout)
            print(result.stdout[-6000:], end='', flush=True)
        if check and result.returncode:
            raise RuntimeError(f'colab {args[0]} failed with exit {result.returncode}. Inspect controller.log.')
        return result

    def prepare(self, cpu=False):
        if self.state['phase'] in {'new', 'stopped'}:
            self.state['generation'] += 1
            self.state['session'] = 'ef-' + self.run_id + '-' + uuid.uuid4().hex[:6]
            self.state['phase'] = 'allocating'
            self.save()
            args = ['new', '-s', self.state['session']]
            if not cpu:
                args.extend(['--gpu', self.config['gpu']])
            self.command(*args, interactive=sys.stdin.isatty(), timeout=600)
            self.state['phase'] = 'awaiting_drive'
            self.save()
        if self.state['phase'] == 'allocating':
            raise RuntimeError('Allocation outcome is uncertain. Inspect status and stop this owned session before retrying.')
        probe = 'import os\nprint("__EFOOTPHY_DRIVE__=" + str(os.path.isdir(' + repr(self.config['project_root']) + ')))\n'
        result = self.command('exec', '-s', self.state['session'], '--timeout', '120', input_text=probe)
        if '__EFOOTPHY_DRIVE__=True' not in result.stdout:
            self.state['phase'] = 'awaiting_drive'
            self.save()
            if not sys.stdin.isatty():
                auth = self.config.get('auth', 'oauth2')
                print('Mount Drive in your terminal, then rerun this command. The session remains allocated.')
                print(shlex.join([self.binary, '--config', str(self.folder / 'colab-sessions.json'), '--auth', auth,
                                  'drivemount', '-s', self.state['session']]))
                return False
            self.command('drivemount', '-s', self.state['session'], interactive=True, timeout=660)
            result = self.command('exec', '-s', self.state['session'], input_text=probe)
            if '__EFOOTPHY_DRIVE__=True' not in result.stdout:
                raise RuntimeError('Drive is mounted but project_root was not found. Correct the path.')
        self.state['phase'] = 'ready'
        self.save()
        return True

    def collect(self):
        remote = str(PurePosixPath(self.config['output_root']) / self.run_id / 'artifacts.zip')
        temporary = self.folder / 'artifacts.zip.part'
        self.command('download', '-s', self.state['session'], remote, temporary, timeout=900)
        report, manifest = verify_archive(temporary)
        if report['run_id'] != self.run_id or report['config_sha256'] != digest(self.config_file):
            raise ValueError('Collected archive belongs to another run or configuration')
        temporary.replace(self.folder / 'artifacts.zip')
        summary = {'run_id': self.run_id, 'status': report['status'], 'steps': report['steps'],
                   'runtime': report['runtime'], 'persistent_output': manifest['persistent_output'],
                   'archive_sha256': digest(self.folder / 'artifacts.zip'),
                   'remote_only_files': [r['path'] for r in manifest['files'] if not r['included']],
                   'release_eligible': False}
        write_json(self.folder / 'summary.json', summary)
        self.state['collected'] = True
        self.state['job_status'] = report['status']
        self.save()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return report['status'] == 'success'

    def stop(self):
        if self.state['phase'] in {'new', 'stopped'}:
            return
        self.command('stop', '-s', self.state['session'], timeout=180)
        self.state['phase'] = 'stopped'
        self.save()

    def run(self, resume=False):
        if self.state.get('job_status') and not resume:
            raise ValueError('A job has already finished under this run ID. Use --resume or choose a new ID.')
        if self.state.get('job_status') == 'success' and resume and (self.folder / 'artifacts.zip').exists():
            verify_archive(self.folder / 'artifacts.zip')
            print('This run already completed and its local artifacts verified.')
            return 0
        try:
            prepared = self.prepare()
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                error.add_note(f'Session cleanup also needs attention: {cleanup_error}')
            raise
        if not prepared:
            return 3
        session = self.state['session']
        try:
            self.command('usage', check=False, timeout=120)
            self.command('install', '-s', session, *self.config['requirements'], timeout=1200)
            root = Path(__file__).resolve().parents[2]
            bundle = self.folder / 'worker.zip'
            paths = ['tools/colab_pipeline/worker.py', 'tools/colab_pipeline/__init__.py',
                     'tools/vision_qc/dataset_qc.py']
            with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in paths:
                    archive.write(root / name, name)
                archive.writestr('tools/__init__.py', '')
                archive.writestr('tools/vision_qc/__init__.py', '')
            remote_base = '/content/ef-' + self.run_id
            self.command('upload', '-s', session, bundle, remote_base + '.zip', timeout=300)
            self.command('upload', '-s', session, self.config_file, remote_base + '.json', timeout=120)
            arguments = [sys.executable, '-u', remote_base + '/tools/colab_pipeline/worker.py',
                         '--config', remote_base + '.json', '--run-id', self.run_id]
            if resume:
                arguments.append('--resume')
            # The VM selects its own Python executable. No shell interpolation is used.
            bootstrap = ('import sys, subprocess, zipfile\n'
                         + f'zipfile.ZipFile({(remote_base + ".zip")!r}).extractall({remote_base!r})\n'
                         + f'sys.path.insert(0, {remote_base!r})\n'
                         + f'args = {arguments!r}\nargs[0] = sys.executable\n'
                         + 'subprocess.run(args, check=True)\n')
            # Children need this path too when launched as individual task processes.
            bootstrap = bootstrap.replace('import sys, subprocess, zipfile', 'import os, sys, subprocess, zipfile')
            bootstrap = bootstrap.replace('subprocess.run(args, check=True)',
                                          f'os.environ["PYTHONPATH"] = {remote_base!r}\nsubprocess.run(args, check=True)')
            self.state['phase'] = 'running'
            self.save()
            try:
                self.command('exec', '-s', session, '--timeout', '0', input_text=bootstrap,
                             timeout=int(self.config['max_minutes'] * 60) + 1800)
            except (RuntimeError, subprocess.TimeoutExpired):
                self.state['phase'] = 'collecting_after_error'
                self.save()
            success = self.collect()
            return 0 if success else 1
        finally:
            try:
                self.command('usage', check=False, timeout=120)
            finally:
                self.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'prepare', 'run', 'collect', 'status', 'stop'])
    parser.add_argument('--config', type=Path, default=Path('tools/colab_pipeline/pilot.example.json'))
    parser.add_argument('--tasks', help='Ordered comma-separated task names')
    parser.add_argument('--run-id', default='pilot-001')
    parser.add_argument('--state-root', type=Path, default=Path('.efootphy-colab'))
    parser.add_argument('--colab', default='colab')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if NAME.fullmatch(args.run_id) is None:
            raise ValueError('Use a simple run ID, up to 48 letters/digits/underscores/hyphens')
        config = read_config(args.config, args.tasks)
        if args.action == 'plan':
            print(json.dumps({'run_id': args.run_id, 'gpu_requested': config['gpu'],
                              'max_worker_minutes': config['max_minutes'],
                              'steps': [t['name'] for t in config['tasks']],
                              'project_root': config['project_root'], 'output_root': config['output_root'],
                              'lifecycle': ['allocate', 'authorize Drive', 'install', 'execute', 'verify archive', 'stop'],
                              'cloud_calls': 0}, ensure_ascii=False, indent=2))
            return 0
        folder = (args.state_root / args.run_id).resolve()
        if args.action == 'status' and not (folder / 'state.json').exists():
            raise ValueError('This run has no saved session state')
        operation = contextlib.nullcontext() if args.action == 'status' else run_lock(folder)
        with operation:
            controller = Controller(config, folder, args.run_id, args.colab)
            if args.action == 'prepare':
                return 0 if controller.prepare() else 3
            if args.action == 'run':
                return controller.run(args.resume)
            if args.action == 'collect':
                if not controller.prepare(cpu=True):
                    return 3
                try:
                    return 0 if controller.collect() else 1
                finally:
                    controller.stop()
            if args.action == 'stop':
                controller.stop()
            else:
                if controller.state['session'] is None:
                    raise ValueError('This run has no allocated session')
                controller.command('status', '-s', controller.state['session'], timeout=120)
                controller.command('usage', timeout=120)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
