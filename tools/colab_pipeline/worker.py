"""Run versioned training and offline QC jobs inside a Colab VM."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load Python module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def prepare_dataset(root, work, task):
    """Reuse the existing V2 data preparation, or stage an explicit YAML export."""
    import yaml
    preparation = task.get('prepare')
    if preparation in ('ball_v2', 'actors_v2'):
        filename = 'train_01_ball_roi_v2.py' if preparation == 'ball_v2' else 'train_02_player_mobile_v2_960.py'
        module = load_module(root / '05_TRAINING_PIPELINES' / filename, 'efootphy_prepare_' + preparation)
        original_root = Path('/content/drive/MyDrive/efootphy_V2')
        for key, value in list(vars(module).items()):
            if isinstance(value, Path) and value.is_relative_to(original_root):
                setattr(module, key, root / value.relative_to(original_root))
        module.LOCAL_WORK_DIR = work / 'dataset'
        if preparation == 'ball_v2':
            module.LOCAL_FIELD_DIR = work / 'field-source'
            data = module.stage_1_prepare_and_inject_hard_negatives()
        else:
            data = module.stage_1_prepare_fast_local_dataset()
            marker = work / 'augmentation-complete.json'
            if not marker.exists():
                module.stage_2_inject_mobile_hud_augmentation(data, ratio=task.get('hud_ratio', 0.35))
                write_json(marker, {'completed': True})
    else:
        source = project_path(root, task['data'])
        staged = work / 'dataset'
        if not staged.exists():
            shutil.copytree(source.parent, staged)
        data = staged / source.name
    data = Path(data)
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    # Existing V2 preparation writes its local root here. Explicit YAML exports
    # are rebased to the copied dataset so sources on Drive stay untouched.
    if preparation not in ('ball_v2', 'actors_v2'):
        original = yaml.safe_load(source.read_text(encoding='utf-8'))
        original_base = Path(original.get('path', source.parent))
        if not original_base.is_absolute():
            original_base = source.parent / original_base
        for split in ('train', 'val', 'test'):
            value = original.get(split)
            if not value:
                continue
            if not isinstance(value, str):
                raise ValueError('Dataset splits must be image directories')
            actual = (original_base / value).resolve()
            if not actual.is_dir() and value.startswith('../'):
                actual = (original_base / value[3:]).resolve()
            try:
                relative = actual.relative_to(source.parent.resolve())
            except ValueError as error:
                raise ValueError(f'{split} images must be inside the supplied dataset directory') from error
            config[split] = str(staged / relative)
        config['path'] = str(staged)
    else:
        base = Path(config.get('path', data.parent))
        for split in ('train', 'val', 'test'):
            value = config.get(split)
            if value:
                config[split] = str((base / value).resolve())
    data.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return data


def audit_dataset(data, output, task):
    import yaml
    from tools.vision_qc.dataset_qc import audit
    report = audit(data, expected_images=task.get('expected_images'), require_test=False)
    write_json(output / 'dataset-audit.json', report)
    if not report['data_valid']:
        raise ValueError(f'Dataset audit failed. Inspect {output / "dataset-audit.json"}')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    expected = task.get('expected_names')
    if expected is not None and report['names'] != expected:
        raise ValueError(f'Class order mismatch: expected {expected}, found {report["names"]}')
    if task.get('kpt_shape') is not None and config.get('kpt_shape') != task['kpt_shape']:
        raise ValueError('Pose keypoint shape differs from the requested annotation contract')
    if task.get('task', 'detect') == 'pose' and not config.get('kpt_shape'):
        raise ValueError('Pose training requires a keypoint dataset')
    return report


def train(root, work, output, task, resume):
    from ultralytics import YOLO
    data = prepare_dataset(root, work, task)
    audit = audit_dataset(data, output, task)
    weights = project_path(root, task['weights'])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    inputs = {'dataset_fingerprint': audit['dataset_fingerprint'], 'source_weights_sha256': digest(weights)}
    if task.get('prepare') in {'ball_v2', 'actors_v2'}:
        filename = 'train_01_ball_roi_v2.py' if task['prepare'] == 'ball_v2' else 'train_02_player_mobile_v2_960.py'
        inputs['preparation_sha256'] = digest(root / '05_TRAINING_PIPELINES' / filename)
    inputs_file = output / 'training-inputs.json'
    if resume and inputs_file.exists() and json.loads(inputs_file.read_text()) != inputs:
        raise ValueError('Training inputs changed since the checkpoint was created')
    write_json(inputs_file, inputs)
    last = output / 'train' / 'weights' / 'last.pt'
    checkpoint = last if resume and last.is_file() else weights
    complete_file = output / 'training-complete.json'
    if resume and complete_file.exists():
        complete = json.loads(complete_file.read_text())
        best = Path(complete['best'])
        if digest(best) != complete['sha256']:
            raise ValueError('Completed training checkpoint was modified')
    else:
        model = YOLO(str(checkpoint))
        if model.task != task.get('task', 'detect'):
            raise ValueError(f'Model task {model.task} differs from dataset task')
        def save_progress(trainer):
            write_json(output / 'progress.json', {'epoch': int(trainer.epoch) + 1, 'observed_at': time.time()})
        model.add_callback('on_model_save', save_progress)
        options = dict(data=str(data), epochs=task.get('epochs', 1), imgsz=task['imgsz'],
                       batch=task.get('batch', 8), device=0, workers=2, seed=42,
                       project=str(output), name='train', exist_ok=bool(resume),
                       save=True, save_period=1, plots=False, patience=task.get('patience', 10))
        if model.task == 'pose':
            options.update(fliplr=0.0, flipud=0.0, degrees=0.0, mosaic=0.0)
        if resume and last.is_file():
            model.train(resume=True)
        else:
            model.train(**options)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError('Training returned without a best checkpoint')
        write_json(complete_file, {'best': str(best), 'sha256': digest(best)})
    if not best.is_file():
        raise RuntimeError('Training returned without a best checkpoint')
    validation = YOLO(str(best)).val(data=str(data), imgsz=task['imgsz'], device=0,
                                   batch=task.get('batch', 8), plots=False,
                                   project=str(output), name='validation', exist_ok=bool(resume))
    return {'best': str(best), 'best_sha256': digest(best), 'last': str(last),
            'data_yaml': str(data), 'dataset_fingerprint': audit['dataset_fingerprint'],
            'source_weights_sha256': digest(weights), 'optimizer_resumed': checkpoint == last,
            'metrics': {k: float(v) for k, v in validation.results_dict.items()},
            'evaluation_scope': 'source validation split', 'release_eligible': False}


def candidate(root, run, tasks, reference):
    if reference.startswith('@'):
        name = reference[1:]
        if name not in tasks or tasks[name]['kind'] != 'train':
            raise ValueError(f'Unknown training dependency: {reference}')
        report = json.loads((run / name / 'result.json').read_text())
        if report['status'] != 'success':
            raise ValueError(f'Training dependency failed: {name}')
        path = Path(report['result']['best'])
        if digest(path) != report['result']['best_sha256']:
            raise ValueError(f'Checkpoint changed since training: {name}')
        return path
    path = project_path(root, reference)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def benchmark(root, run, output, tasks, task):
    sys.path.insert(0, str(root))
    module = load_module(root / task.get('script', 'run_optimized_v5.py'), 'efootphy_benchmark')
    field = candidate(root, run, tasks, task['field_model'])
    ball = candidate(root, run, tasks, task['ball_model'])
    pose = candidate(root, run, tasks, task['pitch_model'])
    video = project_path(root, task['video'])
    if not video.is_file():
        raise FileNotFoundError(video)
    begin = time.perf_counter()
    module.run_v5_60fps_engine(video_path=str(video), field_model=str(field),
                             ball_roi_model=str(ball), pitch_pose_model=str(pose),
                             output_dir=str(output), export_fp16=False,
                             render_panel_every_n=task.get('render_every', 3),
                             max_frames=task.get('max_frames', 300))
    elapsed = time.perf_counter() - begin
    report_path = output / 'BRAIN_V5_60FPS_REPORT.json'
    report = json.loads(report_path.read_text())
    if not isinstance(report.get('total_frames'), int) or report['total_frames'] <= 0:
        raise ValueError('Benchmark produced no processed frames')
    return {'source_report': report, 'observed_wall_seconds': elapsed,
            'processed_frames_per_wall_second': report['total_frames'] / elapsed,
            'benchmark_script_sha256': digest(root / task.get('script', 'run_optimized_v5.py')),
            'video_sha256': digest(video),
            'model_sha256': {'field': digest(field), 'ball': digest(ball), 'pitch': digest(pose)},
            'measurement_scope': 'offline file processing including rendering; device latency unmeasured',
            'release_eligible': False}


def export_model(root, run, output, tasks, task):
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    weights = candidate(root, run, tasks, task['weights'])
    local = output / 'model.pt'
    shutil.copy2(weights, local)
    path = Path(YOLO(str(local)).export(format='onnx', imgsz=task['imgsz'],
                                      half=False, simplify=False, dynamic=False, opset=17, device='cpu'))
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return {'model': str(path), 'sha256': digest(path), 'precision': 'FP32',
            'input': [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_inputs()],
            'output': [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_outputs()],
            'numerical_parity': 'not measured', 'release_eligible': False}


def summarize_csv(path, fps=59.7):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'frame', 'source'}.issubset(rows[0]):
        return None
    gaps = []
    for missing, group in itertools.groupby(rows, key=lambda r: r['source'] == 'NO_DETECTION'):
        sequence = list(group)
        if missing:
            gaps.append({'start_frame': int(sequence[0]['frame']), 'end_frame': int(sequence[-1]['frame']),
                         'frames': len(sequence), 'duration_ms': len(sequence) * 1000 / fps})
    count = sum(r['source'] != 'NO_DETECTION' for r in rows)
    return {'frames': len(rows), 'reported_observation_fraction': count / len(rows),
            'longest_gaps': sorted(gaps, key=lambda v: v['frames'], reverse=True)[:20],
            'fps_assumption': fps, 'ground_truth_accuracy': None,
            'scope': 'detector self-reported gaps; includes cutscenes and unreviewed detections'}


def inventory(root):
    folders = ['02_MODELS', '05_TRAINING_PIPELINES', '03_BRAIN_CORE', '02_BALL_DETECTION', '00_SOURCE_DATASETS']
    files = []
    for folder in folders:
        path = root / folder
        for item in sorted(path.rglob('*')) if path.exists() else []:
            if item.is_file() and item.suffix in {'.pt', '.onnx', '.py', '.zip', '.yaml'}:
                row = {'path': str(item.relative_to(root)), 'bytes': item.stat().st_size}
                if item.suffix in {'.pt', '.onnx', '.py', '.yaml'}:
                    row['sha256'] = digest(item)
                files.append(row)
    return {'project': str(root), 'files': files, 'dataset_training_history': 'requires training manifests'}


def run_step(config, run, task, resume):
    name = task['name']
    root = Path(config['project_root'])
    work = Path(config.get('scratch_root', '/content/efootphy-colab')) / run.name / name
    output = run / name
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if task['kind'] == 'inventory':
            value = inventory(root)
        elif task['kind'] == 'train':
            value = train(root, work, output, task, resume)
        elif task['kind'] == 'audit':
            value = audit_dataset(prepare_dataset(root, work, task), output, task)
        elif task['kind'] == 'benchmark':
            value = benchmark(root, run, output, {t['name']: t for t in config['tasks']}, task)
        elif task['kind'] == 'export':
            value = export_model(root, run, output, {t['name']: t for t in config['tasks']}, task)
        else:
            raise ValueError('Unsupported task kind')
        report = {'status': 'success', 'result': value}
    except Exception as error:
        traceback.print_exc()
        report = {'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)}
    write_json(output / 'result.json', report)
    return 0 if report['status'] == 'success' else 1


def archive(run, limit_mb):
    files = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name not in {'artifacts.zip', 'artifact-manifest.json'} and '.tmp' != path.suffix:
            files.append({'path': str(path.relative_to(run)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    def priority(row):
        suffix = Path(row['path']).suffix
        return (0 if suffix == '.json' else 1 if suffix in {'.csv', '.txt', '.log', '.yaml'} else 2 if suffix in {'.pt', '.onnx'} else 3, row['path'])
    total = 0
    for row in sorted(files, key=priority):
        row['included'] = total + row['bytes'] <= limit_mb * 1024 * 1024
        if row['included']:
            total += row['bytes']
    manifest = {'schema_version': 1, 'files': files, 'archive_input_bytes': total,
                'persistent_output': str(run)}
    write_json(run / 'artifact-manifest.json', manifest)
    temporary = run / 'artifacts.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(run / 'artifact-manifest.json', 'artifact-manifest.json')
        for row in files:
            if row['included']:
                bundle.write(run / row['path'], row['path'])
    temporary.replace(run / 'artifacts.zip')
    return manifest


def seal_step(output):
    files = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'step-integrity.json':
            files.append({'path': str(path.relative_to(output)), 'sha256': digest(path)})
    write_json(output / 'step-integrity.json', {'files': files})


def verify_step(output):
    receipt = json.loads((output / 'step-integrity.json').read_text())
    for row in receipt['files']:
        if digest(output / row['path']) != row['sha256']:
            raise ValueError(f'Completed step artifact changed: {row["path"]}')


def run_job(config_file, run_id, resume=False):
    import torch
    config = json.loads(Path(config_file).read_text())
    root = Path(config['project_root'])
    if not root.is_dir():
        raise FileNotFoundError('Project directory is unavailable. Mount Drive and verify project_root.')
    if any(t['kind'] in {'train', 'benchmark'} for t in config['tasks']) and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Colab session')
    run = Path(config['output_root']) / run_id
    config_hash = digest(config_file)
    if run.exists():
        old = json.loads((run / 'job.json').read_text())
        if not resume or old['config_sha256'] != config_hash:
            raise ValueError('Run already exists. Resume with identical configuration or choose a new run ID.')
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, run / 'config.json')
    report = {'schema_version': 1, 'run_id': run_id, 'config_sha256': config_hash,
              'started_at': time.time(), 'status': 'running', 'steps': [],
              'worker_sha256': digest(__file__),
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              'release_eligible': False}
    write_json(run / 'job.json', report)
    deadline = time.monotonic() + config['max_minutes'] * 60
    for task in config['tasks']:
        result_file = run / task['name'] / 'result.json'
        if resume and result_file.exists() and json.loads(result_file.read_text())['status'] == 'success':
            verify_step(result_file.parent)
            report['steps'].append({'name': task['name'], 'status': 'reused'})
            continue
        args = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(config_file),
                '--run-id', run_id, '--step', task['name']]
        if resume:
            args.append('--resume')
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with (result_file.parent / 'execution.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                print(json.dumps({'task': task['name'], 'state': 'running', 'remaining_seconds': max(0, int(deadline-time.monotonic()))}), flush=True)
                if time.monotonic() >= deadline:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    write_json(result_file, {'status': 'failed', 'error': 'job_time_budget_exceeded'})
                    break
                time.sleep(5)
        result = json.loads(result_file.read_text()) if result_file.exists() else {'status': 'failed', 'error': 'missing_step_receipt'}
        if child.returncode != 0 and result['status'] == 'success':
            result = {'status': 'failed', 'error': f'worker_exit_{child.returncode}'}
            write_json(result_file, result)
        report['steps'].append({'name': task['name'], 'status': result['status']})
        if result['status'] == 'success':
            seal_step(result_file.parent)
        write_json(run / 'job.json', report)
        if result['status'] != 'success':
            report['status'] = 'failed'
            break
    if report['status'] == 'running':
        report['status'] = 'success'
    report['finished_at'] = time.time()
    write_json(run / 'job.json', report)
    for csv_file in run.rglob('*.csv'):
        summary = summarize_csv(csv_file, config.get('source_fps', 59.7))
        if summary is not None:
            write_json(csv_file.with_suffix('.gaps.json'), summary)
    archive(run, config.get('download_limit_mb', 200))
    return 0 if report['status'] == 'success' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--step')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.step:
        selected = next(t for t in config['tasks'] if t['name'] == args.step)
        return run_step(config, Path(config['output_root']) / args.run_id, selected, args.resume)
    return run_job(args.config, args.run_id, args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
