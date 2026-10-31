"""Train a reproducible research baseline after a full-data preflight. No cloud upload."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
from pathlib import Path

import yaml

from .dataset_qc import audit, split_directory
from .run_control import RunControl

PROFILES = {
    'minimap': {'task': 'detect', 'base': 'yolo11n.pt', 'imgsz': 320,
                'names': ['ball', 'controlled', 'dot']},
    'field': {'task': 'detect', 'base': 'yolo11n.pt', 'imgsz': 640,
              'names': ['ball', 'cursor', 'cursor_secondary', 'goalkeeper', 'player', 'powerbar', 'referee']},
    'landmarks': {'task': 'pose', 'base': 'yolo11n-pose.pt', 'imgsz': 640, 'names': ['pitch']},
}


def export_parity(model, onnx_file, data_config, image_size):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_file), sess_options=options, providers=['CPUExecutionProvider'])
    folder = Path(data_config['val'])
    images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})[:3]
    if not images: raise ValueError('No validation images for export parity')
    errors = []
    model.model.eval()
    for path in images:
        image = cv2.imread(str(path))
        image = LetterBox(new_shape=(image_size,image_size),auto=False)(image=image)
        tensor = np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        with torch.inference_mode():
            reference = model.model(torch.from_numpy(tensor))
            if isinstance(reference,(tuple,list)): reference = reference[0]
            reference = reference.cpu().numpy()
        actual = session.run(None,{session.get_inputs()[0].name:tensor})[0]
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise RuntimeError('ONNX output shape/finite check failed')
        errors.append(float(np.max(np.abs(actual-reference))))
    if max(errors) > .05:
        raise RuntimeError(f'ONNX/PyTorch parity failed: max absolute raw-output error={max(errors)}')
    return {'images':len(images),'max_absolute_raw_output_error':max(errors),'passed':True,
            'scope':'FP32 Python CPU outputs; Android preprocessing/runtime still requires device verification'}


def validate_report(report, profile, research_only):
    spec = PROFILES[profile]
    if report['task'] != spec['task'] or report['names'] != spec['names']:
        raise ValueError('Profile task/class order does not match data.yaml')
    if profile=='landmarks' and report['kpt_shape']!=[32,3]:
        raise ValueError('Landmark profile requires exactly 32 keypoints with visibility')
    if not report['data_valid']:
        raise ValueError(f'Dataset QC failed: {len(report["data_errors"])} errors; normalize and audit the full export first')
    if not research_only and not report['eligible_for_independent_evaluation']:
        raise ValueError('Independent evaluation is not established; provide video groups/test data or use --research-only')
    return report


def preflight(data, expected_images, groups, profile, research_only):
    return validate_report(audit(data, expected_images, groups), profile, research_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--expected-images', type=int, required=True)
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--research-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help='AdamW initial learning rate')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--portable-cpu', action='store_true', help='Generic CPU kernels for restricted hosts; slower than optimized backends')
    parser.add_argument('--cpu-backend', choices=['generic','onednn'], default='generic')
    parser.add_argument('--weights', type=Path, help='Warm-start checkpoint; optimizer is newly initialized')
    parser.add_argument('--max-minutes', type=float, help='Training budget checked at saved epoch boundaries')
    parser.add_argument('--pose-train-sigma', type=float, help='Custom training OKS sigma; validation metric remains unchanged')
    parser.add_argument('--no-plots', action='store_true', help='Skip optional upstream plotting; numeric validation is unchanged')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or args.batch < 1 or not 0 < args.lr <= 0.01:
        parser.error('Positive epochs, threads and batch are required')
    if args.weights is not None and not args.weights.is_file():
        parser.error('Warm-start checkpoint does not exist')
    if args.max_minutes is not None and (not math.isfinite(args.max_minutes) or args.max_minutes <= 0):
        parser.error('Budget must be positive and finite')
    if args.pose_train_sigma is not None and (args.profile != 'landmarks' or not 0 < args.pose_train_sigma <= .25):
        parser.error('Training sigma requires landmarks and must be in (0,.25]')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory must be new or empty')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / 'cache'
    for key, name in [('TMPDIR', 'tmp'), ('YOLO_CONFIG_DIR', 'ultralytics'), ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch')]:
        folder = cache / name; folder.mkdir(parents=True, exist_ok=True); os.environ[key] = str(folder.resolve())
    qc = audit(args.data, args.expected_images, args.groups)
    (args.out / 'dataset-qc.json').write_text(json.dumps(qc, indent=2))
    validate_report(qc, args.profile, args.research_only)
    # YOLO receives absolute split directories; never relies on ../ fallback behavior.
    config = yaml.safe_load(args.data.read_text())
    for key in ['train', 'val', 'test']:
        if config.get(key):
            try: config[key] = str(split_directory(args.data.resolve().parent, config[key]))
            except ValueError:
                if key != 'test': raise
                config[key] = None
    config.pop('path', None)
    prepared = args.out / 'prepared-data.yaml'
    prepared.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.portable_cpu:
        os.environ['ATEN_CPU_CAPABILITY'] = 'default'
    os.environ.setdefault('POLARS_MAX_THREADS', str(args.threads))
    import torch
    if args.portable_cpu:
        torch.backends.mkldnn.enabled = args.cpu_backend == 'onednn'
        torch.backends.nnpack.enabled = False
    import matplotlib
    matplotlib.use('Agg')
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    torch.set_num_threads(args.threads)
    spec = PROFILES[args.profile]
    base = str(args.weights.resolve()) if args.weights else spec['base']
    trainer = None
    if args.portable_cpu:
        from .portable_cpu import prepare_checkpoint, trainer_class
        base = str(prepare_checkpoint(base, args.out/'pretrained-fp32.pt'))
        trainer = trainer_class(spec['task'])
    model = YOLO(base)
    control = RunControl(args.out, args.max_minutes)
    def configure_threads(trainer):
        # select_device resets this to NUM_THREADS during trainer construction.
        torch.set_num_threads(args.threads)
    def begin_training(trainer):
        control.start()
        if args.pose_train_sigma is not None:
            criterion = trainer.model.init_criterion()
            criterion.keypoint_loss.sigmas = torch.full((32,), args.pose_train_sigma, device=trainer.device)
            trainer.model.criterion = criterion
    model.add_callback('on_pretrain_routine_start', configure_threads)
    model.add_callback('on_train_start', begin_training)
    model.add_callback('on_fit_epoch_end', control.epoch_end)
    # Geometric augmentation is disabled for landmark ids until flip semantics are verified.
    result = model.train(trainer=trainer, data=str(prepared.resolve()), epochs=args.epochs, patience=args.patience,
                         imgsz=spec['imgsz'], batch=args.batch, device=args.device, workers=0,
                         seed=20261030, deterministic=True, cache=False, optimizer='AdamW', mosaic=0.0, mixup=0.0,
                         lr0=args.lr, lrf=0.01, warmup_epochs=3.0, warmup_bias_lr=0.0,
                         fliplr=0.0 if spec['task'] == 'pose' else 0.5, flipud=0.0, degrees=0.0,
                         translate=0.0 if spec['task'] == 'pose' else 0.1,
                         scale=0.0 if spec['task'] == 'pose' else 0.5,
                         hsv_h=0.0, hsv_s=0.15, hsv_v=0.2, plots=not args.no_plots,
                         project=str(args.out.resolve()), name='run', exist_ok=False)
    best = Path(model.trainer.best)
    epochs_completed = int(model.trainer.epoch) + 1
    if not best.is_file(): raise RuntimeError('Training produced no best checkpoint')
    model = YOLO(best)
    checkpoint_epoch = int(model.ckpt.get('epoch', -1)) + 1
    validation = model.val(data=str(prepared.resolve()), split='val', imgsz=spec['imgsz'],
                           device=args.device, batch=args.batch, workers=0, plots=not args.no_plots,
                           project=str(args.out.resolve()), name='validation')
    export = Path(model.export(format='onnx', imgsz=spec['imgsz'], opset=17, simplify=False,
                              dynamic=False, nms=False, device='cpu'))
    parity = export_parity(model,export,config,spec['imgsz'])
    bundle = args.out / 'bundle'; bundle.mkdir()
    target = bundle / 'model.onnx'; shutil.copy2(export, target)
    versions = {}
    for name in ['ultralytics', 'torch', 'torchvision', 'onnx', 'onnxruntime', 'numpy', 'Pillow', 'PyYAML']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    with target.open('rb') as handle:
        model_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    versions['torch'] = torch.__version__
    manifest = {'schema_version': 1, 'profile': args.profile, 'task': spec['task'],
                'classes': spec['names'], 'input_size': spec['imgsz'], 'input_layout': 'NCHW',
                'output_layout': 'channels_first', 'normalization': 'RGB/255',
                'model_sha256': model_sha256,
                'dataset_fingerprint': qc['dataset_fingerprint'], 'decoded_images': qc['decoded_images'],
                'seed': 20261030, 'epochs_requested': args.epochs, 'epochs_completed': epochs_completed,
                'checkpoint': 'best.pt', 'checkpoint_epoch': checkpoint_epoch,
                'learning_rate': args.lr,
                'optional_plots_enabled': not args.no_plots,
                'training_seconds': control.seconds, 'training_budget_minutes': args.max_minutes,
                'training_stop_reason': control.reason or 'upstream_early_stop_or_epochs',
                'cpu_backend': args.cpu_backend if args.portable_cpu else 'upstream',
                'warm_start': args.weights.name if args.weights else None, 'optimizer_resumed': False,
                'kpt_shape': config.get('kpt_shape'), 'pose_training_sigma': args.pose_train_sigma,
                'pose_validation_sigmas': 'upstream defaults; not modified by training sigma',
                'geometric_augmentation': 'disabled for unverified pitch landmark semantics' if spec['task']=='pose' else 'upstream detection settings',
                'torch_runtime_version': torch.__version__, 'portable_cpu': args.portable_cpu,
                'ball_semantics': 'marker_unverified' if args.profile == 'minimap' else 'unverified',
                'models_validated': False, 'release_eligible': False,
                'evaluation_scope': 'source validation split; no independent Mobile accuracy or Android benchmark',
                'validation_metrics': {k: float(v) for k, v in validation.results_dict.items()},
                'per_class_box_metrics': {spec['names'][int(i)]: {'precision':float(validation.box.p[j]),
                    'recall':float(validation.box.r[j]), 'map50':float(validation.box.ap50[j]),
                    'map50_95':float(validation.box.ap[j])} for j,i in enumerate(validation.box.ap_class_index)},
                'versions': versions, 'platform': platform.platform(),
                'android_benchmark': None, 'source_licenses': config.get('roboflow', {}),
                'export_parity': parity,
                'weights_origin': 'Ultralytics YOLO11 nano pretrained checkpoint; retain upstream licensing'}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.make_archive(str(args.out / 'model-bundle'), 'zip', bundle)
    print(json.dumps({'bundle': str(args.out/'model-bundle.zip'), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
