"""Build a private beginner bundle from hash-verified field weights and HUD APK.

Model/APK bytes are supplied locally, never committed to the public repository.
This packages existing checkpoints; it does not train or improve a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


RUNTIME_FILES = ('__init__.py', 'efootball_pc_brain.py', 'runtime_safety.py',
                 'field_postprocess.py', 'transport_rtt.py', 'ball_shadow.py')


def checked_bytes(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Artifact SHA-256 mismatch: '+str(path))
    return raw


def build(model, apk, output, model_sha256, apk_sha256, source_commit):
    model_bytes = checked_bytes(model, model_sha256)
    apk_bytes = checked_bytes(apk, apk_sha256)
    repo = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    files = {'models/field640.onnx': model_bytes,
             'efootphy-pc-hud-v0.2.1.apk': apk_bytes}
    for name in RUNTIME_FILES:
        files['tools/pc_brain/'+name] = (source/name).read_bytes()
    for name in ('01_CAI_DAT_PC.bat', '02_CHAY_TRO_LY.bat', 'requirements-pc-cpu.txt'):
        files[name] = (source/'windows'/name).read_bytes()
    files['start_pc_brain.bat'] = (repo/'start_pc_brain.bat').read_bytes()
    files['HUONG_DAN.txt'] = b'\xef\xbb\xbf'+(repo/'docs/handover/BEGINNER_PC.vi.md').read_bytes()
    for name in files:
        if name.endswith('.bat'):
            files[name] = files[name].replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    manifest = {
        'bundle': 'efootphy_PC_START_v0.2.1', 'source_commit': source_commit,
        'inference_device': 'Windows PC CPU', 'android_application_id': 'com.example.minimap.pchud',
        'model': {'role': 'main gameplay field detection, NOT minimap or pitch keypoints',
                  'input': [1,3,640,640], 'output': [1,11,8400], 'sha256': model_sha256,
                  'class_order': ['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee'],
                  'status': 'existing research baseline; no new training or accuracy promotion'},
        'physical_pc_phone_validation': 'pending',
        'files_sha256': {name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(files.items())}}
    files['bundle_manifest.json'] = (json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,11,9,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Bundle integrity check failed')
    return dict(file=str(output),bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),entries=len(files))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('model','apk','output','model-sha256','apk-sha256','source-commit'):
        parser.add_argument('--'+option,required=True)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
