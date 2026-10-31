"""Evaluate a frozen field ONNX export on a source split; no independent claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path,help='Prepared YAML from the training run')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--plots',action='store_true',help='Optional plots; numeric evaluation runs by default')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True)
    for key,name in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib')]:
        folder=a.out/'cache'/name;folder.mkdir(parents=True);os.environ[key]=str(folder.resolve())
    os.environ['ATEN_CPU_CAPABILITY']='default'
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO,settings
    from .train import PROFILES
    torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True;torch.set_num_threads(2)
    settings.update({'sync':False})
    manifest=json.loads((a.bundle/'manifest.json').read_text());model=a.bundle/'model.onnx'
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    if digest!=manifest['model_sha256'] or manifest['profile']!='field' or manifest['classes']!=PROFILES['field']['names']:
        raise ValueError('Field model checksum/profile/class mismatch')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape!=[1,3,640,640] or session.get_outputs()[0].shape!=[1,11,8400]:
        raise ValueError('Unexpected field ONNX shape')
    del session
    metrics=YOLO(model,task='detect').val(data=str(a.data.resolve()),split=a.split,imgsz=640,
        batch=1,device='cpu',workers=0,plots=a.plots,project=str(a.out.resolve()),name='metrics')
    report={'schema_version':1,'model_sha256':digest,'split':a.split,
        'scope':'source export only; match/video separation and Mobile generalization not established',
        'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
        'precision_recall_scope':'upstream PR-curve operating point on this split; not a frozen deployment threshold',
        'per_class':{manifest['classes'][int(i)]:{'map50':float(metrics.box.ap50[j]),
            'map50_95':float(metrics.box.ap[j])} for j,i in enumerate(metrics.box.ap_class_index)},
        'models_validated':False,'release_eligible':False,'android_benchmark':None}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
