"""Validate/export another checkpoint from one completed research run."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--reference-bundle',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path,help='Prepared YAML from the same completed run')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True)
    for key,name in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp')]:
        folder=a.out/'cache'/name;folder.mkdir(parents=True);os.environ[key]=str(folder.resolve())
    os.environ['ATEN_CPU_CAPABILITY']='default'
    import torch
    import yaml
    from ultralytics import YOLO,settings
    from .train import PROFILES,export_parity
    settings.update({'sync':False})
    torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    manifest=json.loads((a.reference_bundle/'manifest.json').read_text())
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    epoch=int(checkpoint['epoch'])+1
    del checkpoint
    model=YOLO(a.checkpoint)
    spec=PROFILES[manifest['profile']]
    names=[model.names[i] for i in range(len(model.names))]
    if names!=spec['names'] or model.task!=spec['task']:raise ValueError('Checkpoint class/task mismatch')
    validation=model.val(data=str(a.data.resolve()),split='val',imgsz=spec['imgsz'],device='cpu',
                         batch=4,workers=0,plots=False,project=str(a.out.resolve()),name='validation')
    exported=Path(model.export(format='onnx',imgsz=spec['imgsz'],opset=17,simplify=False,
                               dynamic=False,nms=False,device='cpu'))
    bundle=a.out/'bundle';bundle.mkdir()
    target=bundle/'model.onnx';shutil.copy2(exported,target)
    parity=export_parity(model,target,yaml.safe_load(a.data.read_text()),spec['imgsz'])
    manifest.update({'model_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'checkpoint':a.checkpoint.name,'checkpoint_epoch':epoch,
        'validation_metrics':{k:float(v) for k,v in validation.results_dict.items()},
        'per_class_box_metrics':{spec['names'][int(i)]:{'precision':float(validation.box.p[j]),
            'recall':float(validation.box.r[j]),'map50':float(validation.box.ap50[j]),
            'map50_95':float(validation.box.ap[j])} for j,i in enumerate(validation.box.ap_class_index)},
        'export_parity':parity,'models_validated':False,'release_eligible':False})
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'checkpoint_epoch':epoch,'bundle':str(bundle),'parity':parity},indent=2))


if __name__=='__main__':main()
