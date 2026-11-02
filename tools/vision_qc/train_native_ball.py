"""Warm-start a ball-only research branch on native-resolution crop data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .dataset_qc import audit
from .train import validate_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--max-minutes',type=float,default=5.)
    p.add_argument('--learning-rate',type=float,default=.0005)
    p.add_argument('--stop-after-epochs',type=int,help='Stop after a saved epoch without changing the requested LR schedule')
    a=p.parse_args()
    if a.out.exists() or not 1<=a.epochs<=100 or not 0<a.max_minutes<=15 or not 0<a.learning_rate<=.01 or (a.stop_after_epochs is not None and not 1<=a.stop_after_epochs<=a.epochs):
        raise ValueError('New output and bounded budget required')
    provenance=json.loads((a.data.parent/'crop-provenance.json').read_text())
    if provenance.get('input_side',provenance.get('native_side'))!=320 or provenance['validation_crop_selection_uses_labels']:
        raise ValueError('Expected native 320px crops and label-independent validation grid')
    qc=audit(a.data,expected_images=sum(provenance['tile_counts'].values()))
    validate_report(qc,'field',research_only=True)
    os.environ['YOLO_CONFIG_DIR']=str(a.out/'settings')
    os.environ['POLARS_MAX_THREADS']='2'
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync':False})
    torch.set_num_threads(2)
    model=YOLO(str(a.weights));started=time.monotonic();history=[]
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'dataset-qc.json').write_text(json.dumps(qc,indent=2))
    def configure(trainer):torch.set_num_threads(2)
    def epoch_end(trainer):
        updates=int(trainer.ema.updates)
        # Upstream invokes this callback again during final validation.
        if history and updates==history[-1]['optimizer_updates']:
            return
        record={'epoch':trainer.epoch+1,'elapsed_seconds':time.monotonic()-started,
                'optimizer_updates':updates,
                'metrics':{str(k):float(v) for k,v in trainer.metrics.items()}}
        history.append(record)
        snapshot=a.out/'snapshots'/f"epoch-{record['epoch']:03d}";snapshot.mkdir(parents=True)
        # Epoch callbacks run after upstream saves model/optimizer state. Publish
        # a closed, fsynced snapshot before exposing its ready marker.
        for name,path in [('last.pt',trainer.last),('best.pt',trainer.best)]:
            if not Path(path).is_file():raise ValueError('Epoch checkpoint was not saved')
            target=snapshot/name;temporary=target.with_suffix('.tmp')
            shutil.copyfile(path,temporary)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            os.replace(temporary,target)
        (a.out/'epoch-history.json').write_text(json.dumps(history,indent=2))
        state={'epoch':record['epoch'],'history':history,'requested_epochs':a.epochs,
               'learning_rate':a.learning_rate,'warm_start_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
               'crop_provenance_sha256':hashlib.sha256((a.data.parent/'crop-provenance.json').read_bytes()).hexdigest(),
               'checkpoint_sha256':{n:hashlib.sha256((snapshot/n).read_bytes()).hexdigest() for n in ['last.pt','best.pt']},
               'independent_match_validation':False,'android_fps_validated':False}
        ready=snapshot/'ready.json';temporary=ready.with_suffix('.tmp')
        with temporary.open('w') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temporary,ready)
        if record['elapsed_seconds']>=a.max_minutes*60 or (a.stop_after_epochs is not None and len(history)>=a.stop_after_epochs):trainer.stop=True
    model.add_callback('on_pretrain_routine_start',configure)
    model.add_callback('on_fit_epoch_end',epoch_end)
    model.train(data=str(a.data.resolve()),imgsz=320,epochs=a.epochs,batch=8,nbs=8,workers=0,device='cpu',
                optimizer='AdamW',lr0=a.learning_rate,lrf=.1,warmup_epochs=.3,warmup_bias_lr=0.,
                classes=[0],mosaic=0.,mixup=0.,scale=.15,translate=.1,fliplr=.5,flipud=0.,
                hsv_h=0.,hsv_s=.1,hsv_v=.2,plots=False,cache=False,seed=20261102,
                deterministic=True,project=str(a.out.resolve()),name='run',exist_ok=False,save=True,verbose=False)
    best=Path(model.trainer.best)
    if not best.exists():raise ValueError('No trained checkpoint')
    selected=YOLO(str(best));onnx=Path(selected.export(format='onnx',imgsz=320,opset=17,simplify=False,
                                                     dynamic=False,nms=False,device='cpu'))
    report={'schema':'native-ball-research-v1','epochs_completed':len(history),'history':history,
            'warm_start_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(),'onnx_path':str(onnx),
            'input':'RGB float32 [1,3,320,320] / 255',
            'output':'YOLO xywh plus seven class scores; ball channel only is trained/used',
            'training_class_ids':[0],'temporal_model_trained':False,'independent_match_validation':False,
            'learning_rate':a.learning_rate,'requested_epochs':a.epochs,
            'crop_provenance_sha256':hashlib.sha256((a.data.parent/'crop-provenance.json').read_bytes()).hexdigest(),
            'dataset_fingerprint':qc['dataset_fingerprint'],
            'models_validated':False,'android_fps_validated':False,'elapsed_seconds':time.monotonic()-started}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
