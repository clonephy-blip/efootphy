"""Bounded local PyTorch ball continuation without Trainer or service callbacks."""
import argparse
from copy import deepcopy
import hashlib,json,math,os,random,sys,time
from pathlib import Path

import cv2
import numpy as np


def ball_labels(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        a=np.array(list(map(float,line.split())),np.float32)
        if a.shape!=(5,) or not np.isfinite(a).all() or a[0]!=int(a[0]) or not 0<=a[0]<7 or not np.all((a[1:]>=0)&(a[1:]<=1)) or np.any(a[3:]<=0) or np.any(a[1:3]-a[3:]/2 < -1e-5) or np.any(a[1:3]+a[3:]/2 > 1+1e-5):raise ValueError('Invalid source crop label')
        if a[0]==0:rows.append(a)
    return np.array(rows,np.float32).reshape(-1,5)


def atomic_torch_save(torch,payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_json(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(payload,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','weights','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--learning-rate',type=float,default=.00015)
    p.add_argument('--max-minutes',type=float,default=10.);a=p.parse_args()
    if a.out.exists() or not a.weights.is_file() or not 1<=a.epochs<=4 or not 0<a.learning_rate<=.001 or not 0<a.max_minutes<=15:raise ValueError('Local checkpoint, new output and bounded continuation required')
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',POLARS_MAX_THREADS='2')
    def no_network(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network operations disabled in this local training worker')
    sys.addaudithook(no_network)
    import torch
    from ultralytics import YOLO,settings,__version__
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import ModelEMA
    from .dataset_qc import audit
    from .train import validate_report
    settings.update({'sync':False});torch.set_num_threads(2);cv2.setNumThreads(1)
    seed=20261103;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    provenance=json.loads((a.data.parent/'crop-provenance.json').read_text())
    qc=audit(a.data,expected_images=sum(provenance['tile_counts'].values()));validate_report(qc,'field',research_only=True)
    a.out.mkdir(parents=True);(a.out/'dataset-qc.json').write_text(json.dumps(qc,indent=2))
    network=YOLO(str(a.weights)).model.float();network.args=get_cfg(overrides={**dict(network.args),'data':str(a.data.resolve()),'imgsz':320,'batch':8,'classes':[0],'device':'cpu','lr0':a.learning_rate})
    if network.model[-1].nc!=7 or network.names[0]!='ball':raise ValueError('Expected the seven-channel source model with ball at zero')
    network.criterion=None
    for n,parameter in network.named_parameters():parameter.requires_grad_('.dfl.' not in n)
    ema=ModelEMA(network)
    decay=[];normal=[];bias=[]
    for module in network.modules():
        for n,v in module.named_parameters(recurse=False):
            if not v.requires_grad:continue
            (bias if n=='bias' else normal if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) else decay).append(v)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.0005},{'params':normal,'weight_decay':0.},{'params':bias,'weight_decay':0.}],lr=a.learning_rate,betas=(.9,.999))
    images=sorted((a.data.parent/'train/images').glob('*.jpg'));rng=np.random.default_rng(seed)
    batch_count=math.ceil(len(images)/8);updates=0;history=[];started=time.monotonic()
    for epoch in range(a.epochs):
        network.train();order=rng.permutation(len(images));running={};loss_total=0.
        for offset in range(0,len(order),8):
            tensors=[];classes=[];boxes=[];indices=[]
            for j,number in enumerate(order[offset:offset+8]):
                image=images[number];frame=cv2.imread(str(image));labels=ball_labels(a.data.parent/'train/labels'/(image.stem+'.txt'))
                if frame is None or frame.shape[:2]!=(320,320):raise ValueError('Expected fixed 320px crop')
                if rng.random()<.5:
                    frame=frame[:,::-1].copy();labels[:,1]=1-labels[:,1]
                # Fixed-source crops already contain position jitter; preserve
                # their geometry while varying saturation and brightness.
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,1]*=rng.uniform(.9,1.1);hsv[:,:,2]*=rng.uniform(.8,1.2)
                frame=cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2BGR)
                tensors.append(np.ascontiguousarray(frame[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255.)
                classes.append(labels[:,:1]);boxes.append(labels[:,1:]);indices.append(np.full(len(labels),j,np.float32))
            batch={'img':torch.from_numpy(np.stack(tensors)),'cls':torch.from_numpy(np.concatenate(classes)),
                   'bboxes':torch.from_numpy(np.concatenate(boxes)),'batch_idx':torch.from_numpy(np.concatenate(indices))}
            warmup=min(1.,(updates+1)/max(1,batch_count*.3));lr=a.learning_rate*(1-.9*epoch/4)*warmup
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);loss,items=network.loss(batch);value=loss.sum()
            if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(network.parameters(),10.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite training gradient')
            optimizer.step();ema.update(network);updates+=1;loss_total+=float(value.detach())
            if updates==1:print(json.dumps({'first_optimizer_step_completed':True,'finite_loss':float(value.detach()),'onnxruntime_imported':'onnxruntime' in sys.modules}),flush=True)
            for k,v in items.items():running[k]=running.get(k,0.)+float(v)
            if (offset//8+1)%25==0:print(json.dumps({'epoch':epoch+1,'batches_done':offset//8+1,'batches':batch_count,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-started}),flush=True)
            if time.monotonic()-started>a.max_minutes*60:raise RuntimeError('Budget exceeded before complete epoch; prior snapshots retained')
        record={'epoch':epoch+1,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-started,'mean_batch_total_loss':loss_total/batch_count,'loss_components':{k:v/batch_count for k,v in running.items()}}
        history.append(record);snapshot=a.out/'snapshots'/f'epoch-{epoch+1:03d}'
        selected=deepcopy(ema.ema).float().eval()
        if hasattr(selected,'criterion'):selected.criterion=None
        payload={'model':selected,'ema':None,'optimizer':None,'epoch':epoch,'version':__version__,'train_args':vars(network.args)}
        atomic_torch_save(torch,payload,snapshot/'model.pt')
        atomic_torch_save(torch,{'model':deepcopy(network).float(),'ema_state':ema.ema.state_dict(),'optimizer':optimizer.state_dict(),'updates':updates,'epoch':epoch},snapshot/'resume.pt')
        state={'history':history,'checkpoint_sha256':hashlib.sha256((snapshot/'model.pt').read_bytes()).hexdigest(),'warm_start_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'frozen_validation_sha256':provenance['frozen_validation_sha256'],'training_class_ids':[0],'validation_used_for_optimization':False,'onnxruntime_imported':'onnxruntime' in sys.modules,'python_network_operations_blocked':True,'runner':'local PyTorch, no Ultralytics Trainer/Validator/Exporter callbacks'}
        atomic_json(snapshot/'ready.json',state);atomic_json(a.out/'epoch-history.json',history);print(json.dumps(record),flush=True)
    target=a.out/'run/weights';target.mkdir(parents=True)
    atomic_torch_save(torch,payload,target/'best.pt')
    export_model=deepcopy(selected).float().eval();head=export_model.model[-1];head.export=True;head.format='onnx';head.dynamic=False
    with torch.inference_mode():torch.onnx.export(export_model,torch.zeros((1,3,320,320)),str(target/'best.onnx'),opset_version=17,input_names=['images'],output_names=['output0'],do_constant_folding=True,dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(target/'best.onnx')))
    report={**state,'schema':'local-ball-continuation-v1','epochs_completed':len(history),'seed':seed,'learning_rate':a.learning_rate,'batch':8,'selected_checkpoint':'Final EMA after the predefined epoch count; no validation-based checkpoint selection','onnx_sha256':hashlib.sha256((target/'best.onnx').read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started,'temporal_model_trained':False,'models_validated':False,'android_fps_validated':False}
    report['onnxruntime_imported']='onnxruntime' in sys.modules
    atomic_json(a.out/'manifest.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
