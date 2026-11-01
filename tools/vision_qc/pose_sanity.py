"""Bounded training-only pose memorization check. Never a generalization score."""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .evaluate_landmarks import aggregate,score_points
from .inspect_frames import iou,letterbox


def checked_samples(root, audit_file, selection):
    root=Path(root).resolve();report=json.loads(Path(audit_file).read_text())
    if report['task']!='pose' or report['kpt_shape']!=[32,3] or not report['data_valid']:
        raise ValueError('A valid 32-point source audit is required')
    indexed={r['image']:r for r in report['images']};result=[]
    if not selection or len(selection)>8 or len(set(selection))!=len(selection):
        raise ValueError('Choose 1..8 unique training images')
    for relative in selection:
        path=(root/relative).resolve();record=indexed.get(relative)
        if not record or record['split']!='train' or not path.is_relative_to(root):
            raise ValueError('Sanity selection must come from the audited training split')
        label=path.parent.parent/'labels'/path.with_suffix('.txt').name
        for file,expected in [(path,record['sha256']),(label,record['label_sha256'])]:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
                raise ValueError('Selected bytes differ from the source audit')
        rows=[x for x in label.read_text().splitlines() if x.strip()]
        row=list(map(float,rows[0].split())) if len(rows)==1 else []
        if len(row)!=101 or not all(math.isfinite(x) for x in row) or row[0]!=0:
            raise ValueError('Expected one finite 32-point pitch instance')
        if not all(0<=x<=1 for x in row[1:5]) or min(row[3:5])<=0:
            raise ValueError('Invalid normalized pitch box')
        if any(not (0<=row[5+3*i]<=1 and 0<=row[6+3*i]<=1 and row[7+3*i] in [0,1,2]) for i in range(32)):
            raise ValueError('Invalid normalized keypoint/visibility')
        if sum(row[7+3*i]==2 for i in range(32))<4:
            raise ValueError('Sanity image needs at least four visible points')
        result.append({'image':relative,'path':path,'row':row,'image_sha256':record['sha256'],
                       'label_sha256':record['label_sha256']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--audit',required=True,type=Path)
    p.add_argument('--selection',required=True,type=Path);p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--max-minutes',type=float,default=4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--sigma',type=float,default=.1);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--refine-after',type=int,help='Training-only sigma/lr refinement after this many updates')
    p.add_argument('--freeze-bn-stats',action='store_true',help='Keep BatchNorm running moments fixed; affine parameters still learn')
    p.add_argument('--head-only-cache',action='store_true',help='Learn cv4 only; cache frozen FPN features, freeze all BN moments')
    a=p.parse_args()
    if a.steps<1 or a.threads<1 or not math.isfinite(a.max_minutes) or a.max_minutes<=0 or not 0<a.sigma<=.25 or not 0<a.lr<=.01:
        p.error('Invalid step/time/thread/loss settings')
    if a.refine_after is not None and not 0<a.refine_after<a.steps:
        p.error('Refinement step must lie inside the optimization budget')
    samples=checked_samples(a.root,a.audit,json.loads(a.selection.read_text()))
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True)
    os.environ['ATEN_CPU_CAPABILITY']='default';os.environ['POLARS_MAX_THREADS']=str(a.threads)
    os.environ['YOLO_CONFIG_DIR']=str((a.out/'cache').resolve())
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from ultralytics import YOLO,settings
    from ultralytics.cfg import get_cfg
    cv2.setNumThreads(1);settings.update({'sync':False})
    torch.set_num_threads(a.threads);torch.manual_seed(20261101)
    torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    loaded=YOLO(a.weights);net=loaded.model
    if net.model[-1].kpt_shape!=[32,3]:raise ValueError('Checkpoint is not a 32-point pose model')
    cfg=get_cfg();vars(cfg).update(loaded.ckpt.get('train_args',{}));net.args=cfg
    # EMA checkpoints have requires_grad=False. Re-enable learning explicitly.
    for name,param in net.named_parameters():param.requires_grad_(not name.endswith('.dfl.conv.weight'))
    head=net.model[-1]
    if a.head_only_cache:
        head_params={id(x) for x in head.cv4.parameters()}
        for param in net.parameters():param.requires_grad_(id(param) in head_params)
    net.criterion=net.init_criterion();net.criterion.keypoint_loss.sigmas=torch.full((32,),a.sigma)
    optimizer=torch.optim.AdamW([x for x in net.parameters() if x.requires_grad],lr=a.lr,weight_decay=.0005)
    original_updates=loaded.ckpt.get('updates')
    prior_debug_updates=int(loaded.ckpt.get('optimization_steps',0))
    cfg.lr0=a.lr
    probe=next(param for name,param in net.named_parameters() if 'cv4' in name and param.requires_grad)
    probe_before=probe.detach().clone()
    bn_before={name:buffer.detach().clone() for name,buffer in net.named_buffers()
               if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name}
    def frozen_digest():
        digest=hashlib.sha256()
        for name,param in net.named_parameters():
            if not param.requires_grad:digest.update(name.encode());digest.update(param.detach().numpy().tobytes())
        return digest.hexdigest()
    original_frozen_digest=frozen_digest()
    for item in samples:
        frame=cv2.imread(str(item['path']));tensor,transform=letterbox(frame,640)
        scale,left,top,width,height=transform;row=np.asarray(item['row'],np.float32)
        points=row[5:].reshape(32,3).copy();points[:,0]=(points[:,0]*width*scale+left)/640;points[:,1]=(points[:,1]*height*scale+top)/640
        box=row[1:5]*[width*scale,height*scale,width*scale,height*scale]
        box[:2]+=[left,top];box/=640
        item.update({'frame':frame,'transform':transform,'target':row[5:].reshape(32,3)*[width,height,1],
            'tensor':torch.from_numpy(tensor),'bboxes':torch.tensor(box[None],dtype=torch.float32),
            'keypoints':torch.from_numpy(points[None])})
    cache_started=time.monotonic()
    if a.head_only_cache:
        net.eval();captured=[]
        def capture(module,inputs):captured.append([x.detach().clone() for x in inputs[0]])
        hook=head.register_forward_pre_hook(capture)
        try:
            for item in samples:
                # Regular no-grad tensors must be usable as inputs to autograd later.
                with torch.no_grad():net(item['tensor'])
                item['features']=captured.pop()
        finally:hook.remove()
    def cache_parity():
        if not a.head_only_cache:return None
        net.eval();errors=[]
        with torch.inference_mode():
            for item in samples:
                cached=head(list(item['features']))[0];full=net(item['tensor'])[0]
                errors.append(float((cached-full).abs().max()))
        if max(errors)>.001:raise RuntimeError('Cached head differs from full-network inference')
        return max(errors)
    initial_cache_parity=cache_parity();cache_preparation_seconds=time.monotonic()-cache_started
    history=[];last_predictions=[]
    def evaluate(step):
        net.eval();all_points=[];records=[];predictions=[]
        for item in samples:
            with torch.inference_mode():raw=head(list(item['features'])) if a.head_only_cache else net(item['tensor'])
            if isinstance(raw,(list,tuple)):raw=raw[0]
            raw=raw.detach().cpu().numpy()
            if raw.shape!=(1,101,8400) or not np.isfinite(raw).all():raise RuntimeError('Invalid raw output')
            pred=raw[0,:,int(raw[0,4].argmax())];scale,left,top,width,height=item['transform']
            cx,cy,bw,bh=map(float,pred[:4]);box=[(cx-bw/2-left)/scale,(cy-bh/2-top)/scale,(cx+bw/2-left)/scale,(cy+bh/2-top)/scale]
            gcx,gcy,gw,gh=np.asarray(item['row'][1:5])*[width,height,width,height]
            overlap=float(iou(box,[gcx-gw/2,gcy-gh/2,gcx+gw/2,gcy+gh/2]))
            matched=bool(float(pred[4])>=.25 and overlap>=.5)
            points=pred[5:].reshape(32,3).copy();points[:,0]=(points[:,0]-left)/scale;points[:,1]=(points[:,1]-top)/scale
            scored=score_points(item['target'],points,matched,width,height)
            all_points.extend(scored);records.append({'image':item['image'],'object_matched':matched,'box_iou':overlap,'summary':aggregate(scored)})
            predictions.append(points)
        summary=aggregate(all_points);record={'optimizer_updates':step,'elapsed_seconds':round(time.monotonic()-started,3),
            'summary':summary,'images':records}
        history.append(record)
        temp=a.out/'progress.json.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(a.out/'progress.json')
        print('POSE_SANITY '+json.dumps({k:record[k] for k in ['optimizer_updates','elapsed_seconds','summary']}),flush=True)
        return summary,predictions
    started=time.monotonic();summary,last_predictions=evaluate(0);consecutive=0;steps=0;reason='step_limit';first_gradient_norm=None
    try:
        for step in range(1,a.steps+1):
            if time.monotonic()-started>=a.max_minutes*60:reason='time_budget';break
            if a.refine_after is not None and step==a.refine_after+1:
                net.criterion.keypoint_loss.sigmas.fill_(1/32)
                for group in optimizer.param_groups:group['lr']=.0001
            item=samples[(step-1)%len(samples)]
            if a.head_only_cache:net.eval();head.train()
            else:net.train()
            if a.freeze_bn_stats or a.head_only_cache:
                for module in net.modules():
                    if isinstance(module,torch.nn.BatchNorm2d):module.eval()
            batch={'img':item['tensor'],'batch_idx':torch.zeros(1),'cls':torch.zeros((1,1)),
                   'bboxes':item['bboxes'].clone(),'keypoints':item['keypoints'].clone()}
            optimizer.zero_grad(set_to_none=True)
            loss,items=net.criterion(head(list(item['features'])),batch) if a.head_only_cache else net(batch)
            total=loss.sum()
            if not torch.isfinite(total):raise RuntimeError('Non-finite loss')
            total.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),10)
            if not torch.isfinite(gradient):raise RuntimeError('Non-finite gradient')
            if first_gradient_norm is None:first_gradient_norm=float(gradient)
            optimizer.step();steps=step
            if step%10==0 or step==a.steps:
                summary,last_predictions=evaluate(step)
                median=summary['coordinate_error_on_detected_objects']['median_px']
                consecutive=consecutive+1 if summary['pck']['5']>=.95 and median is not None and median<=3 else 0
                if consecutive>=2:reason='memorization_gate';break
    except KeyboardInterrupt:reason='interrupted'
    if history[-1]['optimizer_updates']!=steps:summary,last_predictions=evaluate(steps)
    elapsed=time.monotonic()-started
    buffers=dict(net.named_buffers())
    bn_change=max((float((buffers[name]-before).abs().max()) for name,before in bn_before.items()),default=0.)
    if (a.freeze_bn_stats or a.head_only_cache) and bn_change!=0:raise RuntimeError('Frozen BatchNorm buffers changed')
    if a.head_only_cache and frozen_digest()!=original_frozen_digest:raise RuntimeError('Frozen model weights changed')
    final_cache_parity=cache_parity()
    # Store weights only: this debug optimizer must never be mistaken for a resume.
    saved=copy.deepcopy(net).float().eval()
    if hasattr(saved,'criterion'):delattr(saved,'criterion')
    checkpoint={'model':saved,'ema':None,'epoch':-1,'optimizer':None,'train_args':vars(cfg),
        'debug_overfit_only':True,'optimization_steps':prior_debug_updates+steps,'models_validated':False}
    temporary=a.out/'debug-weights.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(a.out/'debug-weights.pt')
    for i,(item,points) in enumerate(zip(samples,last_predictions)):
        im=Image.fromarray(cv2.cvtColor(item['frame'],cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
        for k,(gx,gy,v) in enumerate(item['target']):
            if v!=2:continue
            px,py,_=map(float,points[k]);gx=float(gx);gy=float(gy)
            draw.ellipse((gx-3,gy-3,gx+3,gy+3),outline='yellow',width=2);draw.text((gx+4,gy),str(k),fill='yellow')
            if math.isfinite(px) and math.isfinite(py):
                draw.line((gx,gy,px,py),fill='red');draw.ellipse((px-3,py-3,px+3,py+3),outline='cyan',width=2)
        im.save(a.out/f'frame-{i:03d}.jpg')
    report={'scope':'training-only memorization diagnostic; never source validation/test or Mobile accuracy',
        'images':[{'image':x['image'],'image_sha256':x['image_sha256'],'label_sha256':x['label_sha256']} for x in samples],
        'source_checkpoint_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'source_checkpoint_optimizer_updates':original_updates,
        'source_checkpoint_debug_updates':prior_debug_updates,'debug_optimizer_updates_total':prior_debug_updates+steps,
        'optimizer_updates':steps,'elapsed_seconds':elapsed,'learning_rate':a.lr,'sigma':a.sigma,'threads':a.threads,
        'batch':1,'gradient_accumulation':1,'augmentation':False,'ema_used':False,'seed':20261101,
        'batchnorm_running_statistics_frozen':a.freeze_bn_stats or a.head_only_cache,'batchnorm_buffer_max_change':bn_change,
        'head_only_cache':a.head_only_cache,'frozen_weights_unchanged':frozen_digest()==original_frozen_digest,
        'cache_parity_before':initial_cache_parity,'cache_parity_after':final_cache_parity,
        'feature_cache_bytes':sum(x.numel()*x.element_size() for s in samples for x in s.get('features',[])),
        'cache_preparation_seconds':cache_preparation_seconds,
        'refinement':None if a.refine_after is None else {'after_update':a.refine_after,'sigma':1/32,'lr':.0001},
        'first_gradient_norm':first_gradient_norm,'coordinate_head_parameter_change_l2':float((probe.detach()-probe_before).norm()),
        'stop_reason':reason,'memorization_gate_passed':reason=='memorization_gate','summary':summary,'history':history,
        'models_validated':False,'release_eligible':False,'debug_weights_only':True}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['optimizer_updates','stop_reason','summary']},indent=2))


if __name__=='__main__':main()
