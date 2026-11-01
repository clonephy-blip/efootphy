"""Research-only dense landmark head; fixed encoder, GroupNorm, train-only warps."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .evaluate_landmarks import aggregate, score_points


def transform_points(points, matrix, size):
    """Transform visible targets with the image; clipped targets become missing."""
    out=np.array(points,dtype=np.float32,copy=True)
    xy=np.c_[out[:,:2],np.ones(len(out))]@np.asarray(matrix).T
    good=np.isfinite(xy).all(1)&(np.abs(xy[:,2])>1e-8)
    xy=xy[:,:2]/np.where(good,xy[:,2],1)[:,None]
    good&=(xy[:,0]>=0)&(xy[:,0]<=size)&(xy[:,1]>=0)&(xy[:,1]<=size)&(out[:,2]==2)
    out[:,:2]=xy;out[~good]=0
    return out


def gaussian_targets(points, size=80, sigma=1.):
    """Pixel-centre convention: source pixel x maps to x/stride - 0.5."""
    import torch
    xy=points[...,:2];visible=points[...,2]==2
    axis=torch.arange(size,device=points.device,dtype=points.dtype)
    x=xy[...,0,None,None];y=xy[...,1,None,None]
    target=torch.exp(-((axis[None,None,None,:]-x)**2+(axis[None,None,:,None]-y)**2)/(2*sigma*sigma))
    target=target/target.sum((-2,-1),keepdim=True).clamp_min(1e-12)
    return target,visible


def decode(logits, visibility, original_size=640):
    """Local soft argmax avoids quantizing coordinates to the heatmap cell."""
    import torch
    b,k,h,w=logits.shape;prob=logits.flatten(2).softmax(-1).reshape(b,k,h,w)
    maximum=prob.flatten(2).argmax(-1);px=maximum%w;py=maximum//w
    ax=torch.arange(w,device=prob.device)[None,None,None,:]
    ay=torch.arange(h,device=prob.device)[None,None,:,None]
    mask=(ax-px[...,None,None]).abs()<=2
    mask=mask&((ay-py[...,None,None]).abs()<=2)
    local=prob*mask;mass=local.sum((-2,-1)).clamp_min(1e-12)
    x=((local*ax).sum((-2,-1))/mass+.5)*original_size/w
    y=((local*ay).sum((-2,-1))/mass+.5)*original_size/h
    return torch.stack((x,y,visibility.sigmoid()),-1)


def supervised_loss(logits, visibility, points, pseudo):
    """Unlabeled transferred points have no coordinate OR negative-visibility gradient."""
    import torch
    from torch.nn import functional as F
    targets,mask=gaussian_targets(points,logits.shape[-1],1.)
    spatial=-(targets*F.log_softmax(logits.flatten(2),-1).reshape_as(logits)).sum((-2,-1))
    visibility_mask=torch.stack([mask[i] if value else torch.ones_like(mask[i]) for i,value in enumerate(pseudo)])
    sample_weight=torch.tensor([.5 if value else 1. for value in pseudo],device=logits.device)[:,None]
    coordinate_weight=mask*sample_weight;visibility_weight=visibility_mask*sample_weight
    bce=F.binary_cross_entropy_with_logits(visibility,mask.float(),reduction='none')
    return ((spatial*coordinate_weight).sum()/coordinate_weight.sum().clamp_min(1)
            +(bce*visibility_weight).sum()/visibility_weight.sum().clamp_min(1))


def make_head(channels=(64,128,256), visibility_mlp=True):
    import torch
    from torch import nn
    from torch.nn import functional as F
    def conv(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU())
    class DenseHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project=nn.ModuleList([nn.Conv2d(c,24,1) for c in channels])
            self.mix=nn.Sequential(conv(74,48),conv(48,48))
            self.refine=nn.Sequential(conv(52,32),nn.Conv2d(32,32,1))
            self.visibility=(nn.Sequential(nn.LayerNorm(channels[-1]),nn.Linear(channels[-1],128),nn.SiLU(),nn.Linear(128,32))
                             if visibility_mlp else nn.Linear(channels[-1],32))
        def forward(self,features,rgb):
            n=features[0].shape[-1]
            x=[F.interpolate(p(f),size=(n,n),mode='bilinear',align_corners=False) for p,f in zip(self.project,features)]
            axis=torch.linspace(-1,1,n,device=rgb.device);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            coords=torch.stack((xx,yy))[None].expand(rgb.shape[0],-1,-1,-1)
            x=self.mix(torch.cat(x+[coords],1));x=F.interpolate(x,size=(n*2,n*2),mode='bilinear',align_corners=False)
            small=F.interpolate(rgb,size=x.shape[-2:],mode='bilinear',align_corners=False)
            white=small.min(1,keepdim=True).values-small.max(1,keepdim=True).values*.5
            logits=self.refine(torch.cat((x,small,white),1))
            vis=self.visibility(features[-1].mean((-2,-1)))
            return logits,vis
    return DenseHead()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--max-minutes',type=float,default=8);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--views',type=int,default=1);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--pseudo',type=Path);p.add_argument('--pseudo-root',type=Path)
    p.add_argument('--initial-head',type=Path)
    p.add_argument('--feature-cache',type=Path,help='Reusable FP32 cache, keyed by encoder/runtime/exact image tensor')
    p.add_argument('--teacher-proposals',type=Path,help='Completed train-only teacher_proposals report; experimental additional loss')
    p.add_argument('--teacher-weight',type=float,default=0.)
    p.add_argument('--teacher-unlabeled',action='store_true',help='Filtered teacher images have unknown base targets; do not train previous transferred labels')
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    if not 1<=a.views<=8 or not 1<=a.batch<=32 or a.steps<1 or not 0<a.max_minutes<=30 or not 0<=a.teacher_weight<=.5:raise ValueError('Invalid budget')
    if a.teacher_weight and not a.teacher_proposals:raise ValueError('Teacher loss needs train-only proposals')
    if a.teacher_unlabeled and not (a.teacher_proposals and a.pseudo):raise ValueError('Teacher-only samples need proposal and reference manifests')
    a.out.mkdir(parents=True);os.environ['ATEN_CPU_CAPABILITY']='default'
    import cv2
    import torch
    from PIL import Image,ImageDraw
    from ultralytics import YOLO
    from torch.nn import functional as F
    from .dataset_qc import audit
    qc=audit(a.root/'data.yaml',103)
    if not qc['data_valid'] or qc['kpt_shape']!=[32,3] or qc['split_counts'].get('train')!=93 or qc['split_counts'].get('valid')!=10 or qc['split_counts'].get('test',0)!=0:
        raise ValueError('Expected the complete, valid 93/10 reviewed pose source')
    torch.set_num_threads(a.threads);torch.manual_seed(20261101);cv2.setNumThreads(1)
    torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    rng=np.random.default_rng(20261101)
    encoder_digest=hashlib.sha256(a.weights.read_bytes()).hexdigest();cache=None;cache_parity=None
    initial_checkpoint=torch.load(a.initial_head,map_location='cpu',weights_only=True) if a.initial_head else None
    if initial_checkpoint is not None and initial_checkpoint['encoder_sha256']!=encoder_digest:raise ValueError('Initial head belongs to a different encoder')
    if a.feature_cache:
        from .feature_cache import FeatureCache
        cache=FeatureCache(a.feature_cache,encoder_digest,str(torch.__version__))
    encoder=YOLO(a.weights).model.float().eval()
    for param in encoder.parameters():param.requires_grad_(False)
    captured=[]
    hook=encoder.model[-1].register_forward_pre_hook(lambda _,args:captured.append([x.detach().clone() for x in args[0]]))
    samples=[];validation=[];sources=[];generated=[];teacher_by_hash={}
    if a.teacher_proposals:
        report=json.loads(a.teacher_proposals.read_text())
        if report.get('completed') is not True or report.get('validation_labels_used') is not False or report.get('manual_ground_truth') is not False:
            raise ValueError('Expected completed, train-only experimental teacher proposals')
        if a.pseudo and report.get('review_sha256')!=hashlib.sha256(a.pseudo.read_bytes()).hexdigest():raise ValueError('Teacher/reference review mismatch')
        for record in report['records']:
            if record['target_split']!='train' or record['image_sha256'] in teacher_by_hash:raise ValueError('Invalid/duplicate teacher training entry')
            teacher_by_hash[record['image_sha256']]=record
        teacher_by_hash={k:v for k,v in teacher_by_hash.items() if any(p[2]==2 for p in v['teacher_points_640'])}
        if a.teacher_weight and not teacher_by_hash:raise ValueError('No qualified teacher points; stop instead of running empty distillation')
    started=time.monotonic()
    source_paths=list(sorted((a.root/'train/images').glob('*.jpg')))
    for path in source_paths:
        row=np.asarray(list(map(float,(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text().split())))
        if row.shape!=(101,):raise ValueError('Expected a single pitch annotation')
        sources.append((path,row[5:].reshape(32,3)*[320,320,1],False))
    if a.pseudo:
        pseudo=json.loads(a.pseudo.read_text())
        for item in pseudo['accepted']:
            path=(a.pseudo_root/item['image']).resolve() if a.pseudo_root else Path(item['path'])
            if a.pseudo_root and not path.is_relative_to((a.pseudo_root/'train/images').resolve()):
                raise ValueError('Pseudo-image must stay inside its training image directory')
            if item['target_split']!='train':raise ValueError('Pseudo-label may only enter training')
            if item.get('review_status')!='approved_geometry_review':raise ValueError('Review geometric proposals before training')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['image_sha256']:raise ValueError('Pseudo-image bytes changed')
            if a.teacher_unlabeled and item['image_sha256'] not in teacher_by_hash:continue
            points=np.zeros((32,3)) if a.teacher_unlabeled else np.asarray(item['points'])*[.5,.5,1]
            sources.append((path,points,True))
    def prepare(path,points,split,pseudo=False):
        nonlocal cache_parity
        frame=cv2.imread(str(path))
        if frame is None or frame.shape[:2] not in [(640,640),(1280,1280)]:raise ValueError('Expected audited square source images')
        original=cv2.resize(frame,(640,640));small=cv2.resize(frame,(320,320))
        teacher_points=np.zeros((32,3),np.float32);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in teacher_by_hash:
            if split!='train' or not pseudo:raise ValueError('Teacher proposals are restricted to reviewed pseudo train images')
            teacher_points=np.asarray(teacher_by_hash[digest]['teacher_points_640'],np.float32)*[.5,.5,1]
            if teacher_points.shape!=(32,3) or not np.isfinite(teacher_points).all():raise ValueError('Invalid teacher points')
        for view in range(a.views if split=='train' else 1):
            matrix=np.eye(3)
            if view:
                corners=np.float32([[0,0],[319,0],[319,319],[0,319]])
                jitter=rng.uniform(-12,12,(4,2)).astype(np.float32)
                matrix=cv2.getPerspectiveTransform(corners,corners+jitter)
            target=transform_points(points,matrix,320)
            image=cv2.warpPerspective(small,matrix,(320,320),borderValue=(40,70,40))
            if view:
                gain=rng.uniform(.85,1.15);image=np.clip(image.astype(np.float32)*gain+rng.uniform(-6,6),0,255).astype(np.uint8)
            tensor=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).float()[None]/255
            def compute():
                with torch.no_grad():encoder(tensor)
                return captured.pop()
            if cache:
                features,hit=cache.get(tensor,compute)
                if hit and cache_parity is None:
                    live=compute();cache_parity=max(float((a-b).abs().max()) for a,b in zip(live,features))
                    if cache_parity>1e-6:raise ValueError('Frozen-feature cache parity failed')
            else:features=compute()
            t=target.copy();t[:,:2]=t[:,:2]/4-.5
            teacher_target=transform_points(teacher_points,matrix,320);teacher_target[:,:2]=teacher_target[:,:2]/4-.5
            item={'path':str(path),'features':features,'rgb':tensor,'points':torch.tensor(t,dtype=torch.float32),
                  'target':target*[2,2,1],'frame':original,'pseudo':pseudo,'view':view,
                  'teacher_points':torch.tensor(teacher_target,dtype=torch.float32)}
            (samples if split=='train' else validation).append(item)
            if view:
                generated_dir=a.out/'generated';generated_dir.mkdir(exist_ok=True)
                key=hashlib.sha256(str(path).encode()).hexdigest()[:16]+f'-v{view}'
                cv2.imwrite(str(generated_dir/f'{key}.png'),image)
                label={'source':str(path),'view':view,'matrix_320':matrix.tolist(),'coordinate_space':'320x320 generated pixels',
                       'points':target.tolist(),'image_sha256':hashlib.sha256((generated_dir/f'{key}.png').read_bytes()).hexdigest()}
                (generated_dir/f'{key}.json').write_text(json.dumps(label))
                generated.append(label)
    for i,(path,points,pseudo) in enumerate(sources):
        prepare(path,points,'train',pseudo)
        if i%25==0:print('CACHE',i,len(sources),round(time.monotonic()-started,1),flush=True)
    for path in sorted((a.root/'valid/images').glob('*.jpg')):
        row=np.asarray(list(map(float,(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text().split())))
        prepare(path,row[5:].reshape(32,3)*[320,320,1],'valid')
    hook.remove()
    used_teacher_hashes={hashlib.sha256(path.read_bytes()).hexdigest() for path,_,pseudo in sources if pseudo}
    if teacher_by_hash.keys()-used_teacher_hashes:raise ValueError('Teacher report contains unconsumed training images')
    train_hashes={hashlib.sha256(x['frame'].tobytes()).hexdigest() for x in samples}
    if any(hashlib.sha256(x['frame'].tobytes()).hexdigest() in train_hashes for x in validation):raise ValueError('Train/validation duplicate pixels')
    head=make_head()
    if initial_checkpoint is not None:head.load_state_dict(initial_checkpoint['head'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    cache_seconds=time.monotonic()-started;history=[];step=0;order=[]
    def batch(items):
        return [torch.cat([x['features'][i] for x in items]) for i in range(3)],torch.cat([x['rgb'] for x in items])
    def evaluate(items,save=None):
        head.eval();records=[];per_image=[]
        for offset in range(0,len(items),a.batch):
            subset=items[offset:offset+a.batch];features,rgb=batch(subset)
            with torch.no_grad():pred=decode(*head(features,rgb)).numpy()
            for item,xy in zip(subset,pred):
                points=score_points(item['target'],xy,True,640,640);records.extend(points)
                per_image.append({'image':item['path'],'points':points,'prediction':xy.tolist(),'summary':aggregate(points)})
                if save:
                    preview=Image.fromarray(item['frame'][:,:,::-1]);draw=ImageDraw.Draw(preview)
                    for j,(gx,gy,v) in enumerate(item['target']):
                        if v!=2:continue
                        px,py,confidence=map(float,xy[j]);gx=float(gx);gy=float(gy)
                        draw.ellipse((gx-3,gy-3,gx+3,gy+3),outline='yellow',width=2);draw.text((gx+4,gy),str(j),fill='yellow')
                        draw.line((gx,gy,px,py),fill='red');draw.ellipse((px-3,py-3,px+3,py+3),outline='cyan',width=2)
                    preview.save(save/f'frame-{len(per_image)-1:03d}.jpg')
        return {'summary':aggregate(records),'records':per_image}
    train_started=time.monotonic();last_loss=None;reason='step_limit';train_rng=np.random.default_rng(20261101)
    while step<a.steps:
        if time.monotonic()-train_started>=a.max_minutes*60:reason='time_budget';break
        if len(order)<a.batch:order+=train_rng.permutation(len(samples)).tolist()
        chosen=order[:a.batch];order=order[a.batch:];subset=[samples[i] for i in chosen]
        head.train();features,rgb=batch(subset);points=torch.stack([x['points'] for x in subset])
        logits,vis=head(features,rgb);loss=supervised_loss(logits,vis,points,[x['pseudo'] for x in subset])
        if a.teacher_weight:
            teacher_points=torch.stack([x['teacher_points'] for x in subset])
            loss=loss+a.teacher_weight*supervised_loss(logits,vis,teacher_points,[True]*len(subset))
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10);optimizer.step()
        step+=1;last_loss=float(loss.detach())
        for group in optimizer.param_groups:group['lr']=.0001+.0009*(1+math.cos(math.pi*step/a.steps))/2
        if step%100==0:
            record={'updates':step,'loss':last_loss,'seconds':round(time.monotonic()-train_started,1)}
            history.append(record);(a.out/'progress.json').write_text(json.dumps(record));print('HEATMAP',json.dumps(record),flush=True)
    train_result=evaluate([x for x in samples if x['view']==0 and not x['pseudo']]);valid_result=evaluate(validation,a.out)
    report={'architecture':'frozen YOLO encoder + dense heatmaps with GroupNorm; no pitch detection gate',
            'scope':'source validation only; original 10 validation images excluded from all training views',
            'encoder_sha256':encoder_digest,'seed':20261101,
            'source_dataset_fingerprint':qc['dataset_fingerprint'],'source_audit':qc,
            'optimizer_updates':step,'batch':a.batch,'gradient_accumulation':1,'source_train_images':len(source_paths),
            'pseudo_train_images':sum(x[2] for x in sources),'training_views':len(samples),'views_per_source':a.views,
            'cache_seconds':cache_seconds,'training_seconds':time.monotonic()-train_started,'stop_reason':reason,
            'history':history,'train':train_result,'validation':valid_result,
            'initial_head_sha256':hashlib.sha256(a.initial_head.read_bytes()).hexdigest() if a.initial_head else None,
            'teacher_weight':a.teacher_weight,'teacher_proposals_sha256':hashlib.sha256(a.teacher_proposals.read_bytes()).hexdigest() if a.teacher_proposals else None,
            'pseudo_policy':'filtered_teacher_only_unknown_base_targets' if a.teacher_unlabeled else 'reviewed_transferred_points',
            'teacher_original_points':sum(sum(p[2]==2 for p in r['teacher_points_640']) for r in teacher_by_hash.values()),
            'feature_cache':{'hits':cache.hits,'misses':cache.misses,'reload_parity_max_abs':cache_parity,'precision':'FP32'} if cache else None,
            'point_threshold':.5,'coordinate_space':'640x640 source pixels','models_validated':False,'release_eligible':False,
            'validation_used_for_optimization_or_checkpoint_selection':False}
    torch.save({'head':head.state_dict(),'head_version':2,'encoder_sha256':report['encoder_sha256'],'architecture':report['architecture'],
                'models_validated':False,'optimizer_updates':step},a.out/'heatmap-head.pt')
    (a.out/'generated-labels.json').write_text(json.dumps(generated))
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['optimizer_updates','training_views','cache_seconds','training_seconds']}))
    print('TRAIN',json.dumps(train_result['summary']));print('VALID',json.dumps(valid_result['summary']))


if __name__=='__main__':main()
