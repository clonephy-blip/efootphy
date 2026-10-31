"""Selected source frames with GT/predicted field boxes; counts are not accuracy."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from .dataset_qc import EXTENSIONS,split_directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True);os.environ['YOLO_CONFIG_DIR']=str((a.out/'cache').resolve());os.environ['ATEN_CPU_CAPABILITY']='default'
    import torch
    import yaml
    from ultralytics import YOLO,settings
    from PIL import Image,ImageDraw
    torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True;settings.update({'sync':False})
    manifest=json.loads((a.bundle/'manifest.json').read_text());model=a.bundle/'model.onnx'
    if manifest['profile']!='field' or hashlib.sha256(model.read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise ValueError('Field profile/checksum mismatch')
    config=yaml.safe_load(a.data.read_text());folder=split_directory(a.data.resolve().parent,config[a.split])
    files=sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not files:raise ValueError('No images')
    indices=sorted({round(i*(len(files)-1)/3) for i in range(4)})
    paths=[files[i] for i in indices];records=[]
    predictions=YOLO(model,task='detect').predict(source=[str(p) for p in paths],imgsz=640,conf=.35,
        device='cpu',batch=1,stream=True,verbose=False)
    canvas=Image.new('RGB',(1280,665*((len(paths)+1)//2)),(12,20,28));pen=ImageDraw.Draw(canvas)
    for i,(path,result) in enumerate(zip(paths,predictions)):
        im=Image.fromarray(result.orig_img[:,:,::-1]);width,height=im.size;draw=ImageDraw.Draw(im)
        gt=collections.Counter();pred=collections.Counter()
        label=folder.parent/'labels'/path.relative_to(folder).with_suffix('.txt')
        for line in label.read_text().splitlines():
            if not line.strip():continue
            cid,cx,cy,bw,bh=map(float,line.split());cid=int(cid);gt[cid]+=1
            box=[(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height]
            draw.rectangle(box,outline='yellow',width=2);draw.text((box[0],box[1]),str(cid),fill='yellow')
        for box in result.boxes:
            cid=int(box.cls.item());pred[cid]+=1;xy=box.xyxy[0].cpu().tolist()
            draw.rectangle(xy,outline='cyan',width=2);draw.text((xy[0],xy[3]),str(cid),fill='cyan')
        name=f'frame-{i:03d}.jpg';im.save(a.out/name)
        x=(i%2)*640;y=(i//2)*665;canvas.paste(im.resize((640,640)),(x,y+25))
        pen.text((x+4,y+4),f'{i} | GT yellow / predicted cyan | threshold .35',fill='white')
        records.append({'image':path.relative_to(a.data.resolve().parent).as_posix(),'preview':name,
            'gt_class_counts':dict(gt),'predicted_class_counts':dict(pred)})
    canvas.save(a.out/'contact.jpg')
    report={'scope':'four deterministically spaced source frames; diagnostic counts, not accuracy or recall',
        'model_sha256':manifest['model_sha256'],'classes':manifest['classes'],'threshold':.35,'records':records,'models_validated':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'images':len(records),'report':str(a.out/'report.json')}))


if __name__=='__main__':main()
