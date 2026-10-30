"""ONNX diagnostic replay for selected Mobile frames; no accuracy claim without labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def letterbox(image, size=320):
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (size-resized_width)//2, (size-resized_height)//2
    canvas = np.full((size,size,3),114,dtype=np.uint8)
    canvas[top:top+resized_height,left:left+resized_width] = cv2.resize(image,(resized_width,resized_height))
    tensor = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    return tensor, (scale,left,top,width,height)


def decode(raw, transform, threshold=.35):
    import numpy as np
    scale,left,top,width,height = transform
    if raw.shape != (1,7,2100) or not np.isfinite(raw).all():
        raise ValueError('Expected finite FP32 minimap [1,7,2100] output')
    candidates=[]
    for row in raw[0].T:
        cx,cy,w,h=map(float,row[:4])
        if w<=0 or h<=0 or not 0 <= (cx-left)/scale <= width or not 0 <= (cy-top)/scale <= height:
            continue
        box=[max(0,min(1,(cx-w/2-left)/scale/width)),max(0,min(1,(cy-h/2-top)/scale/height)),
             max(0,min(1,(cx+w/2-left)/scale/width)),max(0,min(1,(cy+h/2-top)/scale/height))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        for label,score in enumerate(row[4:]):
            if threshold<=score<=1:candidates.append({'box':box,'label':label,'score':float(score)})
    kept=[]
    for candidate in sorted(candidates,key=lambda d:-d['score']):
        if not any(k['label']==candidate['label'] and iou(k['box'],candidate['box'])>.45 for k in kept):
            kept.append(candidate)
            if len(kept)==64:break
    return kept


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union>0 else 0


def coarse_scene(image):
    # Mirror the current Android research gate. This is not a learned scene classifier.
    import numpy as np
    h,w=image.shape[:2];grid=image[:h//4*4:4,:w//4*4:4].astype(np.int16)
    b,g,r=grid[:,:,0],grid[:,:,1],grid[:,:,2];gh,gw=r.shape
    yy,xx=np.indices(r.shape)
    grass=(g>r+14)&(g>b+18)&(g>=55)&(g<=190)
    formation=grass&(xx>gw*.25)&(xx<gw*.92)&(yy>=int(gh*.10))&(yy<=int(gh*.88))
    sidebar=(xx<gw*.22)&(r<45)&(g<55)&(b>=35)&(b<=110)
    total=gw*gh
    if sidebar.sum()/(total*.22)>.45 and formation.sum()>total*.18:return 'gameplan'
    return 'live' if grass.sum()/total>.38 else 'unknown'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True,help='Unpacked bundle directory')
    parser.add_argument('--frames',type=Path,required=True,help='JSON list with image, match and reviewed scene')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--capture-sides',type=int,nargs='+',default=[1280,2340])
    parser.add_argument('--roi',type=float,nargs=4,default=[.44188,.80000,.55898,.95185])
    args=parser.parse_args()
    import cv2
    import onnxruntime as ort
    from PIL import Image,ImageDraw
    if args.out.exists():raise ValueError('Output directory must be new')
    if not 0<=args.roi[0]<args.roi[2]<=1 or not 0<=args.roi[1]<args.roi[3]<=1:
        raise ValueError('ROI must be normalized left,top,right,bottom')
    if any(side<32 for side in args.capture_sides):raise ValueError('Invalid capture side')
    args.out.mkdir(parents=True)
    manifest=json.loads((args.bundle/'manifest.json').read_text());model=args.bundle/'model.onnx'
    with model.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=manifest['model_sha256']:raise ValueError('Model digest mismatch')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    records=json.loads(args.frames.read_text());results=[]
    root=args.frames.resolve().parent
    colors=[(0,220,255),(255,255,0),(255,255,255)]
    for record in records:
        path=(root/record['image']).resolve()
        if not path.is_relative_to(root):raise ValueError('External image path')
        original=cv2.imread(str(path))
        if original is None:raise ValueError('Image could not be decoded')
        for side in args.capture_sides:
            scale=min(1,side/max(original.shape[:2]))
            size=(round(original.shape[1]*scale),round(original.shape[0]*scale))
            image=cv2.resize(original,size) if scale<1 else original
            h,w=image.shape[:2];l,t,r,b=[int(v*d) for v,d in zip(args.roi,[w,h,w,h])]
            crop=image[t:b,l:r];tensor,transform=letterbox(crop)
            raw=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            boxes=decode(raw,transform)
            players=[]
            for d in sorted((d for d in boxes if d['label'] in [1,2]),key=lambda d:-d['score']):
                if not any(iou(p['box'],d['box'])>.55 for p in players):players.append(d)
            preview=cv2.resize(crop,(548,328));ph,pw=preview.shape[:2]
            for detection in boxes:
                x1,y1,x2,y2=detection['box'];color=colors[detection['label']]
                cv2.rectangle(preview,(int(x1*pw),int(y1*ph)),(int(x2*pw),int(y2*ph)),color,1)
            name=path.stem+f'_capture{side}.jpg';cv2.imwrite(str(args.out/name),preview)
            results.append({**record,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'capture_size':list(size),'roi_pixels':[l,t,r-l,b-t],'coarse_scene':coarse_scene(image),
                'detections':boxes,'player_candidates_after_dedup':len(players),
                'classes':{name:sum(d['label']==i for d in boxes) for i,name in enumerate(manifest['classes'])},
                'guidance_enabled':False,'preview':name})
    report={'model_sha256':digest,'scope':'diagnostic selected frames; one match; no player/ID ground truth or accuracy estimate',
            'detection_threshold':.35,'nms_iou':.45,'models_validated':False,'results':results}
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    columns=min(2,len(args.capture_sides))
    canvas=Image.new('RGB',(560*columns,360*math.ceil(len(results)/columns)),(12,18,25));draw=ImageDraw.Draw(canvas)
    for i,record in enumerate(results):
        x=(i%columns)*560;y=(i//columns)*360
        with Image.open(args.out/record['preview']) as image:canvas.paste(image,(x,y+24))
        draw.text((x+4,y+4),f"{record['image']} | {record['capture_size'][0]}px | n={record['player_candidates_after_dedup']}",fill='white')
    canvas.save(args.out/'contact.jpg')
    print(json.dumps({'frames':len(records),'variants':len(results),'report':str(args.out/'report.json')}))


if __name__=='__main__':main()
