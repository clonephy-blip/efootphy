"""Run an experimental dense landmark checkpoint on one image, outside the Android app."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def load_head(path, encoder_sha256):
    import torch
    from .heatmap_pose import make_head
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['encoder_sha256']!=encoder_sha256:raise ValueError('Head belongs to a different encoder')
    head=make_head(visibility_mlp='visibility.weight' not in checkpoint['head'])
    head.load_state_dict(checkpoint['head']);return head.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',type=Path,required=True);p.add_argument('--head',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    os.environ['ATEN_CPU_CAPABILITY']='default'
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from ultralytics import YOLO
    from .heatmap_pose import decode
    torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True;cv2.setNumThreads(1)
    digest=hashlib.sha256(a.encoder.read_bytes()).hexdigest();head=load_head(a.head,digest)
    encoder=YOLO(a.encoder).model.float().eval();frame=cv2.imread(str(a.image))
    if frame is None:raise ValueError('Image did not decode')
    height,width=frame.shape[:2];small=cv2.resize(frame,(320,320))
    tensor=torch.from_numpy(np.ascontiguousarray(small[:,:,::-1].transpose(2,0,1))).float()[None]/255
    captured=[];hook=encoder.model[-1].register_forward_pre_hook(lambda _,args:captured.append(list(args[0])))
    started=time.perf_counter()
    try:
        with torch.no_grad():encoder(tensor);prediction=decode(*head(captured.pop(),tensor))[0].numpy()
    finally:hook.remove()
    milliseconds=(time.perf_counter()-started)*1000;prediction[:,:2]*=[width/640,height/640]
    a.out.mkdir(parents=True);preview=Image.fromarray(frame[:,:,::-1]);draw=ImageDraw.Draw(preview)
    for i,(x,y,score) in enumerate(prediction):
        if score>=.5:
            x=float(x);y=float(y);draw.ellipse((x-4,y-4,x+4,y+4),outline='cyan',width=2);draw.text((x+5,y),str(i),fill='cyan')
    preview.save(a.out/'prediction.jpg')
    report={'scope':'experimental single-image prediction; no Android or Mobile accuracy certification',
            'encoder_sha256':digest,'head_sha256':hashlib.sha256(a.head.read_bytes()).hexdigest(),
            'image_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),'image_size':[width,height],
            'coordinate_space':'original image pixels','keypoints':prediction.tolist(),'point_threshold':.5,
            'cpu_forward_ms_first_run':milliseconds,'models_validated':False,'release_eligible':False}
    (a.out/'prediction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
