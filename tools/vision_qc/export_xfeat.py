"""Export only XFeat's dense CNN; sparse decoding/matching is NOT in this graph."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--images',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    import cv2
    import torch
    import onnxruntime as ort
    sys.path.insert(0,str(a.repository.resolve()))
    from modules.model import XFeatModel
    torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True;cv2.setNumThreads(1)
    net=XFeatModel().float().eval();net.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True))
    a.out.mkdir(parents=True);model=a.out/'xfeat-dense.onnx'
    names=['descriptors','keypoint_logits','reliability'];sample=torch.zeros(1,3,320,640)
    torch.onnx.export(net,sample,str(model),input_names=['rgb'],output_names=names,opset_version=17,dynamo=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']);rows=[]
    for path in a.images:
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Image failed to decode')
        # Exact raw uint8-array detectAndCompute input convention, cast to float.
        x=np.ascontiguousarray(cv2.resize(frame,(640,320))[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)
        with torch.inference_mode():expected=[t.numpy() for t in net(torch.from_numpy(x))]
        session.run(None,{'rgb':x});times=[]
        for _ in range(5):
            started=time.perf_counter();actual=session.run(None,{'rgb':x});times.append((time.perf_counter()-started)*1000)
        parity={name:{'shape':list(b.shape),'max_abs':float(np.max(np.abs(b-c))),
                      'mean_abs':float(np.mean(np.abs(b-c)))} for name,b,c in zip(names,expected,actual)}
        rows.append({'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'outputs':parity,
                     'ort_cpu_forward_ms_median':float(np.median(times))})
    passed=all(o['max_abs']<=.001 for r in rows for o in r['outputs'].values())
    report={'schema_version':1,'profile':'xfeat_dense_research','model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'bytes':model.stat().st_size,
            'input':{'name':'rgb','shape':[1,3,320,640],'dtype':'float32','range':[0,255],'color':'RGB','resize':'direct'},
            'outputs':names,'opset':17,'parity_tolerance_max_abs':.001,'parity_passed':passed,'records':rows,
            'includes_sparse_postprocessing':False,'includes_matcher':False,'android_import_supported':False,
            'scope':'CPU raw-tensor parity only; no Android device benchmark or minimap bundle',
            'code_source':'https://github.com/verlab/accelerated_features','models_validated':False}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('ONNX parity failed; do not distribute as a validated model')


if __name__=='__main__':main()
