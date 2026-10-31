"""Source-label support and centre-point consistency, not model accuracy."""
import argparse
import json
import math
from pathlib import Path

from .dataset_qc import audit


def intersection(a,b,c,d):
    rx,ry=b[0]-a[0],b[1]-a[1];sx,sy=d[0]-c[0],d[1]-c[1]
    denominator=rx*sy-ry*sx
    if abs(denominator)<1e-10:return None
    t=((c[0]-a[0])*sy-(c[1]-a[1])*sx)/denominator
    return [a[0]+t*rx,a[1]+t*ry]


def diagnose(data,out):
    from PIL import Image,ImageDraw
    data=Path(data).resolve();out=Path(out)
    qc=audit(data,103)
    if not qc['data_valid'] or qc['kpt_shape']!=[32,3]:raise ValueError('Invalid 32-point export')
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    counts={s:[0]*32 for s in ['train','valid','test']};records=[];checks=[]
    for item in qc['images']:
        image=data.parent/item['image'];label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        rows=[x for x in label.read_text().splitlines() if x.strip()]
        if len(rows)!=1:raise ValueError('Expected one pitch')
        points=[list(map(float,rows[0].split()))[5+3*i:8+3*i] for i in range(32)]
        visible={i for i,p in enumerate(points) if p[2]==2}
        for i in visible:counts[item['split']][i]+=1
        records.append({'image':item['image'],'split':item['split'],'points':points,'visible':sorted(visible)})
        if {13,15,16,30,31}<=visible:
            xy=intersection(points[13],points[16],points[30],points[31])
            if xy:
                error=math.hypot((xy[0]-points[15][0])*item['size'][0],(xy[1]-points[15][1])*item['size'][1])
                checks.append({'image':item['image'],'error_px':error})
    values=sorted(x['error_px'] for x in checks)
    def percentile(q):
        if not values:return None
        p=q*(len(values)-1);lo=math.floor(p);hi=math.ceil(p)
        return values[lo]+(values[hi]-values[lo])*(p-lo)
    uncovered=set(range(32));selected=[];available=records.copy()
    while uncovered and available:
        item=max(available,key=lambda x:len(uncovered&set(x['visible'])))
        if not uncovered&set(item['visible']):break
        selected.append(item);uncovered-=set(item['visible']);available.remove(item)
    report={'scope':'source labels only; counts and geometric consistency are not detector accuracy',
        'dataset_fingerprint':qc['dataset_fingerprint'],'split_counts':qc['split_counts'],
        'visible_v2_counts':counts,'unsupported_training_ids':[i for i,n in enumerate(counts['train']) if n==0],
        'unsupported_validation_ids':[i for i,n in enumerate(counts['valid']) if n==0],
        'centre_intersection':{'cases':len(checks),'median_px':percentile(.5),'p95_px':percentile(.95),
                               'max_px':max(values) if values else None,'records':checks},
        'review_selection':[{'image':x['image'],'visible_ids':x['visible']} for x in selected],
        'schema_status':'single review, provisional; world coordinates not verified',
        'observed_semantics':{'5':'left near touchline corner; can be outside view',
             '13':'far halfway touchline intersection','14':'far centre-circle / halfway intersection',
             '15':'centre spot','16':'near halfway touchline intersection',
             '30':'left centre-circle / pitch lengthwise centreline intersection',
             '31':'right centre-circle / pitch lengthwise centreline intersection'},
        'physical_pitch_dimensions_verified':False,'camera_calibration_enabled':False,'models_validated':False}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(640*min(2,len(selected)),665*math.ceil(len(selected)/2)),(12,20,28));draw=ImageDraw.Draw(canvas)
    for k,item in enumerate(selected):
        with Image.open(data.parent/item['image']) as src:im=src.convert('RGB').resize((640,640))
        pen=ImageDraw.Draw(im)
        for i in item['visible']:
            x,y,_=item['points'][i];x*=640;y*=640
            pen.ellipse((x-3,y-3,x+3,y+3),outline='yellow',width=2);pen.text((x+4,y),str(i),fill='yellow')
        x=(k%2)*640;y=(k//2)*665;canvas.paste(im,(x,y+25));draw.text((x+4,y+4),f'{k} | source keypoint IDs',fill='white')
    canvas.save(out/'review-contact.jpg')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();r=diagnose(a.data,a.out);print(json.dumps({k:r[k] for k in ['split_counts','unsupported_validation_ids','centre_intersection']},indent=2))
