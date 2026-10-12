"""Local proposals for human review. No trained detector or cloud API is used.

All coordinates stay in the original captures; 640 processing follows approval.
"""
from pathlib import Path
import base64
import csv
import io
import json
import math
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\Win11Pro\Desktop\capture')
TEMPLATES = HERE / 'dataset_yolo26_boxreviewed_20261008'
OUT = HERE / 'capture_box_review_20261008'


def masks(image):
    a = np.asarray(image.convert('RGB')).astype(np.int16)
    bg = np.median(a.reshape(-1, 3), axis=0)
    dark = bg.mean() < 100
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    blue = ((b-r > (20 if dark else 12)) & (g-r > 5)
            & (b-g > 5) & (np.max(abs(a-bg), axis=2) > 15))
    neutral = (np.max(a, axis=2)-np.min(a, axis=2) < 15)
    black = neutral & ((np.min(a, axis=2) > 125) if dark else (np.max(a, axis=2) < 110))
    return (blue.astype(np.uint8)*255, black.astype(np.uint8)*255, dark)


def blue_without_wires(mask):
    # Seed proposals only. Long rectangle edges may disappear here; a separate
    # visual pass restores the full body before delivering review artifacts.
    lines = np.maximum(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 65), np.uint8)),
                       cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((65, 1), np.uint8)))
    return cv2.subtract(mask, lines)


def boxes_from_mask(mask, kind):
    residual = blue_without_wires(mask) if kind == 'blue' else mask
    connected = cv2.dilate(residual, np.ones((7 if kind == 'blue' else 5,)*2, np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
    out = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        region = (labels[y:y+h, x:x+w] == label)
        original = residual[y:y+h, x:x+w] > 0
        yy, xx = np.where(region & original)
        if len(xx) < (45 if kind == 'blue' else 650):
            continue
        x1, y1, x2, y2 = x+int(xx.min()), y+int(yy.min()), x+int(xx.max())+1, y+int(yy.max())+1
        if min(x2-x1, y2-y1) < (14 if kind == 'blue' else 40):
            continue
        if kind!='blue' and y2-y1<65:continue
        if kind!='blue':
            # Large digits and leaders have few enclosed regions; detailed
            # connector reference drawings have many. Still a proposal rule.
            patch=mask[y1:y2,x1:x2]
            contours,hierarchy=cv2.findContours(patch,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            holes=0 if hierarchy is None else sum(1 for idx,hierarchy_item in enumerate(hierarchy[0])
                if hierarchy_item[3]>=0 and cv2.contourArea(contours[idx])>=3)
            if holes<5:continue
        if max(x2-x1, y2-y1)/min(x2-x1, y2-y1) > (8 if kind=='blue' else 4):
            continue
        pad = 4 if kind == 'blue' else 2
        out.append([max(0,x1-pad), max(0,y1-pad), min(mask.shape[1],x2+pad), min(mask.shape[0],y2+pad)])
    return out


def normalized(mask, size=80):
    yy, xx = np.where(mask > 0)
    if not len(xx): return np.zeros((size,size),np.uint8),1.
    patch=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
    aspect=patch.shape[1]/patch.shape[0]
    # Keep shape proportions; matching also uses aspect separately.
    scale=(size-8)/max(patch.shape)
    w,h=max(1,round(patch.shape[1]*scale)),max(1,round(patch.shape[0]*scale))
    patch=cv2.resize(patch,(w,h),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((size,size),np.uint8)
    canvas[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=patch
    return canvas,aspect


def template_bank():
    sources=json.loads((TEMPLATES/'source_boxes.json').read_text(encoding='utf-8'))
    bank=[]
    for sid,s in sources.items():
        if sid.startswith(('cad_','pdf_extra_')):
            path=TEMPLATES/s['source_capture'];box=s['input_bbox_xyxy_px']
        else:
            path=TEMPLATES/s['image'];box=s['bbox_xyxy_px']
        with Image.open(path) as im:
            im=im.convert('RGB').crop(tuple(map(round,box)))
        blue,black,_=masks(im)
        kind='blue' if np.count_nonzero(blue)>np.count_nonzero(black) else 'reference'
        mask=blue if kind=='blue' else black
        # Apply the same wire removal to blue template and candidate.
        if kind=='blue':mask=blue_without_wires(mask)
        for angle in range(4):
            norm,aspect=normalized(np.rot90(mask,angle))
            binary=(norm>40).astype(np.uint8)
            bank.append(dict(cls=s['class_name'],id=s['class_id'],kind=kind,
                             norm=binary, aspect=aspect, sid=sid, angle=angle*90,
                             distance=cv2.distanceTransform(1-binary,cv2.DIST_L2,3)))
    return bank


def classify(mask,box,kind,bank):
    x1,y1,x2,y2=box
    patch=mask[y1:y2,x1:x2]
    if kind=='blue':patch=blue_without_wires(patch)
    norm,aspect=normalized(patch)
    binary=(norm>40).astype(np.uint8)
    dt=cv2.distanceTransform(1-binary,cv2.DIST_L2,3)
    scores={}
    for t in bank:
        if t['kind']!=kind:continue
        if not binary.any() or not t['norm'].any():continue
        chamfer=(t['distance'][binary>0].mean()+dt[t['norm']>0].mean())/2
        score=float(chamfer+2.5*abs(math.log(aspect/t['aspect'])))
        if t['cls'] not in scores or score<scores[t['cls']]['distance']:
            scores[t['cls']]={'class_name':t['cls'],'class_id':t['id'],
                              'distance':round(score,3),'template_id':t['sid']}
    return sorted(scores.values(),key=lambda x:x['distance'])[:3]


def propose():
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    bank=template_bank()
    captures=[]
    for i,path in enumerate(sorted(SOURCE.glob('*.png')),1):
        with Image.open(path) as im:im=im.convert('RGB')
        blue,black,dark=masks(im)
        objects=[]
        for kind,mask in [('blue',blue),('reference',black)]:
            for box in boxes_from_mask(mask,kind):
                candidates=classify(mask,box,kind,bank)
                if not candidates:continue
                objects.append({'class_name':candidates[0]['class_name'],
                                'class_id':candidates[0]['class_id'],
                                'bbox_xyxy_px':box,'kind':kind,
                                'candidates':candidates,'status':'unreviewed',
                                'touches_edge':box[0]<=1 or box[1]<=1 or box[2]>=im.width-1 or box[3]>=im.height-1})
        objects.sort(key=lambda b:(b['bbox_xyxy_px'][1],b['bbox_xyxy_px'][0]))
        for j,obj in enumerate(objects,1):obj['box_id']=f'{i:03d}-{j:02d}'
        name=f'capture_{i:03d}.png'
        shutil.copy2(path,OUT/'images'/name)
        captures.append({'capture_id':f'{i:03d}','filename':name,'original_name':path.name,
                         'width':im.width,'height':im.height,'background':'dark' if dark else 'light',
                         'status':'unreviewed','objects':objects})
    data={'version':1,'method':'local color segmentation and template comparison; NOT trained YOLO',
          'coordinate_space':'original_capture_pixels','review_required':True,
          'classes':json.loads((TEMPLATES/'classes.json').read_text()),'captures':captures}
    (OUT/'proposals.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'captures':len(captures),'boxes':sum(len(c['objects']) for c in captures),
                      'empty_captures':[c['capture_id'] for c in captures if not c['objects']]},ensure_ascii=True))


if __name__=='__main__':propose()
