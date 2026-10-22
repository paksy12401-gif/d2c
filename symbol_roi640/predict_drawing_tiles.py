"""Detect at several PDF window sizes without shrinking the whole drawing.

The original PDF geometry is not changed. Window detections map back to the
rendered page and PDF points. This output is detection candidates, not CAD
blocks or certified accuracy on a completely annotated PDF.
"""
from pathlib import Path
from collections import Counter
import argparse
import html
import json
import os

import train_yolo26_noaug as local
from ultralytics import YOLO
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def starts(length,window):
    end=max(0,length-window)
    return sorted(set([*range(0,end+1,max(1,round(window*.75))),end]))


def suppress(rows,iou_threshold=.45):
    keep=[]
    for cls in sorted({r['class_id'] for r in rows}):
        selected=[i for i,r in enumerate(rows) if r['class_id']==cls]
        selected.sort(key=lambda i:rows[i]['confidence'],reverse=True)
        while selected:
            i=selected.pop(0);keep.append(rows[i])
            a=np.asarray(rows[i]['raster_bbox_xyxy'])
            if not selected:break
            b=np.asarray([rows[j]['raster_bbox_xyxy'] for j in selected])
            wh=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2]))
            intersection=wh[:,0]*wh[:,1]
            area=(a[2]-a[0])*(a[3]-a[1]);areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
            overlap=intersection/np.maximum(area+areas-intersection,1e-9)
            selected=[j for j,v in zip(selected,overlap) if v<iou_threshold]
    return sorted(keep,key=lambda r:(r['class_id'],r['raster_bbox_xyxy'][1],r['raster_bbox_xyxy'][0]))


def run(image_path,weights,out,dpi=144,confidence=.25,window_sizes=(320,640,1280,2560),classes=None):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    # Trusted local PDF render: permit its known 119 MP page dimensions.
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(image_path) as source:
        image=source.convert('RGB')
    image.save(out/'drawing_source.png')
    w,h=image.size
    model=YOLO(str(weights));model.add_callback('on_predict_start',local.disable_external_callbacks)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    rows=[];tiles=[];metadata=[];processed=0
    def flush():
        nonlocal processed
        predictions=model.predict(source=tiles,imgsz=640,batch=16,device=0,conf=confidence,
                                  classes=classes,iou=.7,augment=False,verbose=False)
        for result,(x,y,window) in zip(predictions,metadata):
            factor=window/640
            for b,c,score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                # Ignore clipped internal window detections: overlap and larger
                # windows offer another chance to see the complete symbol.
                if ((b[0]<=3 and x>0) or (b[1]<=3 and y>0) or
                    (b[2]>=637 and x+window<w) or (b[3]>=637 and y+window<h)):
                    continue
                box=[max(0,float(b[0]*factor+x)),max(0,float(b[1]*factor+y)),
                     min(w,float(b[2]*factor+x)),min(h,float(b[3]*factor+y))]
                rows.append({'class_id':int(c),'class_name':model.names[int(c)],'confidence':float(score),
                             'raster_bbox_xyxy':box,'pdf_bbox_xyxy_points':[v*72/dpi for v in box],
                             'source_window_px':window})
        processed+=len(tiles);tiles.clear();metadata.clear()
        if processed%160==0:print('processed_tiles',processed,flush=True)
    # 320 preserves/enlarges tiny glyphs; 640 preserves native raster detail;
    # 1280/2560 provide context for tall connector reference pictures.
    for window in window_sizes:
        for y in starts(h,window):
            for x in starts(w,window):
                tile=Image.new('RGB',(window,window),'white')
                tile.paste(image.crop((x,y,min(w,x+window),min(h,y+window))),(0,0))
                if np.count_nonzero(np.min(np.asarray(tile),axis=2)<248)<24:
                    continue  # truly empty margins, without removing faint linework
                tiles.append(tile.resize((640,640),Image.Resampling.LANCZOS))
                metadata.append((x,y,window))
                if len(tiles)==16:flush()
    if tiles:flush()
    detections=suppress(rows)
    annotated=image.copy();draw=ImageDraw.Draw(annotated)
    colors=['#e60049','#0bb4ff','#50e991','#e6d800','#9b19f5','#ffa300']
    for i,r in enumerate(detections):
        r['detection_id']=f'D{i+1:04d}';box=r['raster_bbox_xyxy'];color=colors[r['class_id']%len(colors)]
        draw.rectangle(box,outline=color,width=3)
        draw.text((box[0],max(0,box[1]-22)),f"{r['class_name']} {r['confidence']:.2f}",font=font,fill=color)
    annotated.save(out/'drawing_detections.png')
    for name in ('drawing_source.png','drawing_detections.png'):
        with (out/name).open('r+b') as saved_image:
            saved_image.flush();os.fsync(saved_image.fileno())
    summary={'raster_size':[w,h],'render_dpi':dpi,'source_image':str(image_path),'weights':str(weights),
             'window_sizes_px':list(window_sizes),'overlap_fraction':.25,'model_input_px':640,
             'classes':classes,
             'processed_tiles':processed,'raw_candidates':len(rows),'merged_candidates':len(detections),
             'confidence_threshold':confidence,'global_nms_iou':.45,
             'class_candidate_counts':dict(Counter(r['class_name'] for r in detections)),
             'scope':'Candidate locations in the known PDF; no complete-page certified ground truth accuracy.'}
    local.write_json(out/'detections.json',detections);local.write_json(out/'summary.json',summary)
    cards=[]
    for r in detections:
        b=r['raster_bbox_xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        cards.append(f'''<button onclick="focusBox({cx},{cy})">{html.escape(r['detection_id'])} · {html.escape(r['class_name'])} · {r['confidence']:.2f}</button>''')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>지정 도면 YOLO26 검출</title>
<style>body{margin:0;font:15px system-ui;background:#f4f6f8}header{padding:12px;background:white}aside{position:absolute;top:85px;bottom:0;width:240px;overflow:auto}button{display:block;width:95%;margin:5px;padding:8px;border:1px solid #ddd;background:white;cursor:pointer}#viewer{position:absolute;left:245px;right:0;top:85px;bottom:0;overflow:auto;background:#bbb}img{display:block;max-width:none}</style>
<header><b>지정 도면 검출 후보</b> · CONF · COUNT개<br>확대 <input id="zoom" type="range" min="0.1" max="2" step="0.05" value="0.25"> <span id="scale"></span> · 왼쪽 후보를 누르면 해당 위치로 이동합니다. 후보의 정확성은 도면에서 확인하세요.</header>
<aside>CARDS</aside><div id="viewer"><img id="drawing" src="drawing_detections.png"></div>
<script>const viewer=document.getElementById('viewer'),picture=document.getElementById('drawing'),zoom=document.getElementById('zoom');
function resize(){picture.style.width=(WIDTH*Number(zoom.value))+'px';document.getElementById('scale').textContent=Math.round(Number(zoom.value)*100)+'%'}
function focusBox(x,y){viewer.scrollTo({left:x*Number(zoom.value)-viewer.clientWidth/2,top:y*Number(zoom.value)-viewer.clientHeight/2,behavior:'smooth'})}
zoom.oninput=resize;resize();</script></html>'''
    page=page.replace('CONF',f'confidence ≥ {confidence}').replace('COUNT',str(len(detections))).replace('CARDS',''.join(cards)).replace('WIDTH',str(w))
    local.write_bytes_durable(out/'review.html',page.encode('utf-8'))
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--out',required=True);p.add_argument('--dpi',type=float,default=144)
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--windows',type=int,nargs='+',default=[320,640,1280,2560])
    p.add_argument('--classes',type=int,nargs='+')
    a=p.parse_args();run(Path(a.image),Path(a.weights),Path(a.out),a.dpi,a.conf,a.windows,a.classes)
