"""Convert approved capture proposals into 640px images and matching labels.

Native pixels are retained except for two references whose bodies exceed 640.
No augmentation, split or training is performed here. The manifest retains the
source -> resize -> crop -> pad transform for every tile and annotation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'capture_box_review_20261008'
OUT = HERE / 'capture_640_20261008'
SIZE = 640
STRIDE = 480
MARGIN = 16
MIN_VISIBILITY = 0.50
MIN_EDGE = 8


def axis_grid(length):
    last = max(0, length-SIZE)
    return sorted(set([*range(0,last+1,STRIDE),last]))


def fit_axis(a, b, length):
    if length<=SIZE:return [0]
    last=max(0,length-SIZE)
    # Allow a little off-page padding to keep a neighbouring target out of a
    # tile rather than leaving a 1px unhelpful fragment at the opposite edge.
    lo=max(-64,math.ceil(b-SIZE))
    hi=min(last+64,math.floor(a))
    if lo>hi:raise ValueError('Object cannot fit inside the tile')
    middle=max(lo,min(hi,round((a+b-SIZE)/2)))
    inner_lo=max(lo,math.ceil(b-SIZE+MARGIN))
    inner_hi=min(hi,math.floor(a-MARGIN))
    values={lo,hi,middle}
    if inner_lo<=inner_hi:
        values.update([inner_lo,inner_hi,max(inner_lo,min(inner_hi,middle))])
    values.update(round(lo+(hi-lo)*i/8) for i in range(9))
    return sorted(values)


def inspect_window(objects, x, y, width, height):
    x2,y2=min(width,x+SIZE),min(height,y+SIZE)
    intersections=[]
    for obj in objects:
        a,b,c,d=obj['resized_bbox_xyxy_px']
        clipped=[max(a,x),max(b,y),min(c,x2),min(d,y2)]
        w,h=clipped[2]-clipped[0],clipped[3]-clipped[1]
        if w<=1e-7 or h<=1e-7:continue
        full=a>=x-1e-7 and b>=y-1e-7 and c<=x2+1e-7 and d<=y2+1e-7
        visibility=(w*h)/((c-a)*(d-b))
        intersections.append((obj,clipped,full,visibility))
    safe=all(full or (visibility>=MIN_VISIBILITY and min(b[2]-b[0],b[3]-b[1])>=MIN_EDGE)
             for _,b,full,visibility in intersections)
    return safe,intersections


def choose_windows(objects,width,height):
    chosen={}
    for y in axis_grid(height):
        for x in axis_grid(width):
            safe,intersections=inspect_window(objects,x,y,width,height)
            if safe:chosen[(x,y)]=intersections
    for target in objects:
        key=target['box_id']
        if any(any(o['box_id']==key and full for o,_,full,_ in ints) for ints in chosen.values()):continue
        a,b,c,d=target['resized_bbox_xyxy_px']
        options=[]
        for x in fit_axis(a,c,width):
            for y in fit_axis(b,d,height):
                safe,intersections=inspect_window(objects,x,y,width,height)
                if not safe:continue
                if not any(o['box_id']==key and full for o,_,full,_ in intersections):continue
                complete=sum(full for _,_,full,_ in intersections)
                partial=sum(not full for _,_,full,_ in intersections)
                center_error=abs((a+c)/2-(x+min(SIZE,width)/2))+abs((b+d)/2-(y+min(SIZE,height)/2))
                options.append((partial,-complete,center_error,x,y,intersections))
        if not options:
            raise RuntimeError(f'{key}: no safe tile containing its full annotated bbox; manual crop needed')
        options.sort(key=lambda t:t[:5])
        *_,x,y,intersections=options[0]
        chosen[(x,y)]=intersections
    if not chosen:
        raise RuntimeError('No safe tiles for this capture')
    return [(x,y,ints) for (x,y),ints in sorted(chosen.items(),key=lambda pair:(pair[0][1],pair[0][0]))]


def background_color(image):
    a=np.asarray(image)
    return tuple(int(v) for v in np.median(a.reshape(-1,3),axis=0))


def label_line(class_id,box):
    x1,y1,x2,y2=box
    return f'{class_id} {(x1+x2)/2/SIZE:.8f} {(y1+y2)/2/SIZE:.8f} {(x2-x1)/SIZE:.8f} {(y2-y1)/SIZE:.8f}'


def render_preview(image,annotations,dark,font):
    if dark:
        a=np.asarray(image).astype(np.int16)
        bg=np.median(a.reshape(-1,3),axis=0)
        v=255-np.clip(np.max(abs(a-bg),axis=2)*3,0,255).astype(np.uint8)
        image=Image.fromarray(v).convert('RGB')
    else:image=image.copy()
    draw=ImageDraw.Draw(image)
    for obj in annotations:
        box=obj['bbox_xyxy_px'];color='#cc0088' if not obj['truncated_by_tile'] else '#c85b00'
        draw.rectangle(box,outline=color,width=2)
        text=f"{obj['source_box_id']} : {obj['class_name']}"
        if obj['truncated_by_tile']:text+=' (crop)'
        x,y=box[:2];y=max(0,y-18)
        right=min(SIZE,x+draw.textlength(text,font=font)+4)
        draw.rectangle([x,y,right,y+18],fill='white')
        draw.text((x+2,y),text,font=font,fill=color)
    return image


def main():
    if (OUT/'manifest.json').exists():
        raise FileExistsError(f'Output already exists: {OUT}; avoid replacing derived files accidentally')
    data=json.loads((SOURCE/'proposals.json').read_text(encoding='utf-8'))
    accepted=json.loads(json.dumps(data))
    accepted['review_required']=False
    accepted['accepted_by']='user_in_chat'
    accepted['acceptance_evidence']='다 잘 잡혀있는거같아 이제 뭐 해야하지 ?'
    accepted['acceptance_date']='2026-10-08'
    for cap in accepted['captures']:
        cap['status']='approved_in_chat'
        for obj in cap['objects']:obj['status']='approved_in_chat'
    for name in ['images','labels','preview','sheets']:(OUT/name).mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    sources=[];tiles=[];resized_sources=[];coverage={}
    for cap in accepted['captures']:
        source_path=SOURCE/'images'/cap['filename']
        source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
        if cap.get('source_sha256'):assert cap['source_sha256']==source_hash
        with Image.open(source_path) as im:raw=im.convert('RGB')
        assert raw.size==(cap['width'],cap['height'])
        max_object=max(max(o['bbox_xyxy_px'][2]-o['bbox_xyxy_px'][0],o['bbox_xyxy_px'][3]-o['bbox_xyxy_px'][1])
                       for o in cap['objects'])
        factor=min(1,(SIZE-2*MARGIN)/max_object) if max_object>SIZE else 1
        new_size=(max(1,round(raw.width*factor)),max(1,round(raw.height*factor)))
        resized=raw if factor==1 else raw.resize(new_size,Image.Resampling.LANCZOS)
        sx,sy=resized.width/raw.width,resized.height/raw.height
        objects=[]
        for obj in cap['objects']:
            a,b,c,d=obj['bbox_xyxy_px']
            objects.append({**obj,'resized_bbox_xyxy_px':[a*sx,b*sy,c*sx,d*sy]})
            coverage[obj['box_id']]=[]
        source_record={'capture_id':cap['capture_id'],'filename':cap['filename'],'original_name':cap['original_name'],
            'source_sha256':source_hash,'original_size':[raw.width,raw.height],
            'resized_size':list(new_size),'scale_xy':[sx,sy],'resized_for_oversize_symbol':factor!=1,
            'source_box_ids':[o['box_id'] for o in objects],'source_edge_box_ids':[o['box_id'] for o in objects if o['touches_edge']],
            'note':cap['note']}
        sources.append(source_record)
        if factor!=1:resized_sources.append(source_record)
        bg=background_color(raw)
        windows=choose_windows(objects,resized.width,resized.height)
        for j,(x,y,intersections) in enumerate(windows,1):
            name=f"capture_{cap['capture_id']}_tile_{j:03d}"
            x2,y2=min(resized.width,x+SIZE),min(resized.height,y+SIZE)
            crop_x,crop_y=max(0,x),max(0,y)
            patch=resized.crop((crop_x,crop_y,x2,y2))
            px=(SIZE-resized.width)//2 if resized.width<=SIZE else max(0,-x)
            py=(SIZE-resized.height)//2 if resized.height<=SIZE else max(0,-y)
            canvas=Image.new('RGB',(SIZE,SIZE),bg);canvas.paste(patch,(px,py))
            # Pixel equality checks apply before any preview rendering.
            assert np.array_equal(np.asarray(canvas)[py:py+patch.height,px:px+patch.width],np.asarray(patch))
            annotations=[]
            for obj,b,full,visibility in intersections:
                dest=[b[0]-crop_x+px,b[1]-crop_y+py,b[2]-crop_x+px,b[3]-crop_y+py]
                annotations.append({'class_id':obj['class_id'],'class_name':obj['class_name'],
                    'source_box_id':obj['box_id'],'source_bbox_xyxy_px':obj['bbox_xyxy_px'],
                    'bbox_xyxy_px':dest,'truncated_by_tile':not full,'visible_fraction':visibility,
                    'source_touches_edge':obj['touches_edge']})
                if full:coverage[obj['box_id']].append(name)
            canvas.save(OUT/'images'/f'{name}.png')
            (OUT/'labels'/f'{name}.txt').write_text('\n'.join(label_line(o['class_id'],o['bbox_xyxy_px']) for o in annotations)
                                                   + ('\n' if annotations else ''),encoding='utf-8')
            preview=render_preview(canvas,annotations,cap['background']=='dark',font)
            preview.save(OUT/'preview'/f'{name}.png')
            tiles.append({'tile_id':name,'parent_capture_id':cap['capture_id'],
                'original_name':cap['original_name'],'source_sha256':source_hash,
                'image':f'images/{name}.png','label':f'labels/{name}.txt','preview':f'preview/{name}.png',
                'size':[SIZE,SIZE],'scale_xy':[sx,sy],'crop_xyxy_resized_px':[crop_x,crop_y,x2,y2],
                'window_origin_resized_px':[x,y],
                'padding_xy':[px,py],'background_rgb':list(bg),'annotations':annotations,
                'split':'UNASSIGNED','augmented':False,
                'coordinate_formula':'tile_x = source_x * scale_x - crop_x + pad_x; same for y'})
    missing=[key for key,tile_names in coverage.items() if not tile_names]
    assert not missing,missing
    manifest={'version':1,'tile_size':SIZE,'base_stride':STRIDE,'classes':data['classes'],
        'source_dataset':str(SOURCE),'accepted_annotations':'accepted_source_annotations.json',
        'source_captures':sources,'tiles':tiles,'full_annotated_bbox_coverage':coverage,
        'source_region_grouping_for_train_val':'not assigned; overlapping/paired captures must be grouped before splitting',
        'partial_policy':f'Every intersecting target is labelled; reject tile if clipped target visibility < {MIN_VISIBILITY} or min edge < {MIN_EDGE}px.',
        'training_started':False,'augmentation_applied':False,'split_assigned':False}
    # Validate output labels, original coordinates round trips, classes and size.
    for tile in tiles:
        with Image.open(OUT/tile['image']) as image:assert image.size==(SIZE,SIZE)
        lines=(OUT/tile['label']).read_text().splitlines()
        assert len(lines)==len(tile['annotations'])
        sx,sy=tile['scale_xy'];x,y,_,_=tile['crop_xyxy_resized_px'];px,py=tile['padding_xy']
        for obj,line in zip(tile['annotations'],lines):
            cls,cx,cy,w,h=map(float,line.split())
            assert data['classes'][str(int(cls))]==obj['class_name']
            assert w>0 and h>0 and cx-w/2>=-1e-7 and cy-h/2>=-1e-7 and cx+w/2<=1+1e-7 and cy+h/2<=1+1e-7
            b=obj['bbox_xyxy_px']
            recovered=[(b[0]+x-px)/sx,(b[1]+y-py)/sy,(b[2]+x-px)/sx,(b[3]+y-py)/sy]
            if not obj['truncated_by_tile']:assert np.allclose(recovered,obj['source_bbox_xyxy_px'],atol=1e-6)
    (OUT/'accepted_source_annotations.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(SOURCE/'classes.json',OUT/'classes.json')
    summary={'source_captures':len(sources),'source_boxes':len(coverage),'output_images':len(tiles),
        'output_labels':len(tiles),'all_image_sizes':[SIZE,SIZE],
        'output_annotation_instances':sum(len(t['annotations']) for t in tiles),
        'full_source_bbox_coverage':len(coverage),'missing_source_boxes':missing,
        'native_scale_captures':sum(not s['resized_for_oversize_symbol'] for s in sources),
        'oversize_resized_captures':resized_sources,
        'tiles_with_partial_boxes':sum(any(o['truncated_by_tile'] for o in t['annotations']) for t in tiles),
        'negative_tiles':sum(not t['annotations'] for t in tiles),
        'source_edge_boxes':sum(len(s['source_edge_box_ids']) for s in sources),
        'augmentation_applied':False,'train_val_split_assigned':False,'training_started':False}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'tiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fieldnames=['tile_id','parent_capture_id','original_name','scale_x','scale_y','crop_x','crop_y','pad_x','pad_y','box_count','split']
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
        for t in tiles:
            writer.writerow(dict(tile_id=t['tile_id'],parent_capture_id=t['parent_capture_id'],original_name=t['original_name'],
                scale_x=t['scale_xy'][0],scale_y=t['scale_xy'][1],crop_x=t['crop_xyxy_resized_px'][0],crop_y=t['crop_xyxy_resized_px'][1],
                pad_x=t['padding_xy'][0],pad_y=t['padding_xy'][1],box_count=len(t['annotations']),split='UNASSIGNED'))
    # Contact sheets are previews only; exported training images stay exactly 640.
    heading_font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for start in range(0,len(tiles),12):
        sheet=Image.new('RGB',(4*330,3*355),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for i,t in enumerate(tiles[start:start+12]):
            with Image.open(OUT/t['preview']) as p:thumb=p.copy()
            thumb.thumbnail((320,320))
            xx,yy=i%4*330,i//4*355
            sheet.paste(thumb,(xx+5,yy+28))
            draw.text((xx+6,yy+4),t['tile_id'].replace('capture_',''),font=heading_font,fill='black')
        sheet.save(OUT/'sheets'/f'640_preview_{start//12+1:02d}.jpg',quality=93)
    info='\n'.join(f"  {s['capture_id']}: {s['original_size']} -> {s['resized_size']}; scale={s['scale_xy']}" for s in resized_sources)
    (OUT/'README.txt').write_text(f'''640×640 이미지와 정답박스 좌표 변환 결과

원본 캡처 {len(sources)}장 / 원본 박스 {len(coverage)}개
출력 이미지 {len(tiles)}장, 같은 이름의 YOLO .txt 라벨 {len(tiles)}개
모든 원본 정답박스가 전체 포함된 타일을 최소 1장 확보했습니다.
이는 원본 캡처 자체에서 잘렸던 형상의 복원을 뜻하지는 않습니다.

108장: 원본 픽셀 크기로 자르거나 동일 배경 여백을 채웠습니다.
2장: 심볼 자체의 높이가 640px보다 커 비율을 유지해 축소했습니다.
{info}

images/: 학습에 사용할 640×640 원본 색상 이미지
labels/: class_id center_x center_y width height (640 기준 정규화)
preview/: 박스 표시/어두운 CAD 대비 보정 이미지. 검수 전용입니다.
sheets/: 12장씩 묶은 검수 미리보기. 학습 입력이 아닙니다.
manifest.json: 원본 ID, 해시, 축소 배율, 자른 좌표, 여백, 박스 변환 상세
accepted_source_annotations.json: 사용자가 채팅에서 확인한 221개 원본 박스
classes.json: 기존 38개 클래스 ID 유지
tiles.csv: 타일과 원본 캡처의 대응표

변환식: 새 좌표 = 원본 좌표 × 배율 − 타일 시작 좌표 + 여백
경계에서 잘린 심볼도 보이는 부분의 박스를 붙이고 partial 플래그를 기록합니다.
지나치게 작은 조각이 되는 타일은 제외했습니다.
일부 원본의 경계 잘림은 원본 자료의 한계이며 manifest에 보존합니다.

증강, 학습·검증 분리, YOLO 학습은 이 단계에서 실행하지 않았습니다.
분리 시 같은 원본의 모든 타일과 겹치는 흰/검은 캡처를 같은 그룹으로 묶으세요.
원본 파일은 수정하지 않았습니다. 외부 API 호출 없음.
''',encoding='utf-8-sig')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():archive.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as archive:assert archive.testzip() is None
    print(json.dumps({k:v for k,v in summary.items() if k!='oversize_resized_captures'},ensure_ascii=True))

if __name__=='__main__':main()
