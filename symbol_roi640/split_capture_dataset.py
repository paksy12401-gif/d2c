"""Region holdout for the existing 640px data; preserve bytes and labels.

Pair different renderings of the same physical region. Quarantine captures
which bridge validation regions, instead of letting those regions leak into
training. This is an intra-drawing check, not a new-drawing benchmark.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import shutil
import zipfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'capture_640_20261008'
RAW=HERE/'capture_box_review_20261008'
WORK=HERE/'capture_split_work_20261008'
OUT=HERE/'capture_train_val_20261008'

# Region identity was checked using orientation, part classes, dimension
# context and the source overview. Repeated class names alone are not groups.
PAIRS={5:28,4:27,6:30,7:33,9:35,13:38,20:43,31:50,32:49,
       39:43,71:52,82:55,96:76,97:75,101:67,102:68,103:69,107:83,108:85}
VAL_REGIONS={
    'horizontal_7_145mm':[5,28],
    'vertical_7_and_adjacent_14':[32,49,48,52,53,70,71],
    'middle_7_and_12':[13,37,38],
    'upper_vertical_7_170mm':[75,97],
    'right_11_120mm':[83,84,107],
}
# Additional conservative bridge exclusions when sparse feature matching
# cannot prove a location. The data are kept, not deleted.
BRIDGE_HOLD={4,6,9,10,12,14,18,27,30,31,35,36,40,50,54,73,74,76,81,85,96,98,106,108}
WEAK_VISUALLY_CHECKED={12,16,18,68,70,74,75,80,98,100,106,108}


def centers(cap):
    return [(o['class_name'],np.array([(o['bbox_xyxy_px'][0]+o['bbox_xyxy_px'][2])/2,
                                     (o['bbox_xyxy_px'][1]+o['bbox_xyxy_px'][3])/2]))
            for o in sorted(cap['objects'],key=lambda o:(o['class_name'],o['bbox_xyxy_px'][1],o['bbox_xyxy_px'][0]))]


def fit_counterpart(src,ref,matrix):
    a=centers(src);b=centers(ref)
    if [n for n,_ in a]!=[n for n,_ in b]:raise ValueError('Counterpart classes differ')
    source=np.array([xy for _,xy in a]);target=np.array([xy for _,xy in b])
    target=target@matrix[:,:2].T+matrix[:,2]
    sa=source-source.mean(axis=0);tb=target-target.mean(axis=0)
    scale=float(np.sum(sa*tb)/np.sum(sa*sa))
    translation=target.mean(axis=0)-source.mean(axis=0)*scale
    result=np.array([[scale,0,translation[0]],[0,scale,translation[1]]])
    residual=float(np.linalg.norm(source*scale+translation-target,axis=1).max())
    assert .1<scale<2,(src['capture_id'],scale)
    return result,residual


def project_box(box,matrix):
    a,b,c,d=box
    p=np.array([[a,b],[c,b],[c,d],[a,d]])@matrix[:,:2].T+matrix[:,2]
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]


def shared_instance(a,b):
    if a['class_name']!=b['class_name']:return False
    x1,y1,x2,y2=a['pdf_bbox'];u1,v1,u2,v2=b['pdf_bbox']
    intersection=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    smaller=min((x2-x1)*(y2-y1),(u2-u1)*(v2-v1))
    # Conservative threshold accommodates cropped/outlined versus filled bodies.
    return smaller>0 and intersection/smaller>=.25


def main():
    if OUT.exists():raise FileExistsError(f'Output already exists: {OUT}')
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    accepted=json.loads((SOURCE/'accepted_source_annotations.json').read_text(encoding='utf-8'))
    caps={int(c['capture_id']):c for c in accepted['captures']}
    locations=json.loads((WORK/'capture_locations_144.json').read_text())
    matrices={};methods={}
    for row in locations:
        i=int(row['capture_id'])
        if row['located'] or i in WEAK_VISUALLY_CHECKED:
            assert row.get('matrix') and abs(row.get('angle',999))<2 and .1<row.get('scale',0)<2
            matrices[i]=np.asarray(row['matrix']);methods[i]={'method':'pdf_feature_registration',
                'feature_inliers':row['inliers'],'visually_checked_below_auto_threshold':not row['located']}
    # 067 is a clipped view of the 120mm interval just left of the 135mm view.
    # Transfer from the right 14 here to the left 14 of 068; do not transfer
    # class-name similarity to an arbitrary occurrence of the same symbol.
    c67=caps[67]['objects'][1];c68=caps[68]['objects'][0]
    p67=np.array([(c67['bbox_xyxy_px'][0]+c67['bbox_xyxy_px'][2])/2,
                  (c67['bbox_xyxy_px'][1]+c67['bbox_xyxy_px'][3])/2])
    p68=np.array([(c68['bbox_xyxy_px'][0]+c68['bbox_xyxy_px'][2])/2,
                  (c68['bbox_xyxy_px'][1]+c68['bbox_xyxy_px'][3])/2])
    target=p68@matrices[68][:,:2].T+matrices[68][:,2]
    scale=float(np.hypot(*matrices[68][:,0]));trans=target-p67*scale
    matrices[67]=np.array([[scale,0,trans[0]],[0,scale,trans[1]]])
    methods[67]={'method':'manual_shared_14_boundary_anchor','reference_capture':'068','approximate':True}
    for src,ref in PAIRS.items():
        matrix,error=fit_counterpart(caps[src],caps[ref],matrices[ref])
        matrices[src]=matrix;methods[src]={'method':'visually_paired_region_anchor_transfer',
                                         'reference_capture':f'{ref:03d}','anchor_residual_pdf_points':round(error,3),
                                         'approximate':True}
    # Only one target is visible in 029. Its part 23 is unique in this drawing.
    src=caps[29];ref=caps[24]
    a=next(xy for name,xy in centers(src) if name=='23')
    b=next(xy for name,xy in centers(ref) if name=='23')
    target=b@matrices[24][:,:2].T+matrices[24][:,2]
    scale=.465;trans=target-a*scale
    matrices[29]=np.array([[scale,0,trans[0]],[0,scale,trans[1]]])
    methods[29]={'method':'manual_unique_part_23_anchor','reference_capture':'024','approximate':True}
    assert set(matrices)==set(caps),sorted(set(caps)-set(matrices))
    projected={i:[{**o,'pdf_bbox':project_box(o['bbox_xyxy_px'],matrices[i])} for o in cap['objects']]
               for i,cap in caps.items()}
    val_ids={i for ids in VAL_REGIONS.values() for i in ids}
    val_objects=[o for i in val_ids for o in projected[i]]
    assignments={};reasons={}
    for i in caps:
        if i in val_ids:assignments[i]='val';reasons[i]='held-out physical region'
        elif i in BRIDGE_HOLD:assignments[i]='quarantine';reasons[i]='conservative overlapping-region bridge'
        else:
            shared=[(a['box_id'],b['box_id']) for a in projected[i] for b in val_objects if shared_instance(a,b)]
            if shared:assignments[i]='quarantine';reasons[i]=f'projected overlap with validation instances: {shared}'
            else:assignments[i]='train';reasons[i]='outside held-out symbol regions'
    # Final direct cross-split symbol exposure check, independent of the source
    # filename/capture IDs and their 640px descendants.
    train_objects=[o for i in caps if assignments[i]=='train' for o in projected[i]]
    cross=[(a['box_id'],b['box_id']) for a in train_objects for b in val_objects if shared_instance(a,b)]
    assert not cross,cross
    # Group same physical instances among the USED captures. Bridges are kept
    # out, so a mixed-context capture cannot pull validation back into training.
    parent={i:i for i in caps}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    used=[i for i in caps if assignments[i]!='quarantine']
    for pos,a in enumerate(used):
        for b in used[pos+1:]:
            if any(shared_instance(x,y) for x in projected[a] for y in projected[b]):
                assert assignments[a]==assignments[b]
                union(a,b)
    groups=defaultdict(list)
    for i in used:groups[find(i)].append(i)
    group_ids={i:f'group_{min(ids):03d}' for ids in groups.values() for i in ids}
    for i in caps:
        if assignments[i]=='quarantine':group_ids[i]=f'bridge_{i:03d}'
    tile_records=[];class_counts={s:Counter() for s in ['train','val','quarantine']}
    source_class_counts={s:Counter() for s in class_counts}
    for i,cap in caps.items():source_class_counts[assignments[i]].update(o['class_name'] for o in cap['objects'])
    for split in class_counts:
        (OUT/'images'/split).mkdir(parents=True);(OUT/'labels'/split).mkdir(parents=True)
        (OUT/'preview'/split).mkdir(parents=True)
    for tile in manifest['tiles']:
        i=int(tile['parent_capture_id']);split=assignments[i]
        name=Path(tile['image']).name;label=Path(tile['label']).name
        img_dest=OUT/'images'/split/name;label_dest=OUT/'labels'/split/label
        shutil.copy2(SOURCE/tile['image'],img_dest);shutil.copy2(SOURCE/tile['label'],label_dest)
        shutil.copy2(SOURCE/tile['preview'],OUT/'preview'/split/name)
        assert img_dest.read_bytes()==(SOURCE/tile['image']).read_bytes()
        assert label_dest.read_bytes()==(SOURCE/tile['label']).read_bytes()
        class_counts[split].update(o['class_name'] for o in tile['annotations'])
        tile_records.append({**tile,'split':split,'source_region_group':group_ids[i],
            'image':f'images/{split}/{name}','label':f'labels/{split}/{label}',
            'preview':f'preview/{split}/{name}',
            'sha256_image':hashlib.sha256(img_dest.read_bytes()).hexdigest()})
    capture_counts=Counter(assignments.values());image_counts=Counter(t['split'] for t in tile_records)
    no_val=[name for name in manifest['classes'].values() if source_class_counts['train'][name] and not source_class_counts['val'][name]]
    absent=[name for name in manifest['classes'].values() if not source_class_counts['train'][name] and not source_class_counts['val'][name]]
    val_without_train=[name for name in source_class_counts['val'] if not source_class_counts['train'][name]]
    assert not val_without_train,val_without_train
    sets={s:{t['sha256_image'] for t in tile_records if t['split']==s} for s in class_counts}
    assert not sets['train'] & sets['val']
    split_manifest={**manifest,'source_dataset':str(SOURCE),'tiles':tile_records,
        'source_assignments':[{'capture_id':f'{i:03d}','split':assignments[i],'group_id':group_ids[i],
            'reason':reasons[i],'pdf_transform':matrices[i].tolist(),'location_evidence':methods[i],
            'projected_source_boxes':[{'box_id':o['box_id'],'class_name':o['class_name'],'pdf_bbox':o['pdf_bbox']} for o in projected[i]]}
            for i in caps],
        'validation_regions':VAL_REGIONS,'split_assigned':True,'augmentation_applied':False,'training_started':False,
        'benchmark_scope':'held-out regions of one drawing; not independent drawings',
        'geometry_check_scope':'annotated physical symbol overlap under estimated/manual PDF transforms; not a certified exhaustive pixel overlap proof',
        'grouping_policy':'Counterpart renderings and shared physical symbols stay in one partition; bridging captures are quarantined.'}
    (OUT/'split_manifest.json').write_text(json.dumps(split_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for f in ['classes.json','accepted_source_annotations.json']:shutil.copy2(SOURCE/f,OUT/f)
    summary={'source_captures':len(caps),'source_tiles':len(tile_records),'capture_counts':dict(capture_counts),
        'image_counts':dict(image_counts),'val_fraction_of_used_tiles':round(image_counts['val']/(image_counts['train']+image_counts['val']),4),
        'used_region_group_counts':dict(Counter(assignments[ids[0]] for ids in groups.values())),
        'validation_classes':sorted(source_class_counts['val']),
        'training_classes_without_validation':no_val,'classes_absent_from_used_capture_data':absent,
        'val_classes_without_training':val_without_train,'cross_split_projected_instance_overlaps':len(cross),
        'cross_split_identical_images':0,'same_parent_cross_split':0,'original_images_and_labels_unchanged':True,
        'all_images_640':True,'augmentation_applied':False,'training_started':False,
        'benchmark_scope':'single-drawing region holdout'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'source_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['capture_id','original_name','split','group_id','reason']);writer.writeheader()
        for i,c in caps.items():writer.writerow(dict(capture_id=f'{i:03d}',original_name=c['original_name'],split=assignments[i],group_id=group_ids[i],reason=reasons[i]))
    with (OUT/'class_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['class_id','class_name','train_source_instances','val_source_instances','quarantine_source_instances','train_tile_instances','val_tile_instances','quarantine_tile_instances']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cid,name in manifest['classes'].items():
            row={'class_id':cid,'class_name':name}
            for s in class_counts:row[f'{s}_source_instances']=source_class_counts[s][name];row[f'{s}_tile_instances']=class_counts[s][name]
            writer.writerow(row)
    (OUT/'train.txt').write_text(''.join(f"./images/train/{Path(t['image']).name}\n" for t in tile_records if t['split']=='train'),encoding='utf-8')
    (OUT/'val.txt').write_text(''.join(f"./images/val/{Path(t['image']).name}\n" for t in tile_records if t['split']=='val'),encoding='utf-8')
    # Show all validation captures together, so the owner's holdout is concrete.
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    (OUT/'sheets').mkdir()
    for s in ['train','val','quarantine']:
        selected=[t for t in tile_records if t['split']==s]
        for start in range(0,len(selected),12):
            canvas=Image.new('RGB',(1320,1065),'#eeeeee');draw=ImageDraw.Draw(canvas)
            for j,t in enumerate(selected[start:start+12]):
                with Image.open(OUT/t['preview']) as im:thumb=im.copy()
                thumb.thumbnail((320,320));x,y=j%4*330,j//4*355
                canvas.paste(thumb,(x+5,y+28));draw.text((x+6,y+4),t['tile_id'].replace('capture_',''),font=font,fill='black')
            canvas.save(OUT/'sheets'/f'{s}_{start//12+1:02d}.jpg',quality=93)
    (OUT/'README.txt').write_text(f'''학습·검증 분리 결과 — 2026-10-08

학습: {image_counts['train']}장 / 검증: {image_counts['val']}장 / 겹침 보류: {image_counts['quarantine']}장
사용 이미지 중 검증 비율: {summary['val_fraction_of_used_tiles']:.1%}
캡처 단위: {dict(capture_counts)}

images/train, labels/train: 학습용 640 이미지와 기존 좌표
images/val, labels/val: 검증용 640 이미지와 기존 좌표
images/quarantine, labels/quarantine: 검증 영역과 함께 보이는 캡처에서 나온 보류 데이터
보류 파일은 삭제하지 않았습니다. 학습·검증 입력에는 포함하지 마세요.

같은 원본의 타일은 모두 같은 분할입니다. 다른 배경/크기의 동일 부품 영역
캡처도 지역/대응 관계를 확인해서 함께 처리했습니다. 검증 영역과 연결되는
넓은 캡처는 겹침 보류로 남겼습니다. source_assignments.csv에서 이유를 볼 수 있습니다.

검증 클래스: {', '.join(summary['validation_classes'])}
나머지 희소 부품들은 별도 도면의 검증 자료가 부족해서 학습 쪽에 두었습니다.
이번 분리로 모든 38개 클래스의 성능을 검증할 수 있는 것은 아닙니다.
원본 한 도면 안의 다른 영역을 검증하므로, 새 도면 일반화 성능과 구분하세요.

split_manifest.json: 원본 대응, PDF 추정 위치, 그룹, 타일, 라벨 상세
class_distribution.csv: 클래스별 원본/타일의 학습·검증·보류 수량
summary.json: 분리 결과와 중복/좌표 검사 결과
sheets/: 분할별 박스 표시 미리보기
train.txt, val.txt: 이번 단계의 이미지 목록

검사: 학습/검증 원본 ID 겹침 0, 동일 이미지 해시 겹침 0,
추정 PDF 좌표의 동일 클래스 부품 겹침 0. 좌표 추정에는 수동 대응도 포함되어
전체 픽셀 겹침을 완전히 증명한 결과로 해석하면 안 됩니다.
이미지와 .txt 라벨은 640 처리 결과에서 바이트 그대로 복사했습니다.
기존 개별 심볼 증강 데이터는 이번 분할에 아직 합치지 않았습니다.
추가 증강, 학습 코드 생성, YOLO 학습은 진행하지 않았습니다.
''',encoding='utf-8-sig')
    for t in tile_records:
        with Image.open(OUT/t['image']) as image:assert image.size==(640,640)
        assert (OUT/t['label']).exists()
    assert sum(image_counts.values())==len(manifest['tiles'])
    assert len({t['tile_id'] for t in tile_records})==len(manifest['tiles'])
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
