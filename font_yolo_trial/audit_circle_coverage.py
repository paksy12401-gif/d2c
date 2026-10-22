"""Audit round callout coverage across both CIRCLE and closed-polyline imports."""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import ezdxf
import numpy as np
import pymupdf as fitz
from ezdxf import bbox
from PIL import Image,ImageDraw,ImageFont

from non_table_ocr import TABLES,intersects
from vector_font_probe import parse_linear_path,bounds,group_glyphs,load_candidates

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_text_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
    page=fitz.open(ROOT/'font_yolo_trial/cad_without_tables/source.pdf')[0]
    doc=ezdxf.readfile(ROOT/'font_yolo_trial/cad_without_tables/non_table_geometry.dxf')
    circles=[];cache=bbox.Cache()
    for e in doc.modelspace():
        if e.dxf.get('true_color')!=2490368:continue
        if e.dxftype()=='CIRCLE' and 9<e.dxf.radius*72<11:
            circles.append(dict(handle=e.dxf.handle,x=e.dxf.center.x*72,y=page.rect.height-e.dxf.center.y*72,
                radius=e.dxf.radius*72,entity_type='CIRCLE'))
        elif e.dxftype()=='LWPOLYLINE' and e.closed:
            b=bbox.extents([e],cache=cache);w=(b.extmax.x-b.extmin.x)*72;h=(b.extmax.y-b.extmin.y)*72
            if 18<w<22 and 18<h<22:
                circles.append(dict(handle=e.dxf.handle,x=(b.extmin.x+b.extmax.x)*36,
                    y=page.rect.height-(b.extmin.y+b.extmax.y)*36,radius=(w+h)/4,entity_type='LWPOLYLINE'))
    circles.sort(key=lambda c:int(c['handle'],16))
    drawings=page.get_drawings()
    source_rings=[d for d in drawings if d['color'] and d['color'][0]>.1 and d['color'][1]<.01 and d['color'][2]<.01
                  and 18<d['rect'].width<22 and 18<d['rect'].height<22 and not any(intersects(list(d['rect']),t) for t in TABLES)]
    assert len(source_rings)==len(circles)==48
    for c in circles:
        assert any(abs((d['rect'].x0+d['rect'].x1)/2-c['x'])<.05 and
                   abs((d['rect'].y0+d['rect'].y1)/2-c['y'])<.05 for d in source_rings)
    (OUT/'circles.json').write_text(json.dumps(circles,ensure_ascii=False,indent=2),encoding='utf-8')
    actual=ezdxf.readfile(ROOT/'output/cad_readability_fixed/dwg_roundtrip.dxf');cache=bbox.Cache();textboxes=[]
    for e in actual.modelspace().query('TEXT MTEXT'):
        b=bbox.extents([e],cache=cache)
        if b.has_data:textboxes.append((e.dxf.handle,[(b.extmin.x+b.extmax.x)*36,page.rect.height-(b.extmin.y+b.extmax.y)*36]))
    checks=[]
    for c in circles:
        x,y,r=c['x'],c['y'],c['radius'];inside=[]
        for index,d in enumerate(drawings):
            b=d['rect']
            if x-r*.82<=b.x0 and b.x1<=x+r*.82 and y-r*.82<=b.y0 and b.y1<=y+r*.82:
                if b.width<.01 and b.height<.01:continue
                inside.append(dict(id=index,box=list(b),kinds=dict(Counter(i[0] for i in d['items']))))
        handles=[h for h,(tx,ty) in textboxes if abs(tx-x)<r*.9 and abs(ty-y)<r*.9]
        assert inside and not handles
        checks.append(dict(circle_handle=c['handle'],inside_paths=inside,editable_text_inside=handles))
    (OUT/'circle_path_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    found=[];selected={c['path_id'] for c in load_candidates(page)}
    for i,node in enumerate(ET.fromstring(page.get_svg_image(text_as_path=True)).iter()):
        if not node.tag.endswith('path') or node.get('stroke')!='#260000':continue
        nums=re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',node.get('transform','matrix(1,0,0,1,0,0)'))
        if len(nums)!=6:continue
        runs=parse_linear_path(node.get('d',''),list(map(float,nums)))
        if not runs:continue
        b=bounds([s for r in runs for s in r])
        c=next((c for c in circles if c['x']-c['radius']*.82<=b[0] and b[2]<=c['x']+c['radius']*.82
                and c['y']-c['radius']*.82<=b[1] and b[3]<=c['y']+c['radius']*.82),None)
        if not c:continue
        groups=group_glyphs(runs);hs=[g['box'][3]-g['box'][1] for g in groups if g['box'][3]>g['box'][1]+.05]
        cap=float(np.median(sorted(hs)[len(hs)//3:])) if hs else 0
        reasons=[]
        if len(groups)<3:reasons.append('fewer_than_3_groups')
        if not 4<=cap<=55:reasons.append('cap_height_outside_4_to_55')
        if b[3]-b[1]>cap*1.65:reasons.append('height_greater_than_1_65_cap')
        if b[2]-b[0]<cap*2:reasons.append('width_less_than_2_cap')
        found.append(dict(circle=c['handle'],svg_path=i,groups=len(groups),estimated_cap=cap,
                          box_pdf=b,rejected_conditions=reasons,in_candidates=i in selected))
    assert len(found)==len(circles) and all(not r['in_candidates'] and r['rejected_conditions'] for r in found)
    counts=dict(Counter(reason for r in found for reason in r['rejected_conditions']))
    (OUT/'omission_filter_evidence.json').write_text(json.dumps(dict(confirmed=len(found),condition_counts=counts,paths=found),ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1400,1140),'white');draw=ImageDraw.Draw(sheet)
    for i,c in enumerate(circles):
        x,y,r=c['x'],c['y'],c['radius'];pix=page.get_pixmap(matrix=fitz.Matrix(6,6),clip=fitz.Rect(x-r*1.12,y-r*1.12,x+r*1.12,y+r*1.12),alpha=False)
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((160,145));a,b=(i%8)*175,(i//8)*190
        sheet.paste(im,(a+5,b+20));draw.text((a+5,b+1),c['handle'],fill='black');draw.text((a+5,b+165),'%.0f, %.0f'%(x,y),fill='black')
        pix.save(OUT/'assets'/('circle_'+c['handle']+'.png'))
    sheet.save(OUT/'circled_number_sources.png')
    scale=.65;pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=fitz.Rect(0,0,5045,page.rect.height),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);canvas=Image.new('RGB',(im.width,im.height+70),'white');canvas.paste(im,(0,70));draw=ImageDraw.Draw(canvas)
    draw.text((20,12),'주황색 표시: 원 안 번호 48곳 — 현재 선으로만 남고 TEXT 변환 누락',font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',26),fill='#8a3900')
    for c in circles:
        x,y,r=c['x']*scale,c['y']*scale+70,c['radius']*scale+6;draw.ellipse((x-r,y-r,x+r,y+r),outline='#ef6c00',width=3)
    canvas.save(OUT/'missing_number_locations.png')
    print('CONFIRMED CALL0UTS',len(circles),Counter(c['entity_type'] for c in circles),counts,flush=True)


if __name__=='__main__':main()
