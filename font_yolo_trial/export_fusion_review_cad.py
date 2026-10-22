"""Package latest saved recognition as editable text in a table-free review DWG.

Recognition answers are never loaded. Original text strokes remain on an off layer;
only imported polylines whose every segment matches a recognized PDF text path move.
"""
import hashlib
import json
import re
import subprocess
from collections import Counter,defaultdict
from pathlib import Path

import ezdxf
import numpy as np
import pymupdf as fitz
from ezdxf.enums import TextEntityAlignment
from scipy.spatial import cKDTree

from non_table_cad import in_table,pdf_bounds
from vector_font_probe import load_candidates

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_readability_fixed'
CORE=Path('C:/Program Files/Autodesk/AutoCAD 2027/accoreconsole.exe')
STEM='non_table_readable_yolo_ocr'
ORIGINAL='SOURCE_TEXT_OUTLINES'
LAYERS={'YOLO_LATIN':(7,None),'YOLO_KOREAN':(6,None),'FRAME_TEXT':(4,None),ORIGINAL:(8,0x888888)}


class StrokeIndex:
    def __init__(self,fields):
        self.edges=defaultdict(set)
        for identity,segments in fields.items():
            for x0,y0,x1,y1 in segments:
                a,b=(round(x0,3),round(y0,3)),(round(x1,3),round(y1,3))
                if a!=b:self.edges[tuple(sorted((a,b)))].add(identity)
        self.points=sorted({p for edge in self.edges for p in edge})
        self.tree=cKDTree(self.points)

    def match(self,points,closed=False):
        if len(points)<2:return None,[]
        distances,indices=self.tree.query(points,distance_upper_bound=.05)
        if not np.all(np.isfinite(distances)):return None,[]
        snapped=[self.points[int(i)] for i in indices]
        if closed and snapped[-1]!=snapped[0]:snapped.append(snapped[0])
        common=None;edges=[]
        for a,b in zip(snapped,snapped[1:]):
            if a==b:continue
            edge=tuple(sorted((a,b)));identities=self.edges.get(edge,set())
            if not identities:return None,[]
            common=set(identities) if common is None else common&identities
            if not common:return None,[]
            edges.append(edge)
        return (sorted(common)[0],edges) if common and edges else (None,[])


def source_strokes(page,data):
    paths={f"P{p['path_id']}":p for p in load_candidates(page)}
    fields={r['id']:[s for g in paths[r['id']]['groups'] for s in g['segments']] for r in data['records']}
    drawings=page.get_drawings()
    for frame in data['frame_labels']:
        path=drawings[int(frame['id'][1:])]
        segments=[]
        for item in path['items']:
            if item[0]!='l':raise ValueError('Unsupported border text path')
            a,b=item[1:3];segments.append((a.x,a.y,b.x,b.y))
        fields[frame['id']]=segments
    return fields


def add_text(model,text,box,cap,layer,style,identity):
    x0,y0,x1,y1=box
    if not text or x1<=x0 or y1<=y0:raise ValueError(identity)
    if in_table(box):raise AssertionError('Text in excluded table')
    entity=model.add_text(text,dxfattribs=dict(layer=layer,style=style,height=cap/72,color=256))
    entity.set_placement((x0/72,(PAGE_HEIGHT-y1)/72,0),(x1/72,(PAGE_HEIGHT-y1)/72,0),align=TextEntityAlignment.FIT)
    entity.set_xdata('CAD_OCR_REVIEW',[(1000,identity)])
    return entity


def run_cad(source,name,commands):
    script=OUT/(name+'.scr')
    script.write_text('\n'.join(['(setvar "FILEDIA" 0)','(setvar "CMDECHO" 0)',*commands,'']),encoding='ascii')
    log=script.with_suffix('.log')
    with log.open('wb') as stream:
        result=subprocess.run([str(CORE),'/i',str(source),'/s',str(script),'/l','en-US'],cwd=OUT,
            stdout=stream,stderr=subprocess.STDOUT,timeout=300,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError(f'AutoCAD failure {result.returncode}: {log}')
    return log


def main():
    global PAGE_HEIGHT
    OUT.mkdir(parents=True,exist_ok=True)
    prediction=ROOT/'output/cad_fusion_parts_applied/combined_results.json'
    data=json.loads(prediction.read_text(encoding='utf-8'))
    if data.get('inference_llm') is not False or data.get('answers_loaded') is not False:raise AssertionError('Unexpected recognition source')
    source=Path(data['source']);page=fitz.open(source)[0];PAGE_HEIGHT=page.rect.height
    if hashlib.sha256(source.read_bytes()).hexdigest()!=data['source_sha256']:raise AssertionError('PDF changed')
    base=ROOT/'font_yolo_trial/cad_without_tables/non_table_geometry.dxf'
    doc=ezdxf.readfile(base);model=doc.modelspace();original_count=len(model)
    native_count=len(model.query('TEXT MTEXT'))
    for name,(color,true_color) in LAYERS.items():
        if name in doc.layers:raise ValueError('Review layer already exists')
        attributes=dict(color=color)
        if true_color is not None:attributes['true_color']=true_color
        doc.layers.new(name,dxfattribs=attributes)
    # Filled TrueType glyphs and adaptive ACI colors remain legible on dark CAD backgrounds.
    doc.styles.new('REVIEW_LATIN',dxfattribs={'font':'arial.ttf'})
    doc.styles.new('REVIEW_KOREAN',dxfattribs={'font':'malgun.ttf'})
    doc.appids.new('CAD_OCR_REVIEW')
    strokes=source_strokes(page,data);index=StrokeIndex(strokes)
    moved=[];matched=set();by_field=Counter()
    for entity in list(model.query('LWPOLYLINE')):
        vertices=list(entity.get_points('xyb'))
        if any(abs(v[2])>1e-10 for v in vertices):continue
        points=[(float(v[0])*72,PAGE_HEIGHT-float(v[1])*72) for v in vertices]
        identity,edges=index.match(points,entity.closed)
        if identity:
            entity.dxf.layer=ORIGINAL;entity.dxf.color=256
            if entity.dxf.hasattr('true_color'):entity.dxf.discard('true_color')
            moved.append(dict(handle=entity.dxf.handle,field_id=identity,segments=len(edges)))
            by_field[identity]+=1;matched.update(edges)
    texts=[]
    for record in data['records']:
        if len(record['fused_lines'])!=1:raise ValueError('Multiline placement needs review')
        text=record['fused_lines'][0]
        korean=bool(re.search('[가-힣]',text))
        layer='YOLO_KOREAN' if korean else 'YOLO_LATIN'
        entity=add_text(model,text,record['box_pdf'],record['cap'],layer,
                        'REVIEW_KOREAN' if korean else 'REVIEW_LATIN',record['id'])
        texts.append(dict(id=record['id'],text=text,handle=entity.dxf.handle,box_pdf=record['box_pdf'],layer=layer))
    for record in data['frame_labels']:
        entity=add_text(model,record['text'],record['box_pdf'],record['cap'],'FRAME_TEXT','REVIEW_LATIN',record['id'])
        texts.append(dict(id=record['id'],text=record['text'],handle=entity.dxf.handle,box_pdf=record['box_pdf'],layer='FRAME_TEXT'))
    doc.layers.get(ORIGINAL).off()
    doc.header['$INSUNITS']=0;doc.header['$LWDISPLAY']=False
    doc.set_modelspace_vport(height=PAGE_HEIGHT/72*1.06,center=(page.rect.width/144,PAGE_HEIGHT/144))
    dxf=OUT/(STEM+'.dxf');doc.saveas(dxf)
    reread=ezdxf.readfile(dxf)
    assert len(reread.modelspace())==original_count+len(texts)
    assert len(reread.modelspace().query('TEXT MTEXT'))==native_count+len(texts)
    assert not reread.audit().has_errors
    assert reread.layers.get(ORIGINAL).is_off()
    assert all(abs(e.dxf.insert.z)<1e-10 for e in reread.modelspace().query('TEXT MTEXT'))
    cache=ezdxf.bbox.Cache()
    assert not any(in_table(b) for e in reread.modelspace() if (b:=pdf_bounds(e,cache)))
    coverage={identity:dict(hidden_entities=by_field[identity],source_segments=len(segments)) for identity,segments in strokes.items()}
    manifest=dict(review_only=True,inference_llm=False,answers_used=False,
        recognition_source=str(prediction.resolve()),recognition_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
        source_pdf_sha256=data['source_sha256'],base_geometry_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        original_entities=original_count,preserved_original_entities=original_count,
        native_editable_text=native_count,added_text=len(texts),body_captions=len(data['records']),frame_labels=len(data['frame_labels']),
        original_stroke_entities_hidden=len(moved),source_stroke_edges=len(index.edges),matched_stroke_edges=len(matched),
        fields_without_hidden_strokes=[key for key in strokes if not by_field[key]],field_coverage=coverage,
        layers={name:dict(default_visible=name!=ORIGINAL,color=color) for name,(color,_) in LAYERS.items()},
        text_fonts=dict(REVIEW_LATIN='arial.ttf',REVIEW_KOREAN='malgun.ttf'),
        display_change='Latin SHX replaced by filled Arial TrueType; fixed dark true colors replaced by theme-adaptive ACI colors. Recognition content unchanged.',
        table_exclusions_pdf=data['excluded_tables'],units='Original PDFIMPORT units; 1 CAD unit = 72 PDF points, no millimetre calibration.',
        limitations=['Recognition is experimental; complete drawing recall has not been established.',
                    'Caption text is fitted to its source extent; font appearance and word spacing differ from original stroke lettering.',
                    'Only fully matching source-text polylines are hidden. All original entities remain, and hidden outlines can be enabled.',
                    'No validation reference or dictionary used to rewrite text. Latest recorded source-stroke vowel disambiguation included.'],
        texts=texts,hidden_originals=moved,dxf_audit_errors=0)
    (OUT/'conversion_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DXF READY',len(texts),'editable labels',len(moved),'source polylines hidden',len(matched),'/',len(index.edges),'source edges',flush=True)
    dwg=OUT/(STEM+'.dwg')
    commands=['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name]
    if dwg.exists():commands.append('_Y')
    run_cad(dxf,'save_review',commands+['_.QUIT','_N'])
    if not dwg.exists() or dwg.stat().st_size<1000:raise AssertionError('DWG not saved')
    roundtrip=OUT/'dwg_roundtrip.dxf'
    if roundtrip.exists():raise ValueError('Roundtrip file already exists; preserve previous verification')
    log=run_cad(dwg,'verify_review',[
        '(setq ss (ssget "_X" (list (cons 410 "Model"))))',
        '(princ (strcat "\\nMODEL_ENTITIES=" (itoa (sslength ss))))',
        '(setq tx (ssget "_X" (list (cons 410 "Model") (cons 0 "TEXT,MTEXT"))))',
        '(princ (strcat "\\nEDITABLE_TEXT=" (itoa (sslength tx))))',
        '(setq ol (tblsearch "LAYER" "SOURCE_TEXT_OUTLINES"))',
        '(princ (strcat "\\nORIGINAL_OFF=" (if (< (cdr (assoc 62 ol)) 0) "1" "0")))',
        '_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
    content=log.read_bytes().decode('utf-16-le',errors='replace')
    counts=re.findall(r'MODEL_ENTITIES=(\d+)',content)
    if not counts or int(counts[-1])!=len(model):raise AssertionError('DWG entity count mismatch')
    if int(re.findall(r'EDITABLE_TEXT=(\d+)',content)[-1])!=native_count+len(texts):raise AssertionError('DWG text count mismatch')
    if re.findall(r'ORIGINAL_OFF=(\d)',content)[-1]!='1':raise AssertionError('Original outlines visible')
    rounddoc=ezdxf.readfile(roundtrip)
    assert not rounddoc.audit().has_errors
    roundtexts={e.dxf.handle:e.dxf.text for e in rounddoc.modelspace().query('TEXT')}
    assert all(roundtexts[r['handle']]==r['text'] for r in texts)
    manifest.update(dwg_reopened=True,dwg_roundtrip_text_verified=True,
                    dwg_sha256=hashlib.sha256(dwg.read_bytes()).hexdigest())
    (OUT/'conversion_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    readme='''최신 YOLO + 한글 OCR 검토용 도면

non_table_readable_yolo_ocr.dwg: 표 두 영역을 제외한 가독성 수정 DWG (AutoCAD 2018 형식).
본문 문구 48개와 테두리 표기 104개를 편집 가능한 TEXT로 넣었습니다. 원래 PDF 문자 59개는 유지했습니다.

레이어:
- YOLO_LATIN: 영문·숫자 본문 문구, Arial 글꼴. 검은 배경에서 흰색 / 흰 배경에서 검정색.
- YOLO_KOREAN: 한글 또는 한글·영문 혼합 문구, 맑은 고딕 글꼴, 자홍색.
- FRAME_TEXT: 도면 테두리 문자·숫자, Arial 글꼴, 밝은 청록색.
- SOURCE_TEXT_OUTLINES: 좌표가 일치한 원본 문자 획, 기본 꺼짐.

원본과 비교하려면 SOURCE_TEXT_OUTLINES를 켜고 위 세 문자 레이어를 끄세요.
원본 선·도형 객체는 삭제하지 않았습니다. 좌표가 완전히 대응하지 않는 문자 획은 남아 있을 수 있습니다.
인식한 문자열은 직전 파일과 동일합니다. 영문 글꼴과 화면 색상을 수정했습니다.
이 파일은 검토용입니다. 전체 영역의 문자 누락·오인식 검증은 끝나지 않았고, 글꼴과 띄어쓰기 모양은 원본과 다릅니다.
실제 치수(mm) 보정은 하지 않았으며, 기존 PDFIMPORT 좌표 단위를 유지했습니다.
'''
    (OUT/'읽어주세요.txt').write_text(readme,encoding='utf-8-sig')
    print('DWG VERIFIED',dwg.name,flush=True)


if __name__=='__main__':main()
