"""Attach saved, uncorrected table predictions to the verified non-table CAD.

No answer sheets are loaded. Original left-table strokes remain on an off layer.
The right raster table is rebuilt as editable lines and text, without an image link.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import cv2
import ezdxf
from ezdxf import bbox, path
from ezdxf.addons import text2path
from ezdxf.fonts.fonts import FontFace
import numpy as np
import pymupdf as fitz

import export_fusion_review_cad as cad

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output/cad_with_tables_ocr'
BASE = ROOT / 'output/cad_readability_fixed'
TABLES = ROOT / 'output/cad_table_benchmark'
PREDICTIONS = ROOT / 'output/cad_table_font_training/combined_predictions.json'
STEM = 'drawing_with_tables_yolo_ocr'
HIDDEN = 'TABLE_SOURCE_OUTLINES'


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


@lru_cache(maxsize=1024)
def font_bounds(text):
    bounds = path.bbox(text2path.make_paths_from_str(text, FontFace(filename='arial.ttf'), size=1))
    if not bounds.has_data:
        raise ValueError('Table prediction has no visible glyphs')
    return bounds.extmin.x, bounds.extmin.y, bounds.extmax.x, bounds.extmax.y


def add_fitted_text(model, text, source_box, page_height, layer, identity):
    """Fit actual TrueType ink extents, including descenders, inside source ink."""
    x0, y0, x1, y1 = source_box
    ax, ay, bx, by = font_bounds(text)
    height = (y1-y0)/72/(by-ay)
    width = (x1-x0)/72/((bx-ax)*height)
    assert height > 0 and .01 < width < 100, (identity, height, width)
    entity = model.add_text(text, dxfattribs=dict(
        layer=layer, style='TABLE_ARIAL', height=height, width=width, color=256,
        insert=(x0/72-ax*height*width, (page_height-y1)/72-ay*height, 0)))
    entity.set_xdata('CAD_TABLE_OCR', [(1000, identity)])
    return entity


def right_ink_box(cell, meta):
    image = cv2.imdecode(np.fromfile(TABLES/'cells'/(cell['id']+'.png'), np.uint8), 0)
    yy, xx = np.where(image < 150)
    assert len(xx), cell['id']
    x0, y0, x1, y1 = meta['right']['bbox_pdf']
    sx, sy = (x1-x0)/1204, (y1-y0)/1584
    # Crops in table_cell_experiment.prepare start two pixels inside each cell.
    px, py = cell['bbox_px'][:2]
    return [x0+(px+2+xx.min())*sx, y0+(py+2+yy.min())*sy,
            x0+(px+3+xx.max())*sx, y0+(py+3+yy.max())*sy]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dwg = OUT/(STEM+'.dwg')
    roundtrip = OUT/'dwg_roundtrip.dxf'
    if dwg.exists() or roundtrip.exists():
        raise FileExistsError('Preserve previous DWG and roundtrip verification')
    base_report = json.loads((BASE/'conversion_report.json').read_text(encoding='utf-8'))
    assert sha(BASE/'non_table_readable_yolo_ocr.dwg') == base_report['dwg_sha256']
    source = ROOT/'font_yolo_trial/cad_without_tables/source.pdf'
    assert sha(source) == base_report['source_pdf_sha256']
    predictions = json.loads(PREDICTIONS.read_text(encoding='utf-8'))
    assert predictions['llm_inference'] is False
    assert predictions['heldout_labels_loaded'] is False
    assert predictions['inference_uses_training_transcripts'] is False
    rows = sorted(predictions['results'], key=lambda r:(r['table'],r['row'],r['column']))
    assert len(rows) == len({r['id'] for r in rows}) == 880
    meta = json.loads((TABLES/'cells.json').read_text(encoding='utf-8'))
    assert meta['source_sha256'] == sha(source)
    cells = {c['id']:c for c in meta['cells']}
    glyphs = {r['id']:r for r in json.loads((TABLES/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    page = fitz.open(source)[0]
    height = page.rect.height
    drawings = page.get_drawings()
    fields = {}
    for cell in meta['cells']:
        if cell['table'] != 'left':
            continue
        box = fitz.Rect(cell['bbox_pdf'])
        candidates = [d for d in drawings if box.contains(d['rect'])]
        assert len(candidates) == 1, cell['id']
        segments = []
        for item in candidates[0]['items']:
            if item[0] == 'l':
                a,b = item[1:3]
                segments.append((a.x,a.y,b.x,b.y))
            elif item[0] == 'qu':
                q = item[1]
                segments.extend((a.x,a.y,b.x,b.y) for a,b in [(q.ul,q.ur),(q.ur,q.lr),(q.lr,q.ll),(q.ll,q.ul)])
            else:
                raise ValueError(item[0])
        fields[cell['id']] = segments
    index = cad.StrokeIndex(fields)
    doc = ezdxf.readfile(BASE/'dwg_roundtrip.dxf')
    model = doc.modelspace()
    base_count = len(model)
    base_text = {e.dxf.handle:(e.dxftype(), e.dxf.text if e.dxftype()=='TEXT' else e.text)
                 for e in model.query('TEXT MTEXT')}
    for layer,color in [('TABLE_LEFT_SOURCE',7),(HIDDEN,8),('TABLE_RIGHT_GRID',8),('TABLE_LEFT_TEXT',7),('TABLE_RIGHT_TEXT',7)]:
        assert layer not in doc.layers
        doc.layers.new(layer,dxfattribs=dict(color=color))
    doc.styles.new('TABLE_ARIAL',dxfattribs=dict(font='arial.ttf'))
    doc.appids.new('CAD_TABLE_OCR')
    raw = ezdxf.readfile(ROOT/'font_yolo_trial/cad_without_tables/raw_import.dxf')
    excluded = json.loads((ROOT/'font_yolo_trial/cad_without_tables/conversion_report.json').read_text(encoding='utf-8'))['removed_entities']
    restored = [];matched=set();by_cell=Counter()
    for record in excluded:
        if record['type'] == 'IMAGE':
            continue
        old = raw.entitydb[record['handle']]
        assert old.dxftype() == 'LWPOLYLINE'
        entity = old.copy()
        entity.dxf.layer = 'TABLE_LEFT_SOURCE'
        doc.modelspace().add_foreign_entity(entity, copy=False)
        vertices = list(entity.get_points('xyb'))
        identity,edges = (None,[])
        if not any(abs(v[2])>1e-10 for v in vertices):
            points=[(float(v[0])*72,height-float(v[1])*72) for v in vertices]
            identity,edges=index.match(points,entity.closed)
        if identity:
            entity.dxf.layer=HIDDEN
            entity.dxf.color=256
            entity.dxf.discard('true_color')
            matched.update(edges);by_cell[identity]+=1
        restored.append(dict(original_handle=record['handle'],handle=entity.dxf.handle,cell=identity))
    assert len(restored)==3504
    assert len(matched)==len(index.edges) and len(by_cell)==160, (len(matched),len(index.edges),len(by_cell))
    doc.layers.get(HIDDEN).off()
    rx0,ry0,rx1,ry1=meta['right']['bbox_pdf']
    def point(x,y):
        return ((rx0+x*(rx1-rx0)/1204)/72, (height-ry0-y*(ry1-ry0)/1584)/72,0)
    grid=[]
    for x in meta['right']['xs']:
        grid.append(model.add_line(point(x,0),point(x,1584),dxfattribs=dict(layer='TABLE_RIGHT_GRID')))
    for y in meta['right']['ys']:
        grid.append(model.add_line(point(0,y),point(1204,y),dxfattribs=dict(layer='TABLE_RIGHT_GRID')))
    assert len(grid)==84
    texts=[]
    for row in rows:
        text=row['text']
        assert text.strip() and '\n' not in text and '\r' not in text, row['id']
        if row['table']=='left':
            boxes=[g['box_pdf'] for g in glyphs[row['id']]['glyphs']]
            box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        else:
            box=right_ink_box(cells[row['id']],meta)
        entity=add_fitted_text(model,text,box,height,'TABLE_'+row['table'].upper()+'_TEXT',row['id'])
        texts.append(dict(id=row['id'],table=row['table'],row=row['row'],column=row['column'],text=text,
                          handle=entity.dxf.handle,box_pdf=list(map(float,box)),layer=entity.dxf.layer))
    doc.header['$LWDISPLAY']=False
    doc.set_modelspace_vport(height=height/72*1.06,center=(page.rect.width/144,height/144))
    expected=base_count+len(restored)+len(grid)+len(texts)
    assert len(model)==expected
    dxf=OUT/(STEM+'.dxf');doc.saveas(dxf)
    assert not ezdxf.readfile(dxf).audit().has_errors
    manifest=dict(llm_inference=False,answers_used_for_export=False,predictions_sha256=sha(PREDICTIONS),
        source_pdf_sha256=sha(source),base_dwg_sha256=base_report['dwg_sha256'],
        base_entities=base_count,base_editable_text=len(base_text),table_cells=len(texts),
        left_table_text=160,right_table_text=720,restored_left_entities=len(restored),
        hidden_left_text_entities=sum(by_cell.values()),matched_source_edges=len(matched),
        right_grid_lines=len(grid),external_image_references=0,expected_entities=expected,
        original_left_title_and_cancel_line_preserved=True,texts=texts,restored_entities=restored,
        units='Original PDFIMPORT units; no physical millimetre calibration.',
        known_limitation='Saved OCR candidates with recorded model/source-shape guards; only 308 cells have source-based evaluation, not unseen-drawing validation.')
    (OUT/'conversion_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DXF_READY',expected,'entities',len(texts),'table texts',flush=True)
    cad.OUT=OUT
    cad.run_cad(dxf,'save_tables',['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
    assert dwg.exists() and dwg.read_bytes()[:6]==b'AC1032'
    log=cad.run_cad(dwg,'verify_tables',[
        '(setq ss (ssget "_X" (list (cons 410 "Model"))))',
        '(princ (strcat "\\nMODEL_ENTITIES=" (itoa (sslength ss))))',
        '(setq tx (ssget "_X" (list (cons 410 "Model") (cons 0 "TEXT,MTEXT"))))',
        '(princ (strcat "\\nEDITABLE_TEXT=" (itoa (sslength tx))))',
        '_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
    content=log.read_bytes().decode('utf-16-le',errors='replace')
    assert int(re.findall(r'MODEL_ENTITIES=(\d+)',content)[-1])==expected
    assert int(re.findall(r'EDITABLE_TEXT=(\d+)',content)[-1])==len(base_text)+880
    actual=ezdxf.readfile(roundtrip);space=actual.modelspace()
    assert not actual.audit().has_errors and len(space)==expected
    for handle,(kind,text) in base_text.items():
        entity=actual.entitydb[handle]
        assert entity.dxftype()==kind
        assert (entity.dxf.text if kind=='TEXT' else entity.text)==text
    bounds_cache=bbox.Cache();outside=[]
    for r in texts:
        e=actual.entitydb[r['handle']]
        assert e.dxftype()=='TEXT' and e.dxf.text==r['text']
        assert list(e.get_xdata('CAD_TABLE_OCR'))[0].value==r['id']
        assert not actual.layers.get(e.dxf.layer).is_off()
        cell=cells[r['id']]['bbox_pdf'];b=bbox.extents([e],cache=bounds_cache)
        pdfbox=[b.extmin.x*72,height-b.extmax.y*72,b.extmax.x*72,height-b.extmin.y*72]
        if any([pdfbox[0]<cell[0]-.6,pdfbox[1]<cell[1]-.6,pdfbox[2]>cell[2]+.6,pdfbox[3]>cell[3]+.6]):
            outside.append(dict(id=r['id'],box=pdfbox,cell=cell))
    assert not outside, outside[:4]
    assert actual.layers.get(HIDDEN).is_off()
    assert actual.layers.get('SOURCE_TEXT_OUTLINES').is_off()
    assert not len(space.query('IMAGE'))
    assert sha(BASE/'non_table_readable_yolo_ocr.dwg')==base_report['dwg_sha256']
    manifest.update(dwg_reopened=True,dwg_roundtrip_text_verified=True,base_text_preserved=True,
        table_text_placement_within_cells=True,dxf_audit_errors=0,dwg_sha256=sha(dwg))
    (OUT/'conversion_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DWG_VERIFIED',dwg.name,flush=True)


if __name__=='__main__':
    main()
