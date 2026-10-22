"""Read this PDF outside its two tables using vector geometry + YOLO + local OCR.

No validation-region or answer file is read. Geometry selection is heuristic;
the output is recognized text with coordinates, not a completed CAD conversion.
"""
import csv
import hashlib
import json
from pathlib import Path

import pymupdf as fitz
import torch
from ultralytics import YOLO

from infer_real_validation import inside
from iterate_fusion import ROOT, raster_fields, vector_fields, ocr_words, hybrid_glyphs, fuse_words, lines
from non_table_ocr import TABLES, intersects
from preserved_reader import PreservedReader
from vector_font_probe import load_candidates

OUT=ROOT.parent/'output/cad_fusion_iterations'


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'cad_without_tables/source.pdf';page=fitz.open(source)[0]
    candidates=[p for p in load_candidates(page) if not any(intersects(p['box'],t) for t in TABLES)]
    fields=[dict(id=f"P{p['path_id']}",box_pdf=p['box'],cap=p['cap']) for p in candidates]
    (OUT/'automatic_regions.json').write_text(json.dumps(dict(source=str(source.resolve()),selection='Linear stroked SVG paths with textlike grouping; no validation positions or expected strings.',excluded_tables=TABLES,fields=fields),ensure_ascii=False,indent=2),encoding='utf-8')
    reader=PreservedReader();trained_path=ROOT/'fusion_iterations/spacing_training/training/weights/best.pt';trained=YOLO(str(trained_path))
    glyph_records=[]
    for start in range(0,len(fields),12):
        subset=fields[start:start+12]
        raster=raster_fields(page,subset,reader,trained,heights=(56,),include_old=False)
        vector=vector_fields(page,subset,reader,heights=(40,),candidates=candidates)
        for field in subset:
            vr=next(r for r in vector if r['id']==field['id'])
            rr=next(r for r in raster if r['id']==field['id'])
            glyph_records.append(dict(id=field['id'],box_pdf=field['box_pdf'],cap=field['cap'],
                glyphs=hybrid_glyphs(vr['glyphs'],rr['glyphs'],field['cap'])))
        (OUT/'automatic_glyph_checkpoint.json').write_text(json.dumps(glyph_records,ensure_ascii=False),encoding='utf-8')
        print('AUTOMATIC FIELDS',min(start+12,len(fields)),'/',len(fields),flush=True)
    words=ocr_words(page,fields)
    detections=[]
    for record in glyph_records:
        glyphs,suppressed=fuse_words(record['glyphs'],words[record['id']])
        record['raw_lines']=lines(record['glyphs']);record['fused_lines']=lines(glyphs)
        record['fused_glyphs']=glyphs;record['suppressed']=suppressed
        for glyph in glyphs:
            if not any(inside(glyph['box_pdf'],t) for t in TABLES):detections.append({**glyph,'field_id':record['id']})
    # Native PDF text is already an exact character source; do not OCR it again.
    native=[]
    for word in page.get_text('words'):
        if any(intersects(word[:4],t) for t in TABLES):continue
        native.append(dict(text=word[4],box_pdf=list(word[:4]),score=1.,engine='native_pdf_text',field_id='native'))
    detections=[g for g in detections if not any(inside(g['box_pdf'],w['box_pdf']) for w in native)]+native
    # Suppress only near-identical same-text duplicates across source paths.
    kept=[]
    for glyph in sorted(detections,key=lambda g:-g['score']):
        a=glyph['box_pdf']
        if any(glyph['text']==g['text'] and max(abs(x-y) for x,y in zip(a,g['box_pdf']))<.25 for g in kept):continue
        kept.append(glyph)
    result=dict(scope='Automatic vector-candidate search outside two manually specified table rectangles. Not guaranteed full text coverage; raster-only/nonlinear paths without candidates are not covered by this command.',
        source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inference_llm=False,answers_loaded=False,word_correction=False,
        original_model_hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in reader.paths.items()},
        trained_model_sha256=hashlib.sha256(trained_path.read_bytes()).hexdigest(),
        candidate_fields=len(fields),excluded_tables=TABLES,words=words,records=glyph_records,detections=kept)
    (OUT/'automatic_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'recognized_text.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['field_id','text','score','engine','x0','y0','x1','y1'])
        for glyph in sorted(kept,key=lambda g:(g['box_pdf'][1],g['box_pdf'][0])):
            writer.writerow([glyph['field_id'],glyph['text'],glyph['score'],glyph['engine'],*glyph['box_pdf']])
    (OUT/'automatic_text.txt').write_text('\n'.join(f"{r['id']}: {' / '.join(r['fused_lines'])}" for r in glyph_records),encoding='utf-8')
    print('AUTOMATIC COMPLETE',len(fields),'fields',len(kept),'detections',flush=True)


if __name__=='__main__':main()
