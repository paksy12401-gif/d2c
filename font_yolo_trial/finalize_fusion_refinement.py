"""Combine machine readings and source-geometry symbols; never read references."""
import csv
import hashlib
import json

import pymupdf as fitz

from ctc_korean_fusion import ROOT,OUT,BASE
from iterate_fusion import fuse_words,lines
from vector_font_probe import load_candidates
from vector_extra_symbols import supplement


def main():
    source_file=OUT/'refined_results.json';data=json.loads(source_file.read_text(encoding='utf-8'))
    page=fitz.open(data['source'])[0];paths={f"P{p['path_id']}":p for p in load_candidates(page)}
    records=[];detections=[]
    for old in data['records']:
        words=data['words'][old['id']]
        glyphs=supplement(old['glyphs'],paths[old['id']],words)
        fused,suppressed=fuse_words(glyphs,words)
        record={**old,'glyphs':glyphs,'fused_glyphs':fused,'suppressed':suppressed,'fused_lines':lines(fused)}
        records.append(record);detections.extend({**g,'field_id':old['id']} for g in fused)
    native=[g for g in data['detections'] if g['engine']=='native_pdf_text'];detections.extend(native)
    frame_data=json.loads((OUT/'frame_results.json').read_text(encoding='utf-8'))
    frame_glyphs=[dict(text=r['text'],box_pdf=r['box_pdf'],score=r['ocr_score'] if r['selected_engine'].startswith('ocr') else r['yolo_score'],engine=r['selected_engine'],field_id=r['id']) for r in frame_data['records'] if r['text']]
    detections.extend(frame_glyphs)
    result={**data,'records':records,'detections':detections,'frame_labels':frame_data['records'],
        'refinement_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
        'method':'Spaced-vector YOLO + trained symbol supplements + CTC-positioned Korean + additional geometric symbols + short border labels.',
        'inference_llm':False,'answers_loaded':False,'word_correction':False,'sequence_correction':False}
    (OUT/'combined_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'recognized_text.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['field_id','text','score','engine','x0','y0','x1','y1'])
        for g in sorted(detections,key=lambda g:(g['box_pdf'][1],g['box_pdf'][0])):writer.writerow([g['field_id'],g['text'],g['score'],g['engine'],*g['box_pdf']])
    (OUT/'recognized_lines.txt').write_text('\n'.join(f"{r['id']}: {' / '.join(r['fused_lines'])}" for r in records),encoding='utf-8')
    print('COMBINED REFINEMENT COMPLETE',len(records),'caption fields',len(frame_glyphs),'border labels',flush=True)


if __name__=='__main__':main()
