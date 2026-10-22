"""Use exact PDF stroke lengths for tall |; no answer strings or LLM inference."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'output/cad_table_benchmark'


def vertical_evidence(box,reference,segments,min_ratio=1.15):
    x0,y0,x1,y1=box;height=y1-y0
    ratio=height/reference if reference>0 else 0
    if reference<=0 or ratio<min_ratio or x1-x0>reference*.04:
        return dict(accepted=False,height=height,reference_height=reference,height_ratio=ratio)
    tolerance=.025;center=(x0+x1)/2;intervals=[]
    for a,b,c,d in segments:
        if max(abs(a-center),abs(c-center))>tolerance:continue
        lo,hi=sorted((b,d))
        if lo>=y0-tolerance and hi<=y1+tolerance and hi>lo:intervals.append((max(y0,lo),min(y1,hi)))
    covered=0.;end=y0
    for lo,hi in sorted(intervals):
        covered+=max(0,hi-max(end,lo));end=max(end,hi)
    coverage=covered/height if height>0 else 0
    return dict(accepted=coverage>=.98,height=height,reference_height=reference,height_ratio=ratio,vertical_stroke_coverage=coverage)


def references_for(index,chars,glyphs):
    box=glyphs[index]['box_pdf'];references=[]
    for i,(char,glyph) in enumerate(zip(chars,glyphs)):
        if i==index or char not in 'ABCDEFGHJKLMNOPQRSTUVWXYZ023456789bdfhk':continue
        b=glyph['box_pdf'];h=b[3]-b[1]
        if h<=0 or b[2]-b[0]<h*.15:continue
        vertical=max(0,min(box[3],b[3])-max(box[1],b[1]))
        if vertical<h*.6:continue
        gap=max(box[0]-b[2],b[0]-box[2],0)
        if gap<=h*6:references.append(dict(index=i,height=h,gap=gap))
    return sorted(references,key=lambda r:r['gap'])[:8]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'output/cad_table_confusion_fixed/predictions.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/cad_bar_height_rule_v2/vector_corrections')
    parser.add_argument('--min-height-ratio',type=float,default=1.15)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(args.min_height_ratio) or args.min_height_ratio<=1:raise ValueError('Invalid ratio')
    source=json.loads(args.input.read_text(encoding='utf-8'))
    fields={r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    cells={r['id']:r for r in json.loads((BASE/'cells.json').read_text(encoding='utf-8'))['cells']}
    pdf=ROOT/'font_yolo_trial/cad_without_tables/source.pdf';page=fitz.open(pdf)[0]
    drawings=page.get_drawings();rows=deepcopy(source['results']);changes=[];review=[]
    for row in rows:
        identity=row['id'];field=fields.get(identity)
        if row['table']!='left' or field is None:continue
        chars=list(''.join(row['text'].split()));glyphs=field['glyphs']
        if len(chars)!=len(glyphs):
            review.append(dict(id=identity,reason='unaligned_character_count'));continue
        updates=[];segments=None
        for i,(char,glyph) in enumerate(zip(chars,glyphs)):
            if char not in 'Iil1|':continue
            refs=references_for(i,chars,glyphs)
            if len(refs)<2:continue
            cap=float(np.percentile([r['height'] for r in refs],75))
            b=glyph['box_pdf']
            if b[3]-b[1]<cap*args.min_height_ratio or b[2]-b[0]>cap*.04:continue
            if segments is None:
                cell=fitz.Rect(cells[identity]['bbox_pdf'])
                paths=[p for p in drawings if cell.contains(p['rect'])]
                if len(paths)!=1:break
                segments=[]
                for item in paths[0]['items']:
                    if item[0]=='l':
                        a,c=item[1:3];segments.append((a.x,a.y,c.x,c.y))
                    elif item[0]=='qu':
                        q=item[1]
                        segments.extend((a.x,a.y,c.x,c.y) for a,c in [(q.ul,q.ur),(q.ur,q.lr),(q.lr,q.ll),(q.ll,q.ul)])
            evidence=vertical_evidence(b,cap,segments,args.min_height_ratio)
            if evidence['accepted'] and char!='|':
                chars[i]='|'
                updates.append(dict(index=i,before=char,after='|',box_pdf=b,references=refs,**evidence))
        if updates:
            old=row['text'];iterator=iter(chars)
            row['text']=''.join(c if c.isspace() else next(iterator) for c in old)
            row['bar_height_updates']=updates
            changes.append(dict(id=identity,before=old,after=row['text'],updates=updates))
    report=dict(llm_inference=False,answers_loaded=False,training=False,min_height_ratio=args.min_height_ratio,
                source_predictions_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),source_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                scope='Vector table glyphs with one-to-one source character alignment and continuous tall vertical strokes. Raster-only regions use the separate image height gate.',
                results=rows,changes=changes,review=review)
    (args.output/'predictions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(changed_cells=len(changes),changed_characters=sum(len(c['updates']) for c in changes),changes=changes),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
