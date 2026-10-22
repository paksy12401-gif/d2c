"""Separate original linear SVG subpaths before YOLO; never load validation answers."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import torch

from preserved_reader import PreservedReader
from infer_real_validation import OUT, ROOT, as_lines, inside
from prepare import write_image


def parse_linear_path(data, matrix):
    tokens=re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',data)
    if any(t.isalpha() and t.upper() not in 'MLHVZ' for t in tokens):return []
    a,b,c,d,e,f=matrix
    def point(x,y):return (a*x+c*y+e,b*x+d*y+f)
    runs=[];run=[];x=y=0.;start=(0.,0.);command=None;i=0
    while i<len(tokens):
        if tokens[i].isalpha():
            command=tokens[i];i+=1
            if command.upper()=='Z':
                if (x,y)!=start:run.append((*point(x,y),*point(*start)))
                x,y=start;continue
        upper=command.upper();relative=command.islower()
        if upper in 'ML':
            u,v=float(tokens[i]),float(tokens[i+1]);i+=2
            nx,ny=(x+u,y+v) if relative else (u,v)
        elif upper=='H':
            u=float(tokens[i]);i+=1;nx,ny=(x+u if relative else u),y
        elif upper=='V':
            v=float(tokens[i]);i+=1;nx,ny=x,(y+v if relative else v)
        else:return []
        if upper=='M':
            if run:runs.append(run)
            run=[];start=(nx,ny);command='l' if relative else 'L'
        else:run.append((*point(x,y),*point(nx,ny)))
        x,y=nx,ny
    if run:runs.append(run)
    return runs


def bounds(segments):
    return [min(min(s[0],s[2]) for s in segments),min(min(s[1],s[3]) for s in segments),
            max(max(s[0],s[2]) for s in segments),max(max(s[1],s[3]) for s in segments)]


def same_column(a,b,tolerance=.025):
    wa,wb=a[2]-a[0],b[2]-b[0]
    if wa<=tolerance and wb<=tolerance:return abs(a[0]-b[0])<=tolerance
    if wa<=tolerance:return b[0]-tolerance<=a[0]<b[2]-tolerance
    if wb<=tolerance:return a[0]-tolerance<=b[0]<a[2]-tolerance
    return min(a[2],b[2])-max(a[0],b[0])>tolerance


def group_glyphs(runs):
    groups=[]
    for run in runs:
        box=bounds(run);matches=[i for i,g in enumerate(groups) if same_column(box,g['box'])]
        if not matches:groups.append({'box':box,'segments':list(run)});continue
        first=matches[0];groups[first]['segments'].extend(run)
        for index in reversed(matches[1:]):groups[first]['segments'].extend(groups.pop(index)['segments'])
        groups[first]['box']=bounds(groups[first]['segments'])
    return sorted(groups,key=lambda g:g['box'][0])


def load_candidates(page, *, min_cap=4, max_cap=55):
    tree=ET.fromstring(page.get_svg_image(text_as_path=True));candidates=[]
    for index,node in enumerate(tree.iter()):
        if not node.tag.endswith('path') or 'stroke' not in node.attrib:continue
        transform=node.get('transform','matrix(1,0,0,1,0,0)')
        numbers=re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',transform)
        if not transform.startswith('matrix') or len(numbers)!=6:continue
        runs=parse_linear_path(node.get('d',''),list(map(float,numbers)))
        if not runs:continue
        groups=group_glyphs(runs)
        if len(groups)<3:continue
        heights=[g['box'][3]-g['box'][1] for g in groups if g['box'][3]>g['box'][1]+.05]
        if not heights:continue
        cap=float(np.median(sorted(heights)[len(heights)//3:]))
        box=bounds([s for run in runs for s in run])
        if not (min_cap<=cap<=max_cap and box[3]-box[1]<=cap*1.65 and box[2]-box[0]>=cap*2):continue
        baseline=float(np.median([g['box'][3] for g in groups if g['box'][3]-g['box'][1]>=cap*.7]))
        candidates.append(dict(path_id=index,box=box,cap=cap,baseline=baseline,
                               width=float(node.get('stroke-width','.37')),groups=groups))
    return candidates


def geometric_symbol(group,cap,baseline):
    a,b,c,d=group['box']
    if d-b<cap*.025 and c-a>cap*.2 and baseline-cap*.12<=b<=baseline+cap*.25:return '_'
    return None


def glyph_canvas(group,cap,target,width):
    scale=target/cap;a,b,c,d=group['box']
    image=np.full((384,384,3),255,dtype=np.uint8)
    # Supersampling preserves thin fractional-width vector strokes.
    high=np.full((384*3,384*3,3),255,dtype=np.uint8)
    origin=[192-(a+c)/2*scale,192-(b+d)/2*scale]
    for x0,y0,x1,y1 in group['segments']:
        p0=tuple(round((v*scale+origin[i])*3) for i,v in enumerate([x0,y0]))
        p1=tuple(round((v*scale+origin[i])*3) for i,v in enumerate([x1,y1]))
        cv2.line(high,p0,p1,(0,0,0),max(1,round(width*scale*3)),cv2.LINE_AA)
    image=cv2.resize(high,(384,384),interpolation=cv2.INTER_AREA)
    return image


def main():
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'regions.json').read_text(encoding='utf-8'))
    source=Path(manifest['source']);page=fitz.open(source)[0]
    candidates=load_candidates(page)
    print('TEXTLIKE_VECTOR_PATHS',len(candidates),flush=True)
    reader=PreservedReader();records=[];jobs=[]
    for region in manifest['regions']:
        if region['kind'] in ['korean','native_font_control']:continue
        for height in [32,40,56]:
            entry=dict(id=region['id'],target_height=height,boxes=[],glyphs=[],path_count=0)
            seen=[]
            for path in candidates:
                if not (path['box'][0]<=region['box_pdf'][2] and path['box'][2]>=region['box_pdf'][0] and
                        path['box'][1]<=region['box_pdf'][3] and path['box'][3]>=region['box_pdf'][1]):continue
                entry['path_count']+=1
                for group in path['groups']:
                    box=group['box']
                    if not inside(box,region['box_pdf'],region['reference_cap_height_pdf']*.035):continue
                    if any(all(abs(a-b)<.02 for a,b in zip(box,old)) for old in seen):continue
                    seen.append(box)
                    geometric=geometric_symbol(group,path['cap'],path['baseline'])
                    glyph=dict(box_pdf=box,path_id=path['path_id'],source='geometry_underscore' if geometric else 'isolated_vector_yolo')
                    entry['glyphs'].append(glyph)
                    if geometric:glyph.update(text=geometric,score=1.0)
                    else:jobs.append((glyph,glyph_canvas(group,path['cap'],height,path['width'])))
            records.append(entry)
    print('ISOLATED_GLYPH_JOBS',len(jobs),flush=True)
    # Classify independent glyphs in batches; neither references nor expected words are available here.
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16]
        predictions=reader.letters.predict([im for _,im in batch],imgsz=384,conf=.1,iou=.5,end2end=False,device=0,batch=16,verbose=False)
        for (glyph,image),result in zip(batch,predictions):
            boxes=result.boxes.cpu().numpy().data.tolist()
            boxes=[b for b in boxes if 145<=(b[0]+b[2])/2<=239 and 145<=(b[1]+b[3])/2<=239]
            best=max(boxes,key=lambda b:b[4],default=None)
            glyph.update(text=reader.names[int(best[5])] if best else '',score=best[4] if best else 0)
        print(f'GLYPHS {min(start+16,len(jobs))}/{len(jobs)}',flush=True)
    for entry in records:
        names={};boxes=[]
        for glyph in entry['glyphs']:
            if not glyph['text']:continue
            index=next((k for k,v in names.items() if v==glyph['text']),None)
            if index is None:index=len(names);names[index]=glyph['text']
            box=list(glyph['box_pdf'])
            if box[3]-box[1]<.01:box[1]-=.08;box[3]+=.08
            boxes.append([*box,glyph['score'],index])
        entry['raw_lines']=as_lines(boxes,names)
    result=dict(llm_inference=False,word_correction=False,training=False,paddle_proposal_gate=False,
                note='Experimental original-vector segmentation. Underscores are explicit geometry rules, not a learned YOLO class.',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                regions_sha256=hashlib.sha256((OUT/'regions.json').read_bytes()).hexdigest(),
                model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in reader.paths.items()},records=records)
    (OUT/'vector_predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VECTOR_PROBE COMPLETE',flush=True)


if __name__=='__main__':main()
