"""Probe all existing table crops; read frozen references only after inference."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_vertical_bar import OUT, alignment, boxes
from prepare import read_image, write_image
from preserved_reader import merge_predictions
from train import read_rows
from vertical_bar_reader import VerticalBarReader
from ultralytics.engine.results import Results

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'output/cad_table_benchmark'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bar-weights',type=Path)
    parser.add_argument('--output',type=Path,default=OUT/'tables')
    parser.add_argument('--adapt-large',action='store_true')
    parser.add_argument('--protect-words',action='store_true')
    parser.add_argument('--min-height-ratio',type=float)
    args=parser.parse_args()
    torch.set_num_threads(4);dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    frozen=dest/'frozen_predictions.json'
    if frozen.exists():
        records=json.loads(frozen.read_text(encoding='utf-8'))['records']
    else:
        cells=json.loads((BASE/'cells.json').read_text(encoding='utf-8'))['cells']
        reader=VerticalBarReader(args.bar_weights,adapt_large=args.adapt_large,protect_words=args.protect_words,min_height_ratio=args.min_height_ratio);records=[]
        for i,cell in enumerate(cells):
            image=read_image(BASE/'clean'/(cell['id']+'.png'))
            ink=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<200
            yy,xx=np.where(ink)
            if len(xx):image=image[yy.min():yy.max()+1,xx.min():xx.max()+1]
            scale=48/image.shape[0]
            image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            image=cv2.copyMakeBorder(image,8,8,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
            result,baseline,details=reader.predict(image,cell['id'],'tiles',384)
            old,_=merge_predictions(boxes(baseline),details['symbols'],details['cleaned'])
            before=Results(orig_img=image,path=cell['id'],names=reader.names,boxes=old)
            record=dict(id=cell['id'],table=cell['table'],before='\n'.join(read_rows(before)),after='\n'.join(read_rows(result)),bar_decisions=details['vertical_bar'])
            if details['vertical_bar']['accepted']:
                write_image(dest/(cell['id']+'_input.png'),image)
                write_image(dest/(cell['id']+'_after.png'),result.plot(line_width=1,font_size=10))
            records.append(record)
            if i%40==0:print('TABLE AUDIT',i+1,len(cells),flush=True)
        frozen.write_text(json.dumps(dict(answers_loaded=False,llm_inference=False,records=records),indent=2),encoding='utf-8')
    refdir=ROOT/'output/cad_table_font_training'
    refs=json.loads((refdir/'holdout.json').read_text(encoding='utf-8'))['answers']+json.loads((refdir/'new_check_reference.json').read_text(encoding='utf-8'))['answers']
    answers={r['id']:r for r in refs}
    changed=[r for r in records if r['before']!=r['after']]
    for rec in changed:
        if rec['id'] in answers:
            answer=answers[rec['id']]['reference']
            rec['reference']=answer
            rec['before_score']=alignment(answer,rec['before'])
            rec['after_score']=alignment(answer,rec['after'])
    scores={}
    for version in ['before','after']:
        scored=[alignment(answers[r['id']]['reference'],r[version]) for r in records if r['id'] in answers]
        n=sum(s['reference_characters'] for s in scored)
        errors=sum(s['edit_distance'] for s in scored)
        scores[version]=dict(cells=len(scored),exact=sum(s['exact'] for s in scored),characters=n,errors=errors,cer=errors/max(n,1))
    summary=dict(scope='Retrospective existing table diagnostic. Raw YOLO probes, not the final EasyOCR/Paddle/font-verifier table pipeline.',cells=len(records),evaluated_cells=len(answers),changed_cells=len(changed),accepted_bars=sum(len(r['bar_decisions']['accepted']) for r in records),
        scores=scores,regressions=[r['id'] for r in changed if 'after_score' in r and r['after_score']['edit_distance']>r['before_score']['edit_distance']],changes=changed)
    (dest/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
