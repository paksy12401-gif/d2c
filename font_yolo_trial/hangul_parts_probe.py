"""Inspect Hangul vowel parts and CTC alternatives; no reference strings or lexicon."""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from ctc_korean_fusion import attach_confidence,vector_line_image
from stroke_parts_probe import normalized_skeleton,axis_parts
from vector_font_probe import load_candidates

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output/cad_hangul_parts'
VOWEL_PAIRS={8:11,13:16,18:19}


def decompose(char):
    value=ord(char)-0xAC00
    if not 0<=value<11172:return None
    return value//588,(value//28)%21,value%28


def add_vertical_vowel(char):
    parts=decompose(char)
    if parts is None:return None
    initial,vowel,final=parts
    if vowel not in VOWEL_PAIRS:return None
    return chr(0xAC00+initial*588+VOWEL_PAIRS[vowel]*28+final)


def right_vertical(image):
    skeleton=normalized_skeleton(image);ys,xs=np.where(skeleton>0)
    if not len(xs):return dict(present=False,parts=[])
    height=float(np.ptp(ys)+1);width=float(np.ptp(xs)+1)
    parts=axis_parts(skeleton)['vertical']
    matched=[p for p in parts if p[0]>=xs.min()+width*.67 and p[2]-p[1]>=height*.68
             and p[1]<=ys.min()+height*.18 and p[2]>=ys.max()-height*.18]
    return dict(present=bool(matched),parts=matched)


def attach_alternatives(model):
    decoder=model.paddlex_predictor.post_op
    characters=decoder.character
    attach_confidence(model)
    original=model.paddlex_predictor.post_op
    def postprocess(pred,**kwargs):
        texts,scores=original(pred,**kwargs)
        probabilities=np.asarray(pred[0]);ids=probabilities.argmax(-1)
        for i,value in enumerate(texts):
            if not isinstance(value,tuple):continue
            text,info=value
            selected=np.r_[True,ids[i,1:]!=ids[i,:-1]] & (ids[i]!=0)
            alternatives=[]
            for row in probabilities[i,selected]:
                indices=np.argpartition(-row,7)[:8]
                indices=indices[np.argsort(-row[indices])]
                alternatives.append([dict(text=characters[int(j)],score=float(row[j])) for j in indices])
            if len(alternatives)!=len(text):raise ValueError('CTC alternatives alignment')
            texts[i]=(text,[*info,alternatives])
        return texts,scores
    model.paddlex_predictor.post_op=postprocess


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
    from prepare import write_image
    data=json.loads((ROOT.parent/'output/cad_fusion_refined/combined_results.json').read_text(encoding='utf-8'))
    page=fitz.open(data['source'])[0];paths={f"P{p['path_id']}":p for p in load_candidates(page)}
    eligible=[];diagnostics=[];jobs=[]
    for identity,words in data['words'].items():
        path=paths[identity];cap=path['cap']
        for index,word in enumerate(words):
            text=word['accepted']
            if not re.fullmatch('[가-힣]+',text):continue
            letters=[i for i,c in enumerate(text) if decompose(c)[1] in set(VOWEL_PAIRS)|set(VOWEL_PAIRS.values()) and decompose(c)[2]==0]
            if not letters:continue
            wb=word['box_pdf'];n=len(text)
            entry=dict(id=f'{identity}_W{index}',field_id=identity,word_index=index,text=text,
                       box_pdf=wb,char_candidates=[],geometry=[])
            for height,pad in [(40,.15),(56,.25),(80,.35)]:
                full,origin,scale=vector_line_image(path,height,pad)
                left=max(0,int(np.floor((wb[0]-origin[0])*scale)))
                right=min(full.shape[1],int(np.ceil((wb[2]-origin[0])*scale)))
                crop=full[:,left:right]
                if not crop.size:raise ValueError('Empty Korean word')
                # Equal-width cells are a hypothesis for this CAD Hangul font, not guaranteed segmentation.
                for pos in letters:
                    a=round(pos*crop.shape[1]/n);b=round((pos+1)*crop.shape[1]/n)
                    glyph=crop[:,a:b]
                    geometry=right_vertical(cv2.copyMakeBorder(glyph,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255)))
                    entry['geometry'].append(dict(index=pos,char=text[pos],height=height,**geometry))
                    if height==56:write_image(OUT/'assets'/f'{entry["id"]}_C{pos}.png',glyph)
                word_image=cv2.copyMakeBorder(crop,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
                jobs.append((entry,height,word_image))
            eligible.append(entry)
    sys.path.insert(0,str(ROOT.parent))
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        attach_alternatives(model)
        for index,(entry,height,image) in enumerate(jobs):
            result=next(iter(model.predict(input=image,batch_size=1,return_word_box=True)))
            raw=result.json.get('res',result.json)
            entry['char_candidates'].append(dict(height=height,raw=raw))
            if index%20==0:print('HANGUL PARTS',index,'/',len(jobs),flush=True)
    result=dict(inference_llm=False,answers_loaded=False,training=False,
                scope='Exploratory Hangul horizontal/combined vowel layout check; CTC alternatives retained, no automatic text changes.',
                limits='Equal-width character cells and right-stem heuristic may fail on other fonts or final consonants. Final consonant syllables excluded.',records=eligible)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('HANGUL PARTS COMPLETE',len(eligible),'words',flush=True)


if __name__=='__main__':main()
