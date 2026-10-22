"""Retry automatically located Korean spans with local OCR, no expected words."""
import json,math,re,sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from models_local import _offline
from paddle_ocr_local import model_dirs
from ctc_korean_fusion import attach_confidence,choose_spans
from read_table_title_local import spans
from refine_table_ocr import ctc_details
from iterate_fusion import fuse_words,lines
from vector_extra_symbols import supplement

OUT=ROOT/'output/cad_table_title_local'


def crop_image(field,box,stroke,stretch):
    scale=64/field['cap'];pad=.15*field['cap']
    origin=[box[0]-pad,box[1]-pad]
    width=math.ceil((box[2]-box[0]+2*pad)*scale)
    height=math.ceil((box[3]-box[1]+2*pad)*scale)
    high=np.full((height*3,width*3,3),255,np.uint8)
    for group in field['groups']:
        for a,b,c,d in group['segments']:
            first=tuple(round((v-origin[i])*scale*3) for i,v in enumerate([a,b]))
            last=tuple(round((v-origin[i])*scale*3) for i,v in enumerate([c,d]))
            cv2.line(high,first,last,(0,0,0),round(stroke*3),cv2.LINE_AA)
    image=cv2.resize(high,(round(width*stretch),height),interpolation=cv2.INTER_AREA)
    return image,origin,scale*stretch


def main():
    torch.set_num_threads(4)
    data=json.loads((OUT/'recognition.json').read_text(encoding='utf-8'));field=data['field'];cap=field['cap']
    candidates=[]
    for record in data['ocr_diagnostics']:
        text,info=record['raw']['rec_text'];columns=float(info[0]);positions=[x for g in info[2] for x in g]
        im=cv2.imdecode(np.fromfile(OUT/(record['variant']+'.png'),np.uint8),1)
        height=int(re.search(r'h(\d+)',record['variant']).group(1));pad={40:.15,56:.25,80:.35}[height]
        scale=height/cap;left=field['box'][0]-pad*cap
        origin=math.floor(left*scale)/scale if record['variant'].startswith('native') else left
        for m in re.finditer(r'[\uac00-\ud7a3]+',text):
            a,b=m.span();x0=max(field['box'][0],origin+positions[a]/columns*im.shape[1]/scale-cap*.6)
            x1=min(field['box'][2],origin+positions[b-1]/columns*im.shape[1]/scale+cap*.6)
            candidates.append([x0,field['box'][1],x1,field['box'][3]])
    windows=[]
    for box in sorted(candidates):
        match=next((g for g in windows if min(g[2],box[2])-max(g[0],box[0])>cap*.3),None)
        if match:match[0]=min(match[0],box[0]);match[2]=max(match[2],box[2])
        else:windows.append(box.copy())
    jobs=[]
    for i,box in enumerate(windows):
        for stroke in [1.,1.5,2.,3.]:
            for stretch in [1.,1.25]:
                image,origin,scale=crop_image(field,box,stroke,stretch)
                variant=f'word{i}_s{stroke}_x{stretch}'
                cv2.imencode('.png',image)[1].tofile(OUT/(variant+'.png'))
                jobs.append((variant,image,origin,scale))
    proposals=[{**p,'variant':v} for p in data['korean_spans'] for v in p['variants']]
    diagnostics=[]
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        import easyocr
        from easyocr.recognition import get_text
        from easyocr.config import imgH
        reader=easyocr.Reader(['ko','en'],gpu=True,detector=False,download_enabled=False,
            model_storage_directory=str(ROOT/'models/easyocr'),user_network_directory=str(ROOT/'models/easyocr/user_network'),verbose=False)
        captured=[]
        hook=reader.recognizer.register_forward_hook(lambda module,inputs,output:captured.append(output.detach().float().softmax(-1).cpu().numpy()))
        for variant,image,origin,scale in jobs:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);captured.clear()
            width=math.ceil(gray.shape[1]/gray.shape[0])*imgH
            values=get_text(reader.character,imgH,width,reader.recognizer,reader.converter,
                [([[0,0],[gray.shape[1],0],[gray.shape[1],gray.shape[0]],[0,gray.shape[0]]],gray)],
                decoder='greedy',batch_size=1,workers=0,device=reader.device,contrast_ths=0)
            _,text,score=values[0];detail=ctc_details(captured[0][0],reader.converter.character)
            assert text==detail['text']
            info=[detail['columns'],[list(text)],[[.5*(c['column']+c['end_column']) for c in detail['chars']]],['symbol'],[c['score'] for c in detail['chars']]]
            raw=dict(rec_text=[text,info],rec_score=float(score));found=spans(raw,image.shape[1],origin,scale,field)
            proposals.extend({**p,'variant':'easy_'+variant} for p in found)
            diagnostics.append(dict(variant='easy_'+variant,raw=raw,spans=found))
            print('EASY',variant,json.dumps(text,ensure_ascii=True),round(float(score),3),flush=True)
        hook.remove();del reader
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        attach_confidence(model)
        for variant,image,origin,scale in jobs:
            result=next(iter(model.predict(input=image,batch_size=1,return_word_box=True)))
            raw=result.json.get('res',result.json);found=spans(raw,image.shape[1],origin,scale,field)
            proposals.extend({**p,'variant':'paddle_'+variant} for p in found)
            diagnostics.append(dict(variant='paddle_'+variant,raw=raw,spans=found))
            print('PADDLE',variant,json.dumps(raw['rec_text'][0],ensure_ascii=True),round(float(raw['rec_score']),3),flush=True)
    selected=choose_spans(proposals,cap)
    combined=[]
    for run in data['results']:
        glyphs=supplement(run['glyphs'],field,selected)
        fused,suppressed=fuse_words(glyphs,selected)
        combined.append(dict(config=run['config'],fused_lines=lines(fused),glyphs=glyphs,fused_glyphs=fused,suppressed=suppressed))
    report=dict(llm_inference=False,answers_loaded=False,word_correction=False,windows=windows,diagnostics=diagnostics,korean_spans=selected,results=combined)
    (OUT/'retry_recognition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RETRY_RESULT',json.dumps([{k:r[k] for k in ['config','fused_lines']} for r in combined],ensure_ascii=True),flush=True)


if __name__=='__main__':main()
