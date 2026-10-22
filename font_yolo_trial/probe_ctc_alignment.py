"""Inspect local OCR character time positions, without reference strings."""
import json
import sys
from pathlib import Path

import cv2
import pymupdf as fitz

from infer_real_validation import render

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from models_local import _offline
from paddle_ocr_local import model_dirs


def main():
    data=json.loads((ROOT.parent/'output/cad_fusion_iterations/automatic_results.json').read_text(encoding='utf-8'))
    page=fitz.open(data['source'])[0];jobs=[]
    for record in data['records']:
        if record['id'] not in ['P39225','P39104','P38554']:continue
        box=record['box_pdf'];scale=56/record['cap']
        image,origin=render(page,[box[0]-record['cap']*.15,box[1]-record['cap']*.15,box[2]+record['cap']*.15,box[3]+record['cap']*.15],scale)
        jobs.append((record,image,origin,scale))
    with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        records=[]
        for record,image,origin,scale in jobs:
            result=next(iter(model.predict(input=image,batch_size=1,return_word_box=True)))
            raw=result.json.get('res',result.json)
            records.append(dict(id=record['id'],shape=list(image.shape),origin=origin,scale=scale,raw=raw))
            print(record['id'],json.dumps(raw,ensure_ascii=True),flush=True)
    out=ROOT/'ocr_refinement';out.mkdir(exist_ok=True)
    (out/'ctc_probe.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
