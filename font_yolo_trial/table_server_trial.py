"""Compare the official server recognizer locally; no validation references."""
import json
import hashlib
import time
from pathlib import Path
import shutil

import cv2

from refine_table_ocr import BASE, ROOT, image_for

OUT=ROOT/'output/cad_table_font_training'


def main():
    from models_local import _offline
    model_info=json.loads((OUT/'server_model_provenance.json').read_text())
    runtime=Path(model_info['model_dir']);runtime.mkdir(parents=True,exist_ok=True)
    for name,meta in model_info['files'].items():
        target=runtime/name
        if not target.exists():shutil.copy2(Path(model_info['archive_dir'])/name,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==meta['sha256']
    cells=json.loads((BASE/'cells.json').read_text())['cells']
    jobs={}
    for cell in cells:
        for variant in ['normal','more_ink']:
            image=cv2.cvtColor(image_for(cell,variant),cv2.COLOR_GRAY2BGR)
            digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            jobs.setdefault(digest,dict(image=image,addresses=[]))['addresses'].append((cell,variant))
    results=[];started=time.monotonic()
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        model=TextRecognition(model_name='PP-OCRv5_server_rec',model_dir=model_info['model_dir'],
            device='cpu',enable_mkldnn=False,cpu_threads=4)
        ordered=sorted(jobs,key=lambda d:jobs[d]['image'].shape[1])
        for start in range(0,len(ordered),8):
            batch=ordered[start:start+8]
            predictions=list(model.predict(input=[jobs[d]['image'] for d in batch],batch_size=8))
            assert len(predictions)==len(batch)
            for digest,prediction in zip(batch,predictions):
                raw=prediction.json.get('res',prediction.json)
                for cell,variant in jobs[digest]['addresses']:
                    results.append(dict(id=cell['id'],table=cell['table'],row=cell['row'],column=cell['column'],
                        variant=variant,text=str(raw['rec_text']),score=float(raw['rec_score'])))
            data=dict(model='PP-OCRv5_server_rec',llm_inference=False,answers_loaded=False,
                      elapsed_seconds=time.monotonic()-started,results=results)
            target=OUT/'server_predictions.json';tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(target)
            print('SERVER OCR',len(results),'/ 1760',round(time.monotonic()-started),'seconds',flush=True)
    assert len(results)==1760


if __name__=='__main__':main()
