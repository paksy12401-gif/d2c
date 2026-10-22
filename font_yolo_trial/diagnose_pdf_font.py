import json,sys
from pathlib import Path
import numpy as np,pymupdf as fitz,torch
sys.path.insert(0,str(Path.cwd()/'font_yolo_trial'))
from preserved_reader import PreservedReader
from prepare import write_image
from train import read_rows
root=Path('font_yolo_trial');work=root/'cad_without_tables'
r=json.loads((work/'ocr_results.json').read_text(encoding='utf-8'))
body=[p for p in r['proposals'] if p['box'][0]<3590 and p['box'][2]>3430 and p['box'][1]<425 and p['box'][3]>400]
print('PROPOSALS_OVER_PREVIOUS_SLEEVE_REGION',json.dumps(body,ensure_ascii=True),flush=True)
torch.set_num_threads(4);reader=PreservedReader();page=fitz.open(work/'source.pdf')[0]
out=root/'diagnose_pdf_font';out.mkdir(exist_ok=True)
results=[]
for index in [29,105]:
 p=r['proposals'][index];x0,y0,x1,y1=p['box'];h=y1-y0
 clip=fitz.Rect(x0-h*.3,y0-h*.6,x1+h*.3,y1+h*.6)
 for target in [32,40,56,72]:
  scale=target/h;pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=clip,alpha=False)
  im=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)[:,:,::-1].copy()
  result,base,_=reader.predict(im,mode='tiles')
  row={'proposal':index,'target_height':target,'paddle_raw':p['paddle_raw'],'baseline':read_rows(base),'merged':read_rows(result)}
  results.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
  write_image(out/f'p{index}_h{target}.png',result.plot(line_width=1,font_size=10))
(out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
