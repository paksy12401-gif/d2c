import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from models_local import _offline
from paddle_ocr_local import model_dirs
import pymupdf as f,numpy as np
out=Path('font_yolo_trial/cad_without_tables').resolve()
p=f.open(out/'source.pdf')[0].get_pixmap(matrix=f.Matrix(1,1),clip=f.Rect(1800,520,2800,1140),alpha=False)
im=np.frombuffer(p.samples,np.uint8).reshape(p.height,p.width,3)[:,:,::-1].copy()
with _offline(paddle_cache=Path.cwd()/'.model_cache/paddlex'):
 from paddleocr import TextDetection
 det,_=model_dirs()
 m=TextDetection(model_name='PP-OCRv5_server_det',model_dir=str(det),device='cpu',cpu_threads=4,enable_mkldnn=True)
 t=time.time();r=m.predict(im,limit_side_len=960,limit_type='max');print('SECONDS',time.time()-t,flush=True);print('BOXES',len(r[0].json['res']['dt_polys']),flush=True)
