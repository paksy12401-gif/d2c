import fitz,json,hashlib,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from pdf_to_dwg_workflow import _import_pdf
out=Path('font_yolo_trial/cad_without_tables').resolve()
p=Path(r'C:/Users/Win11Pro/Desktop/1. DWG 객체 변환 도면 생성 샘플/DWG 객체 변환 샘플.pdf')
doc=fitz.open(p);page=doc[0]
for name,box in [('bom',(4950,140,7200,1280)),('circuit',(7900,30,11250,2250)),('notes',(5950,1260,7220,1650))]:
 page.get_pixmap(matrix=fitz.Matrix(.7,.7),clip=fitz.Rect(box),alpha=False).save(out/f'{name}_inspection.png')
lines=[]
for d in page.get_drawings():
 for item in d['items']:
  if item[0]=='l':
   a,b=item[1:3]
   if abs(a.y-b.y)<.1 and abs(a.x-b.x)>900:lines.append([round(a.x,2),round(a.y,2),round(b.x,2),round(b.y,2)])
(out/'long_lines.json').write_text(json.dumps(lines,indent=2))
print('SOURCE_SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
print('BOM_LINES',[l for l in lines if 4800<min(l[0],l[2])<7300],flush=True)
local=out/'source.pdf';shutil.copy2(p,local)
print('IMPORTING',flush=True)
_import_pdf(local,out/'non_table_native.dwg')
print('IMPORTED',flush=True)
