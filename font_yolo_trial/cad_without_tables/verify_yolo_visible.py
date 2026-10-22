import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'font_yolo_trial'))
from export_non_table_outputs import run_cad,OUT
log=run_cad(OUT/'non_table_yolo_ocr.dwg','verify_yolo_visible.scr',[
 '(setq yl (tblsearch "LAYER" "YOLO_TEXT"))',
 '(princ (strcat "\\nYOLO_VISIBLE=" (if (and yl (> (cdr (assoc 62 yl)) 0) (= (logand (cdr (assoc 70 yl)) 1) 0)) "1" "0")))',
 '_.QUIT','_N'])
text=log.read_bytes().decode('utf-16-le',errors='replace')
assert re.findall(r'YOLO_VISIBLE=(\d)',text)[-1]=='1'
p=OUT/'cad_verification.json';report=json.loads(p.read_text(encoding='utf-8'))
report['non_table_yolo_ocr']['dwg_yolo_layer_visible']=True
p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('DWG_YOLO_LAYER_VISIBLE',True)
