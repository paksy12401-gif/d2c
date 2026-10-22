"""Save and reopen the experiment in AutoCAD; package only reviewable outputs."""
import hashlib
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import ezdxf

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'font_yolo_trial/cad_without_tables'
OUT=ROOT/'output/cad_without_tables_yolo_applied'
CORE=r'C:/Program Files/Autodesk/AutoCAD 2027/accoreconsole.exe'


def run_cad(source, script_name, lines):
    script=OUT/script_name
    script.write_text('\n'.join(['(setvar "FILEDIA" 0)','(setvar "CMDECHO" 0)',*lines,'']),encoding='ascii')
    log=script.with_suffix('.log')
    with log.open('wb') as stream:
        result=subprocess.run([CORE,'/i',str(source),'/s',str(script),'/l','en-US'],cwd=OUT,
                              stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError(f'AutoCAD failure: {log}')
    return log


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}
    for stem in ('non_table_geometry','non_table_yolo_ocr'):
        dxf=OUT/f'{stem}.dxf';shutil.copy2(WORK/dxf.name,dxf)
        dwg=OUT/f'{stem}.dwg'
        save=['_.SAVEAS','_2018',dwg.name]
        if dwg.exists():save.append('_Y')
        run_cad(dxf,stem+'_save.scr',save+['_.QUIT','_N'])
        assert dwg.exists() and dwg.stat().st_size>1000
        log=run_cad(dwg,stem+'_reopen.scr',[
            '(setq ss (ssget "_X" (list (cons 410 "Model"))))',
            '(princ (strcat "\\nMODEL_ENTITIES=" (itoa (sslength ss))))',
            '(setq tx (ssget "_X" (list (cons 410 "Model") (cons 0 "TEXT,MTEXT"))))',
            '(princ (strcat "\\nEDITABLE_TEXT=" (itoa (if tx (sslength tx) 0))))',
            '(setq ys (ssget "_X" (list (cons 410 "Model") (cons 8 "YOLO_TEXT") (cons 0 "TEXT"))))',
            '(princ (strcat "\\nYOLO_TEXT=" (itoa (if ys (sslength ys) 0))))',
            '_.AUDIT','_N','_.QUIT','_N'])
        text=log.read_bytes().decode('utf-16-le',errors='replace')
        counts=re.findall(r'MODEL_ENTITIES=(\d+)',text)
        if not counts:raise AssertionError(f'No reopen verification: {log}')
        parsed=ezdxf.readfile(dxf)
        assert int(counts[-1])==len(parsed.modelspace())
        assert not parsed.audit().has_errors
        yolo_count=len(parsed.modelspace().query('TEXT[layer=="YOLO_TEXT"]'))
        assert int(re.findall(r'YOLO_TEXT=(\d+)',text)[-1])==yolo_count
        yolo_visible=yolo_count>0 and not parsed.layers.get('YOLO_TEXT').is_off() and not parsed.layers.get('YOLO_TEXT').is_frozen()
        if stem=='non_table_yolo_ocr':assert yolo_visible
        checks[stem]={'dwg_reopened':True,'model_entities':int(counts[-1]),
                      'editable_text':int(re.findall(r'EDITABLE_TEXT=(\d+)',text)[-1]),
                      'yolo_text':yolo_count,'yolo_layer_visible':yolo_visible,
                      'dxf_audit_errors':0,'dwg_sha256':hashlib.sha256(dwg.read_bytes()).hexdigest()}
    for filename in ('recognized_text.csv','conversion_report.json','ocr_results.json','geometry_preview.png'):
        shutil.copy2(WORK/filename,OUT/filename)
    report=json.loads((OUT/'ocr_results.json').read_text(encoding='utf-8'))
    report['llm_used']=False;report['training']=False
    report['model_sha256']={key:hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in report['models'].items()}
    (OUT/'ocr_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'cad_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
