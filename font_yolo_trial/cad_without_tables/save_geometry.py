from pathlib import Path
import subprocess
out=Path('font_yolo_trial/cad_without_tables').resolve()
s=out/'geometry_dwg.scr';s.write_text('(setvar "FILEDIA" 0)\n_.SAVEAS\n_2018\nnon_table_geometry.dwg\n_.QUIT\n_N\n',encoding='ascii')
with (out/'geometry_dwg.log').open('wb') as f:
 r=subprocess.run([r'C:/Program Files/Autodesk/AutoCAD 2027/accoreconsole.exe','/i',str(out/'non_table_geometry.dxf'),'/s',str(s),'/l','en-US'],cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=300)
print(r.returncode,(out/'non_table_geometry.dwg').exists())
