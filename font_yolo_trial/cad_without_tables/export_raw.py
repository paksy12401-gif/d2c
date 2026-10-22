from pathlib import Path
import subprocess
out=Path('font_yolo_trial/cad_without_tables').resolve()
s=out/'export_raw.scr'
s.write_text('(setvar "FILEDIA" 0)\n_.SAVEAS\n_DXF\n16\nraw_import.dxf\n_.QUIT\n_N\n',encoding='ascii')
with (out/'export_raw.log').open('wb') as f:
 r=subprocess.run([r'C:/Program Files/Autodesk/AutoCAD 2027/accoreconsole.exe','/i',str(out/'non_table_native.dwg'),'/s',str(s),'/l','en-US'],cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=300)
print(r.returncode,(out/'raw_import.dxf').exists())
