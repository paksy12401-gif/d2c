from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import ezdxf
from ezdxf.addons.drawing import RenderContext,Frontend,layout,config
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.addons.drawing import pymupdf
from ezdxf.math import BoundingBox2d
out=Path('font_yolo_trial/cad_without_tables')
doc=ezdxf.readfile(out/(sys.argv[1] if len(sys.argv)>1 else 'non_table_geometry.dxf'))
backend=pymupdf.PyMuPdfBackend()
configuration=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,color_policy=config.ColorPolicy.BLACK,lineweight_scaling=.6)
Frontend(RenderContext(doc),backend,config=configuration).draw_layout(doc.modelspace(),finalize=True)
page=layout.Page(450,90,layout.Units.mm,margins=layout.Margins.all(3))
(out/(sys.argv[2] if len(sys.argv)>2 else 'geometry_preview.png')).write_bytes(backend.get_pixmap_bytes(page,fmt='png',dpi=160,render_box=BoundingBox2d([(0,0),(173.622083875,33.11027695833333)])))
print('RENDERED')
