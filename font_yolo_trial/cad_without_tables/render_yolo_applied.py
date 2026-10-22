from pathlib import Path

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend, config, layout, pymupdf
from ezdxf.math import BoundingBox2d

work=Path(__file__).resolve().parent
doc=ezdxf.readfile(work/'non_table_yolo_ocr.dxf')
settings=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                              color_policy=config.ColorPolicy.COLOR,lineweight_scaling=.6)
for name,box,page in [
    ('yolo_applied_overview.png',[(0,0),(173.622083875,33.11027695833333)],layout.Page(450,90,margins=layout.Margins.all(3))),
    ('yolo_applied_detail.png',[(3080/72,(2383.93994140625-1600)/72),(3730/72,(2383.93994140625-1290)/72)],
     layout.Page(220,110,margins=layout.Margins.all(3)))
]:
    backend=pymupdf.PyMuPdfBackend()
    Frontend(RenderContext(doc),backend,config=settings).draw_layout(doc.modelspace(),finalize=True)
    (work/name).write_bytes(backend.get_pixmap_bytes(page,fmt='png',dpi=160,render_box=BoundingBox2d(box)))
    print(name,flush=True)
