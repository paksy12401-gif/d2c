"""Local manual PDF crop UI. A file is saved only when the user presses Save."""
import argparse
import io
import json
import math
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image
import pypdfium2 as pdfium
from roi_tools import pdf_info, render_roi, write_json


def render_view(pdf_path, page, x, y, scale, width, height):
    """Render the visible window only; do not rasterize this very wide whole PDF."""
    if not (.02 <= scale <= 8 and 1 <= width <= 2200 and 1 <= height <= 1400):
        raise ValueError("View bounds exceed limits")
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        p = doc[page-1]
        pw, ph = p.get_size()
        x, y = min(max(0, x), pw-.1), min(max(0, y), ph-.1)
        crop = (x, max(0,ph-y-height/scale), max(0,pw-x-width/scale), y)
        bitmap = p.render(scale=scale,crop=crop)
        image = bitmap.to_pil().convert("RGB").copy()
        bitmap.close()
        canvas = Image.new("RGB",(width,height),"white")
        canvas.paste(image.crop((0,0,min(width,image.width),min(height,image.height))),(0,0))
        return canvas


def make_handler(pdf_path, output, token):
    lock = threading.Lock()  # PDFium is not thread safe.
    pages = pdf_info(pdf_path)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,status,data,kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(data)

        def respond(self,value,status=200):
            self.send(status,json.dumps(value,ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            parsed=urlparse(self.path)
            q=parse_qs(parsed.query)
            try:
                if parsed.path=="/":
                    html=(Path(__file__).parent/"crop_ui.html").read_text(encoding="utf-8")
                    self.send(200,html.replace("__SAVE_TOKEN__",token).encode("utf-8"),"text/html; charset=utf-8")
                elif parsed.path=="/health":
                    self.respond({"app":"manual-pdf-roi640","pdf":str(pdf_path),"output":str(output)})
                elif parsed.path=="/info":
                    self.respond({"pdf_name":pdf_path.name,"output":str(output),"pages":pages,
                                  "saved":len(list(output.glob("roi_*.png")))})
                elif parsed.path in ("/view","/crop"):
                    page=int(q.get("page",[1])[0])
                    if not 1<=page<=len(pages):
                        raise ValueError("Invalid PDF page")
                    with lock:
                        if parsed.path=="/view":
                            image=render_view(pdf_path,page,float(q["x"][0]),float(q["y"][0]),
                                              float(q["scale"][0]),int(q["w"][0]),int(q["h"][0]))
                        else:
                            image,_=render_roi(pdf_path,page,[int(q["x"][0]),int(q["y"][0])],int(q["dpi"][0]))
                    data=io.BytesIO();image.save(data,"PNG")
                    self.send(200,data.getvalue(),"image/png")
                else:
                    self.respond({"error":"Not found"},404)
            except (ValueError,KeyError,IndexError,OverflowError) as exc:
                self.respond({"error":str(exc)},400)

        def do_POST(self):
            if self.path!="/save":
                self.respond({"error":"Not found"},404);return
            try:
                size=int(self.headers.get("Content-Length",0))
                if not 0<size<=4096:
                    raise ValueError("Invalid request size")
                data=json.loads(self.rfile.read(size))
                if data.get("token")!=token:
                    self.respond({"error":"Forbidden"},403);return
                page,dpi=int(data["page"]),int(data["dpi"])
                x,y=int(data["x"]),int(data["y"])
                if not 1<=page<=len(pages): raise ValueError("Invalid page")
                with lock:
                    image,meta=render_roi(pdf_path,page,[x,y],dpi)
                    path=output/f"roi_p{page:02d}_d{dpi}_x{x}_y{y}.png"
                    if path.exists():
                        self.respond({"error":"같은 위치의 사진이 이미 저장되어 있습니다.","path":str(path)},409)
                        return
                    image.save(path)
                    write_json(path.with_suffix(".json"),meta)
                self.respond({"path":str(path),"size":[640,640],"saved":len(list(output.glob("roi_*.png")))})
            except (ValueError,KeyError,IndexError,json.JSONDecodeError,OSError) as exc:
                self.respond({"error":str(exc)},400)
    return Handler


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pdf",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--port",type=int,default=8510)
    args=parser.parse_args()
    source=Path(args.pdf).resolve();output=Path(args.output).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(source,output,secrets.token_hex(24)))
    print(f"Manual crop UI: http://127.0.0.1:{args.port}/",flush=True)
    server.serve_forever()
