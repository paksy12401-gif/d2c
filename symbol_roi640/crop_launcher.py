"""Start the local crop screen from a VS Code notebook without a console window."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def start_crop_ui(pdf_path, output_dir, port=8510):
    pdf_path=Path(pdf_path).resolve()
    output_dir=Path(output_dir).resolve()
    if not pdf_path.is_file(): raise FileNotFoundError(pdf_path)
    url=f"http://127.0.0.1:{int(port)}"
    try:
        with urlopen(url+"/health",timeout=2) as response:
            health=json.load(response)
        if (health.get("app")=="manual-pdf-roi640" and Path(health["pdf"])==pdf_path
                and Path(health["output"])==output_dir):
            return url+"/"
        raise ValueError("Port is used by another file/application; choose another port")
    except (URLError,TimeoutError,ConnectionError):
        pass
    output_dir.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).parent/"crop_server.py"
    log_path=Path(__file__).parent/"crop_server.log"
    with log_path.open("ab") as log:
        process=subprocess.Popen(
            [sys.executable,"-u",str(script),"--pdf",str(pdf_path),"--output",str(output_dir),"--port",str(port)],
            stdout=log,stderr=log,cwd=script.parent,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    for _ in range(80):
        if process.poll() is not None:
            raise RuntimeError(f"Crop server failed; check {log_path}")
        try:
            with urlopen(url+"/health",timeout=.5) as response:
                if json.load(response).get("app")=="manual-pdf-roi640":
                    return url+"/"
        except (URLError,TimeoutError,ConnectionError):
            pass
        time.sleep(.25)
    raise RuntimeError(f"Crop server did not start; check {log_path}")
