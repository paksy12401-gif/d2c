"""Explicit local YOLO11 training. No dataset is selected implicitly."""
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".model_cache" / "ultralytics")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"


def main(dataset, epochs=20, weights=None):
    import subprocess, sys
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "check_runtime.py")], check=True)
    import faulthandler
    faulthandler.enable()
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(2)
    dataset = Path(dataset).resolve()
    from models_local import train_yolo, load_yolo, _offline
    import torch

    # Ultralytics checks for this font even with plots=False. Reuse Windows' local font.
    cache = Path(os.environ["YOLO_CONFIG_DIR"])
    cache.mkdir(parents=True, exist_ok=True)
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    if font.is_file():
        shutil.copyfile(font, cache / "Arial.ttf")
    output = ROOT / "runs" / "control_circuits"
    device = "0" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    result = train_yolo(dataset, weights, output, task="detect", epochs=epochs,
                        imgsz=640, device=device, batch=8)
    destination = ROOT / "models" / "control_circuit_best.pt"
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(result["best_weights"], destination)
    detector = load_yolo(destination, task="detect")
    with _offline():
        test = detector.val(data=str(dataset), split="test", imgsz=640, batch=8,
                            device=device, workers=0, plots=False,
                            project=str(output), name="test", verbose=False)
    summary = {"dataset": str(dataset), "task": "detect",
               "model": "YOLO11n", "initialization": str(weights) if weights else "random",
               "epochs": epochs, "imgsz": 640, "batch": 8, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
               "elapsed_seconds": round(time.monotonic()-started, 1),
               "best_weights": str(destination), "validation_metrics": result["metrics"],
               "test_metrics": {str(k):float(v) for k,v in test.results_dict.items()},
               "provenance": json.loads((dataset.parent / "provenance.json").read_text(encoding="utf-8")) if (dataset.parent / "provenance.json").exists() else None,
               "scope": "Local symbol detection trial only; not harness OCR, connectivity, CAD or end-to-end accuracy."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--data", required=True, help="Explicit local dataset.yaml path")
    parser.add_argument("--weights", default=None, help="Trusted local checkpoint; omitted = scratch")
    args = parser.parse_args()
    main(args.data, args.epochs, args.weights)
