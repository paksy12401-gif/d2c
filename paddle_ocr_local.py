"""Offline PP-OCRv5 adapter for Korean/English engineering drawings."""
import os
from pathlib import Path

from models_local import load_ocr, recognize_texts


def model_root():
    configured = os.environ.get("CAD_PADDLE_MODEL_ROOT")
    return Path(configured) if configured else Path.home() / "Documents" / "CAD_OCR_MODELS"


def model_dirs():
    official = model_root() / "official_models"
    return (official / "PP-OCRv5_server_det",
            official / "korean_PP-OCRv5_mobile_rec")


def models_ready():
    return all((folder / "inference.json").is_file() and
               (folder / "inference.pdiparams").is_file()
               for folder in model_dirs())


def load_reader():
    det_dir, rec_dir = model_dirs()
    reader = load_ocr(det_dir, rec_dir)
    reader._cad_model_root = str(model_root())
    return reader


def read_text(gray, reader=None, *, min_confidence=0.5):
    return recognize_texts(gray, reader or load_reader(),
                           min_confidence=min_confidence)
