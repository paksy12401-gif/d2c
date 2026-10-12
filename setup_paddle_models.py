"""Download public PP-OCRv5 models once; no drawing is read or transmitted."""
import os
from pathlib import Path


root = Path(os.environ.get("CAD_PADDLE_MODEL_ROOT",
                           Path.home() / "Documents" / "CAD_OCR_MODELS"))
if any(ord(char) > 127 for char in str(root)):
    raise ValueError("Windows Paddle 모델 경로는 영문 경로를 사용하세요.")
root.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "PADDLE_PDX_CACHE_HOME": str(root),
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "PADDLE_PDX_MODEL_SOURCE": "bos",
})

from paddleocr import PaddleOCR

PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
    enable_mkldnn=False,
)
print(f"PaddleOCR models ready: {root}")
