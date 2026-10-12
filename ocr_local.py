"""Downloaded EasyOCR Korean/English models, with network blocked at inference."""
from pathlib import Path
import numpy as np
from models_local import _offline

MODEL_DIR = Path(__file__).parent / "models/easyocr"


def load_reader():
    with _offline():
        import easyocr
        return easyocr.Reader(['ko','en'], gpu=False,
            model_storage_directory=str(MODEL_DIR),
            user_network_directory=str(MODEL_DIR / 'user_network'),
            download_enabled=False, verbose=False)


def read_text(gray, reader=None):
    if reader is None:
        reader = load_reader()
    import cv2
    # Tiny circuit labels benefit from interpolation; coordinates are mapped back.
    scale = min(3.0, max(1.0, 1800/max(gray.shape)))
    large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    with _offline():
        results = reader.readtext(large, detail=1, paragraph=False, workers=0,
                                 min_size=4, text_threshold=.55, low_text=.3,
                                 link_threshold=.3, add_margin=.02)
    texts = []
    for polygon, value, confidence in results:
        value = ' '.join(value.split())
        if not value or confidence < .15:
            continue
        pts=np.asarray(polygon, dtype=float)/scale
        x,y=np.maximum(pts.min(axis=0),0)
        x1,y1=np.minimum(pts.max(axis=0),[gray.shape[1]-1,gray.shape[0]-1])
        if x1<=x or y1<=y:
            continue
        texts.append({'id':f'OCR_{len(texts)+1:04d}','text':value,'confidence':float(confidence),
                      'bbox':[float(x),float(y),float(x1-x),float(y1-y)],
                      'polygon':pts.tolist(),'links':[], 'status':'needs_review'})
    return texts
