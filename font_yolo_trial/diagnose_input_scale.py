"""Compare fixed inference settings on the prior-seen real sample; no training."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from prepare import read_image, write_image
from train import read_rows

ROOT = Path(__file__).resolve().parent


def padded(image, scale):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    assert h <= 384 and w <= 384
    canvas = np.full((384, 384, 3), 255, dtype=np.uint8)
    x, y = (384 - w) // 2, (384 - h) // 2
    canvas[y:y+h, x:x+w] = resized
    assert np.array_equal(canvas[y:y+h, x:x+w], resized)
    return canvas


def main():
    torch.set_num_threads(4)
    source_path = ROOT / 'alphanumeric/user_comparison/input_original.png'
    source = read_image(source_path)
    weights = ROOT / 'alphanumeric/training/weights/best.pt'
    output = ROOT / 'alphanumeric/input_scale_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    inputs = [(f'original_{size}_rect_{rect}', source, size, rect)
              for size in (384, 640) for rect in (True, False)]
    inputs += [(f'white_canvas_scale_{scale}', padded(source, scale), 384, False)
               for scale in (.25, .4, .55, .65)]
    report = {'scope': 'Prior-seen sample diagnostic; not an independent accuracy test.',
              'source': str(source_path), 'weights': str(weights),
              'confidence_threshold': .25, 'iou': .5,
              'word_correction': False, 'predictions': []}
    # The installed Ultralytics version selects its inference head with end2end.
    for end2end in (True, False):
        model = YOLO(str(weights))
        for name, pixels, size, rect in inputs:
            result = model.predict(pixels, imgsz=size, rect=rect, conf=.25, iou=.5,
                                   end2end=end2end, device=0, verbose=False)[0]
            item = {'input': name, 'end2end': end2end, 'imgsz': size, 'rect': rect,
                    'raw_lines': read_rows(result),
                    'detections': [{'label': result.names[int(box[5])],
                                    'confidence': box[4], 'input_xyxy': box[:4]}
                                   for box in result.boxes.data.cpu().tolist()]}
            report['predictions'].append(item)
            write_image(output / f'{name}_end2end_{end2end}.png',
                        result.plot(line_width=1, font_size=10))
            print(json.dumps({k: v for k, v in item.items() if k != 'detections'}), flush=True)
    assert len(report['predictions']) == 16
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
