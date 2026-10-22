"""Read letters with the unchanged 62-class model and punctuation with the 67-class model."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from compare_punctuation import tiled
from prepare import read_image, write_image
from train import overlap, read_rows

ROOT = Path(__file__).resolve().parent


def foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polarity = cv2.THRESH_BINARY_INV if np.median(gray) > 127 else cv2.THRESH_BINARY
    return cv2.threshold(gray, 0, 255, polarity + cv2.THRESH_OTSU)[1]


def remove_drawing_rules(image, letters):
    """Remove long horizontal rules for symbol inference, protecting text baselines."""
    ink = foreground(image)
    heights = [b[3] - b[1] for b in letters]
    length = max(120, round(3 * np.median(heights))) if heights else 120
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(lines)
    removed = np.zeros_like(ink)
    records = []
    for index in range(1, count):
        x, y, w, h, _ = stats[index]
        center_y = y + (h - 1) / 2
        coverage = np.zeros(w, dtype=bool)
        for box in letters:
            if box[1] <= center_y < box[3]:
                left, right = max(x, int(np.floor(box[0]))), min(x+w, int(np.ceil(box[2])))
                coverage[max(0, left-x):max(0, right-x)] = True
        # A connected character baseline is mostly covered by letter boxes.
        protected = float(coverage.mean()) >= .5
        records.append({"xywh": [int(x), int(y), int(w), int(h)],
                        "letter_coverage": float(coverage.mean()), "protected": protected})
        if not protected:
            removed[components == index] = 255
    clean = image.copy()
    clean[removed > 0] = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    return clean, removed, records


def symbol_nms(boxes):
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if not any(int(box[5]) == int(other[5]) and overlap(np.asarray(box[:4]), np.asarray(other[:4])) > .3
                   for other in kept):
            kept.append(box)
    return kept


def merge_predictions(letters, symbols, cleaned):
    """Replace a letter-shaped false positive only when its surviving ink is a symbol."""
    ink = foreground(cleaned)
    h, w = ink.shape
    kept, replaced = [], []
    for letter in letters:
        x0, y0, x1, y1 = letter[:4]
        left, top = int(np.floor(x0)), int(np.floor(y0))
        right, bottom = int(np.ceil(x1)), int(np.ceil(y1))
        left, top, right, bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
        ys, xs = np.where(ink[top:bottom, left:right] > 0)
        xs, ys = xs + left, ys + top
        # Adjacent letters can share boundary strokes with a dash-shaped false positive.
        owned = np.ones(len(xs), dtype=bool)
        for other in letters:
            center_x = (other[0] + other[2]) / 2
            if other is letter or x0 <= center_x <= x1:
                continue
            owned &= ~((xs >= other[0]) & (xs < other[2]) & (ys >= other[1]) & (ys < other[3]))
        xs, ys = xs[owned], ys[owned]
        replacement = None
        for symbol in symbols:
            sx0, sy0, sx1, sy1 = symbol[:4]
            if symbol[4] < .25 or not (x0 <= (sx0+sx1)/2 <= x1):
                continue
            # Confident letters win ambiguous colon/dot/quote/slash conflicts.
            # Dashes additionally have raster evidence after a drawing rule is removed.
            if int(symbol[5]) != 63 and letter[4] >= .5:
                continue
            if len(xs) < 3:
                continue
            covered = (xs >= sx0-1) & (xs <= sx1+1) & (ys >= sy0-1) & (ys <= sy1+1)
            fraction = float(covered.mean())
            if fraction >= .7:
                replacement = {"letter_box": letter, "symbol_box": symbol, "ink_coverage": fraction}
                break
        if replacement:
            replaced.append(replacement)
        else:
            kept.append(letter)
    return np.asarray(kept + symbols, dtype=np.float32).reshape(-1, 6), replaced


class PreservedReader:
    def __init__(self):
        self.paths = {"letters": ROOT / "alphanumeric/training/weights/best.pt",
                      "symbols": ROOT / "punctuation_fixed/training/weights/best.pt"}
        self.letters = YOLO(str(self.paths["letters"]))
        self.symbols = YOLO(str(self.paths["symbols"]))
        self.names = self.symbols.names
        assert len(self.letters.names) == 62 and len(self.names) == 67
        assert all(self.letters.names[i] == self.names[i] for i in range(62))
        assert self.names[63] == "-"

    @staticmethod
    def _run(model, source, path, mode, imgsz):
        return tiled(model, source, path) if mode == "tiles" else model.predict(
            source, imgsz=imgsz, rect=True, conf=.1, iou=.5, end2end=False,
            agnostic_nms=False, device=0, verbose=False)[0]

    def predict(self, image, path="input.png", mode="tiles", imgsz=384):
        baseline = self._run(self.letters, image, path, mode, imgsz)
        letters = baseline.boxes.cpu().numpy().data.tolist()
        clean, removed, rules = remove_drawing_rules(image, letters)
        candidates = self._run(self.symbols, clean, path, mode, imgsz)
        symbols = symbol_nms([b for b in candidates.boxes.cpu().numpy().data.tolist() if int(b[5]) >= 62])
        merged, replaced = merge_predictions(letters, symbols, clean)
        result = Results(orig_img=image, path=str(path), names=self.names, boxes=merged)
        return result, baseline, {"cleaned": clean, "removed": removed, "rules": rules,
                                  "symbols": symbols, "replaced": replaced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("tiles", "full"), default="tiles")
    parser.add_argument("--imgsz", type=int, default=384)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    reader = PreservedReader()
    image = read_image(args.input)
    result, baseline, details = reader.predict(image, args.input, args.mode, args.imgsz)
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output / "detected.png", result.plot(line_width=1, font_size=10))
    write_image(args.output / "baseline.png", baseline.plot(line_width=1, font_size=10))
    write_image(args.output / "symbols_input.png", details.pop("cleaned"))
    write_image(args.output / "removed_rules.png", details.pop("removed"))
    report = {"source": str(args.input.resolve()), "mode": args.mode, "imgsz": args.imgsz,
              "conf": .1, "iou": .5, "end2end": False, "word_correction": False,
              "baseline_lines": read_rows(baseline), "raw_lines": read_rows(result),
              "model_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in reader.paths.items()},
              "detections": [{"label": reader.names[int(b[5])], "confidence": b[4], "xyxy": b[:4]}
                             for b in result.boxes.cpu().numpy().data.tolist()], **details}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("baseline_lines", "raw_lines")}), flush=True)


if __name__ == "__main__":
    main()
