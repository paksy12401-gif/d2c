"""Train/evaluate the supplied CAD glyph experiment; no CAD files are edited.

Run prepare.py first, then this script with the repository's .venv-ml Python.
All splits reuse the same glyph templates: scores measure synthetic feasibility.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from prepare import write_image

ROOT = Path(__file__).resolve().parent
LETTERS = string.ascii_uppercase


def copy_classifier_rows(source, target, names):
    """Transfer existing character logits with case-sensitive label matching."""
    lookup = {name: index for index, name in source.names.items()}
    pairs = [(index, lookup[name]) for index, name in names.items() if name in lookup]
    copied = 0
    with torch.no_grad():
        for attr in ("cv3", "one2one_cv3"):
            old_heads = getattr(source.model[-1], attr, [])
            new_heads = getattr(target.model[-1], attr, [])
            assert len(old_heads) == len(new_heads)
            for old, new in zip(old_heads, new_heads):
                old, new = old[-1], new[-1]
                assert old.weight.shape[1:] == new.weight.shape[1:]
                unmatched = [i for i in range(len(names)) if i not in {p[0] for p in pairs}]
                original_new = new.weight[unmatched].clone(), new.bias[unmatched].clone()
                for dst, src in pairs:
                    new.weight[dst].copy_(old.weight[src])
                    new.bias[dst].copy_(old.bias[src])
                    assert torch.equal(new.weight[dst], old.weight[src])
                assert torch.equal(new.weight[unmatched], original_new[0])
                assert torch.equal(new.bias[unmatched], original_new[1])
                copied += 1
    return {"matched_classes": len(pairs), "classifier_convolutions": copied,
            "case_sensitive": True, "new_classes": [name for name in names.values() if name not in lookup]}


class CharacterTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if weights is not None:
            report = copy_classifier_rows(weights, model, self.data["names"])
            (self.save_dir / "class_transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Case-sensitive classifier transfer:", report, flush=True)
        return model


def overlap(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0, right - left)))
    area_a = float(np.prod(np.maximum(0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0, b[2:] - b[:2])))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def read_labels(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = map(float, line.split())
        boxes.append((int(cls), np.array([(x-w/2)*width, (y-h/2)*height,
                                          (x+w/2)*width, (y+h/2)*height])))
    return boxes


def read_rows(result):
    """Group predicted boxes geometrically; no expected words are used."""
    entries = [(box[:4], result.names[int(box[5])])
               for box in result.boxes.cpu().numpy().data]
    rows = []
    # Build rows from full letters before assigning small dots and high apostrophes.
    for box, letter in sorted(entries, key=lambda item: (not item[1].isalnum(), (item[0][1]+item[0][3])/2)):
        center = float((box[1]+box[3])/2)
        height = float(box[3]-box[1])
        if letter.isalnum():
            candidates = [row for row in rows if abs(row[0]-center) <= height * .55]
        else:
            candidates = []
            for row in rows:
                top = min(b[1] for b, _ in row[1])
                bottom = max(b[3] for b, _ in row[1])
                margin = (bottom - top) * .2
                if box[1] <= bottom + margin and box[3] >= top - margin:
                    candidates.append(row)
        closest = min(candidates, key=lambda row: abs(row[0]-center), default=None)
        if closest is None:
            rows.append([center, [(box, letter)]])
        else:
            closest[1].append((box, letter))
    output = []
    for _, items in sorted(rows):
        items.sort(key=lambda item: item[0][0])
        median_height = np.median([box[3]-box[1] for box, _ in items])
        text = ""
        previous = None
        for box, letter in items:
            if previous is not None and box[0]-previous[2] > median_height*.55:
                text += " "
            text += letter
            previous = box
        output.append(text)
    return output


def evaluate(model, confidence=.25, head="auto"):
    assert overlap(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1
    assert overlap(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0
    paths = sorted((ROOT / "images/test").glob("*.png"))
    if not paths:
        raise FileNotFoundError("Run prepare.py before evaluating")
    stats = np.zeros((len(LETTERS), 3), dtype=int)  # TP, FP, FN
    folded_stats = np.zeros(3, dtype=int)
    image_stats = []
    negative_images = negative_fp = 0
    previews = ROOT / "test_predictions"
    previews.mkdir(exist_ok=True)
    head_options = {} if head == "auto" else {"end2end": head == "end2end"}
    predictions = model.predict([str(p) for p in paths], imgsz=384, conf=confidence,
                                iou=.5, device=0, batch=8, verbose=False, stream=True,
                                **head_options)
    review_images = []
    # Exhaust the stream so the predictor releases its lock before demo inference.
    for index, result in enumerate(predictions):
        path = paths[index]
        height, width = result.orig_shape
        truth = read_labels(ROOT / "labels/test" / (path.stem+".txt"), width, height)
        available = set(range(len(truth)))
        tp = fp = 0
        detections = sorted(result.boxes.data.cpu().numpy(), key=lambda x: -x[4])
        for detection in detections:
            cls = int(detection[5])
            candidates = [(overlap(detection[:4], truth[j][1]), j)
                          for j in available if truth[j][0] == cls]
            best = max(candidates, default=(0, -1))
            if best[0] >= .5:
                available.remove(best[1])
                stats[cls, 0] += 1
                tp += 1
            else:
                stats[cls, 1] += 1
                fp += 1
        for j in available:
            stats[truth[j][0], 2] += 1
        folded_available = set(range(len(truth)))
        for detection in detections:
            cls = int(detection[5])
            candidates = [(overlap(detection[:4], truth[j][1]), j)
                          for j in folded_available
                          if LETTERS[truth[j][0]].upper() == LETTERS[cls].upper()]
            best = max(candidates, default=(0, -1))
            if best[0] >= .5:
                folded_available.remove(best[1])
                folded_stats[0] += 1
            else:
                folded_stats[1] += 1
        folded_stats[2] += len(folded_available)
        if not truth:
            negative_images += 1
            negative_fp += fp
        image_stats.append({"image": path.name, "tp": tp, "fp": fp,
                            "fn": len(available), "truth_count": len(truth)})
        review_images.append((fp+len(available), index, result.plot(line_width=1,
                              font_size=9, labels=True, conf=False)))
    assert len(image_stats) == len(paths)
    per_class = {}
    for letter, (tp, fp, fn) in zip(LETTERS, stats.tolist()):
        per_class[letter] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1)}
    tp, fp, fn = stats.sum(axis=0).tolist()
    # Include the first examples and worst failures rather than only successes.
    chosen = review_images[:3] + sorted(review_images, key=lambda x: -x[0])[:3]
    panels = []
    for errors, index, annotated in chosen:
        panel = cv2.copyMakeBorder(annotated, 24, 0, 0, 0, cv2.BORDER_CONSTANT,
                                   value=(250, 250, 250))
        cv2.putText(panel, f"Test {index}: FP+FN={errors}", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1)
        panels.append(panel)
        write_image(previews / paths[index].name, annotated)
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    write_image(ROOT / "test_results.png", montage)
    demo = model.predict(str(ROOT / "demo_spline.png"), imgsz=640, conf=confidence,
                         iou=.5, device=0, verbose=False, **head_options)[0]
    write_image(ROOT / "demo_detected.png", demo.plot(line_width=1, conf=False))
    folded_tp, folded_fp, folded_fn = folded_stats.tolist()
    groups = {}
    for group, indices in {
        "letters": [i for i, char in enumerate(LETTERS) if char.isalpha()],
        "digits": [i for i, char in enumerate(LETTERS) if char.isdigit()],
        "punctuation": [i for i, char in enumerate(LETTERS) if not char.isalnum()],
    }.items():
        if indices:
            group_tp, group_fp, group_fn = stats[indices].sum(axis=0).tolist()
            groups[group] = {"tp": group_tp, "fp": group_fp, "fn": group_fn,
                             "precision": group_tp/max(group_tp+group_fp, 1),
                             "recall": group_tp/max(group_tp+group_fn, 1)}
    report = {
        "scope": "Synthetic feasibility only. All splits reuse one source glyph per class.",
        "independent_real_drawing_accuracy": None,
        "model": str(ROOT / "training/weights/best.pt"),
        "confidence_threshold": confidence, "matching_iou": .5, "inference_head": head,
        "test_images": len(paths), "test_glyphs": tp+fn,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1),
        "negative_images": negative_images, "negative_false_positives": negative_fp,
        "case_insensitive": {"tp": folded_tp, "fp": folded_fp, "fn": folded_fn,
                             "precision": folded_tp/max(folded_tp+folded_fp, 1),
                             "recall": folded_tp/max(folded_tp+folded_fn, 1)},
        "demo_predicted_rows": read_rows(demo),
        "groups": groups,
        "per_class": per_class, "per_image": image_stats,
    }
    (ROOT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_class", "per_image")}, indent=2), flush=True)


def main():
    global ROOT, LETTERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "mixed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--weights", type=Path,
                        default=ROOT.parent / "models/yolo26s.pt")
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--head", choices=("auto", "end2end", "nms"), default="auto")
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("Confidence must be in (0, 1]")
    ROOT = args.data_dir.resolve()
    LETTERS = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["classes"]
    if not torch.cuda.is_available():
        raise RuntimeError("This trial expects the available local CUDA GPU")
    torch.set_num_threads(4)
    weights = ROOT / "training/weights/best.pt"
    if not args.evaluate_only:
        if (ROOT / "training").exists():
            raise FileExistsError("Training already exists; use --evaluate-only or a separate trial")
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        model = YOLO(str(args.weights))
        model.train(trainer=CharacterTrainer, data=str(ROOT / "data.yaml"), epochs=args.epochs,
                    imgsz=384, batch=16, device=0, workers=0,
                    project=str(ROOT), name="training", seed=20261016,
                    # Ultralytics name remapping lowercases names, conflating A/a.
                    cls_remap=False,
                    optimizer="AdamW", lr0=args.learning_rate, lrf=.1, warmup_epochs=2,
                    patience=15, amp=False, cache=False, plots=True,
                    mosaic=0, mixup=0, cutmix=0, copy_paste=0,
                    degrees=0, translate=0, scale=0, shear=0, perspective=0,
                    fliplr=0, flipud=0, hsv_h=0, hsv_s=0, hsv_v=0)
    evaluate(YOLO(str(weights)), args.confidence, args.head)


if __name__ == "__main__":
    main()
