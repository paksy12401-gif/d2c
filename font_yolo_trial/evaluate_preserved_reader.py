"""Evaluate fixed dual-model inference on existing glyph tests and user samples."""
import hashlib
import json

import numpy as np
import torch

from prepare import read_image, write_image
from preserved_reader import ROOT, PreservedReader, merge_predictions, remove_drawing_rules, symbol_nms
from train import overlap, read_labels, read_rows


def match(boxes, truth):
    available = set(range(len(truth)))
    matched, fp = set(), 0
    for box in sorted(boxes, key=lambda b: -b[4]):
        candidate = max(((overlap(np.asarray(box[:4]), truth[j][1]), j)
                         for j in available if int(box[5]) == truth[j][0]), default=(0, -1))
        if candidate[0] >= .5:
            available.remove(candidate[1])
            matched.add(candidate[1])
        else:
            fp += 1
    return matched, fp, len(available)


def evaluate_set(reader, directory):
    paths = sorted((directory / "images/test").glob("*.png"))
    baseline = list(reader.letters.predict([str(p) for p in paths], imgsz=384,
        conf=.1, iou=.5, end2end=False, agnostic_nms=False, device=0, batch=8, stream=True, verbose=False))
    cleaned = [remove_drawing_rules(r.orig_img, r.boxes.cpu().numpy().data.tolist())[0] for r in baseline]
    symbols = reader.symbols.predict(cleaned, imgsz=384, conf=.1, iou=.5,
        end2end=False, agnostic_nms=False, device=0, batch=8, stream=True, verbose=False)
    stats = {"baseline": np.zeros(3, dtype=int), "combined": np.zeros(3, dtype=int)}
    lost_letters, new_symbol_fp, images = [], 0, 0
    for i, prediction in enumerate(symbols):
        images += 1
        base = baseline[i].boxes.cpu().numpy().data.tolist()
        marks = symbol_nms([b for b in prediction.boxes.cpu().numpy().data.tolist() if int(b[5]) >= 62])
        merged, replaced = merge_predictions(base, marks, cleaned[i])
        truth = read_labels(directory / "labels/test" / (paths[i].stem + ".txt"), 384, 384)
        before, fp_before, fn_before = match(base, truth)
        after, fp_after, fn_after = match(merged, truth)
        stats["baseline"] += [len(before), fp_before, fn_before]
        stats["combined"] += [len(after), fp_after, fn_after]
        lost = [j for j in before-after if truth[j][0] < 62]
        if lost:
            lost_letters.append({"image": paths[i].name, "truth_indices": lost, "replacements": replaced})
        if directory.name == "alphanumeric":
            new_symbol_fp += len(marks)
    assert images == len(paths) == 130
    return {"scope": "Same-template synthetic diagnostic; not independent CAD accuracy.",
            "images": images, "counts_tp_fp_fn": {key: values.tolist() for key, values in stats.items()},
            "previously_correct_letters_lost": lost_letters, "symbol_fp_on_alphanumeric_only_set": new_symbol_fp}


def main():
    torch.set_num_threads(4)
    reader = PreservedReader()
    output = ROOT / "preserved_letters"
    output.mkdir(exist_ok=True)
    report = {"confidence": .1, "iou": .5, "end2end": False, "word_correction": False,
              "letter_model_unchanged_sha256": hashlib.sha256(reader.paths["letters"].read_bytes()).hexdigest(),
              "symbol_model_sha256": hashlib.sha256(reader.paths["symbols"].read_bytes()).hexdigest(),
              "synthetic": {}, "samples": []}
    for dataset in ("alphanumeric", "punctuation_fixed"):
        report["synthetic"][dataset] = evaluate_set(reader, ROOT / dataset)
        print(json.dumps({"dataset": dataset, **report["synthetic"][dataset]}), flush=True)
    samples = {
        "cad": (ROOT / "alphanumeric/sample_23476557/source.png", "tiles", 384),
        "punctuation": (ROOT / "punctuation_fixed/source_punctuation.png", "full", 384),
        "digits": (ROOT / "alphanumeric/source_digits.png", "full", 640),
        "compact": (ROOT / "alphanumeric/user_comparison/input_original.png", "full", 384),
    }
    for name, (path, mode, size) in samples.items():
        image = read_image(path)
        result, baseline, details = reader.predict(image, path, mode, size)
        write_image(output / f"{name}_detected.png", result.plot(line_width=1, font_size=10))
        write_image(output / f"{name}_symbols_input.png", details["cleaned"])
        boxes = result.boxes.cpu().numpy().data.tolist()
        item = {"sample": name, "source": str(path), "mode": mode, "imgsz": size,
                "baseline_lines": read_rows(baseline), "raw_lines": read_rows(result),
                "rules": details["rules"], "replaced": details["replaced"],
                "detections": [{"label": reader.names[int(b[5])], "score": b[4], "xyxy": b[:4]} for b in boxes]}
        report["samples"].append(item)
        print(json.dumps({k: item[k] for k in ("sample", "baseline_lines", "raw_lines")}), flush=True)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
