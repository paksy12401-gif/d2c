"""Build a small, reproducible same-template CAD alphabet experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import string

import cv2
import numpy as np


LETTERS = string.ascii_uppercase
SPLITS = {"train": (312, 20261016), "val": (52, 20261017), "test": (78, 20261018)}
MIXED_SPLITS = {"train": (416, 20261116), "val": (78, 20261117), "test": (104, 20261118)}
ALPHANUMERIC_SPLITS = {"train": (520, 20261216), "val": (104, 20261217), "test": (130, 20261218)}
PUNCTUATION = ":-/.'"
PUNCTUATION_SPLITS = {"train": (624, 20261316), "val": (104, 20261317), "test": (130, 20261318)}
GLYPH_FILENAMES = dict(zip(PUNCTUATION, ("colon", "hyphen", "slash", "period", "apostrophe")))
SIZE = 384


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def extract_glyphs(source, output, letters=LETTERS, class_offset=0, reference_char=None):
    image = read_image(source)
    white = (np.ptp(image.astype(np.int16), axis=2) < 15) & (image.mean(2) > 150)
    columns = white.any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, columns, False].astype(np.int8)))
    spans = edges.reshape(-1, 2)
    if len(spans) != len(letters):
        raise ValueError(f"Expected {len(letters)} separate glyphs, found {len(spans)}")
    ref_index = letters.index(reference_char or ("0" if letters.isdigit() else ("H" if letters.isupper() else "h")))
    ref_left, ref_right = spans[ref_index]
    ref_ys = np.where(white[:, ref_left:ref_right])[0]
    baseline = int(ref_ys.max()) + 1
    reference_height = baseline - int(ref_ys.min())
    background = np.median(image[~white], axis=0).astype(np.float32)
    alpha = np.clip(((image.astype(np.float32) - background) / (255 - background)).mean(2), 0, 1)
    alpha[alpha < 0.025] = 0
    glyphs, boxes = [], []
    for index, (letter, (left, right)) in enumerate(zip(letters, spans)):
        left, right = max(0, int(left) - 1), min(image.shape[1], int(right) + 1)
        ys, xs = np.where(alpha[:, left:right] > 0.04)
        top, bottom = max(0, int(ys.min()) - 1), min(image.shape[0], int(ys.max()) + 2)
        glyph = alpha[top:bottom, left:right].copy()
        glyphs.append({"alpha": glyph, "char": letter, "reference_height": reference_height,
                       "top_from_baseline": top - baseline})
        rgba = np.full((*glyph.shape, 4), 255, dtype=np.uint8)
        rgba[:, :, 3] = np.rint(glyph * 255).astype(np.uint8)
        filename = GLYPH_FILENAMES.get(letter, letter)
        write_image(output / "glyphs" / f"{class_offset + index:02d}_{filename}.png", rgba)
        boxes.append({"char": letter, "class_id": class_offset + index, "bbox_xyxy": [left, top, right, bottom],
                      "reference_char": letters[ref_index], "reference_height": reference_height,
                      "source_baseline_y": baseline, "normalized_crop_width": (right - left) / reference_height,
                      "normalized_top_from_baseline": (top - baseline) / reference_height,
                      "normalized_bottom_from_baseline": (bottom - baseline) / reference_height})
    return glyphs, boxes


def resize_glyph(glyph, cap_height, thickness):
    scale = cap_height / glyph["reference_height"]
    height, width = (max(2, round(value * scale)) for value in glyph["alpha"].shape)
    alpha = cv2.resize(glyph["alpha"], (width, height), interpolation=cv2.INTER_LINEAR)
    if thickness == 2:
        alpha = cv2.dilate(alpha, np.ones((2, 2), dtype=np.uint8))
    return alpha, round(glyph["top_from_baseline"] * scale)


def paste_glyph(image, alpha, x, y, foreground, class_id, row, char):
    height, width = alpha.shape
    patch = image[y:y + height, x:x + width]
    assert patch.shape[:2] == alpha.shape
    patch[:] = np.rint(patch * (1 - alpha[:, :, None]) + foreground * alpha[:, :, None]).astype(np.uint8)
    ys, xs = np.where(alpha > 0.06)
    box = [int(x + xs.min()), int(y + ys.min()), int(x + xs.max() + 1), int(y + ys.max() + 1)]
    if char in PUNCTUATION:
        # Ultralytics augmentation discards boxes with a side <=2px, even without a warp.
        for axis, limit in enumerate((image.shape[1], image.shape[0])):
            if box[axis + 2] - box[axis] < 4:
                start = max(0, min((box[axis] + box[axis + 2] - 4) // 2, limit - 4))
                box[axis], box[axis + 2] = start, start + 4
    return {"class_id": int(class_id), "char": char, "row": int(row), "bbox_xyxy": box}


def make_scene(glyphs, rng, blank=False):
    palettes = [(27, 31, 35), (105, 78, 75), (238, 241, 244), (225, 221, 213)]
    background = np.clip(np.array(palettes[int(rng.integers(len(palettes)))]) + rng.integers(-12, 13, 3), 0, 255)
    image = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    image[:] = background
    light = background.mean() > 150
    foreground = np.full(3, int(rng.integers(12, 48) if light else rng.integers(222, 256)))
    if len(glyphs) >= 62 and rng.random() < 0.25:
        background[:] = 255 if light else 0
        image[:] = background
        foreground[:] = 0 if light else 255
    clutter = tuple(int(v) for v in np.clip(background + (-1 if light else 1) * rng.integers(25, 85), 0, 255))
    for _ in range(int(rng.integers(1, 9))):
        start, end = rng.integers(0, SIZE, (2, 2)).tolist()
        cv2.line(image, start, end, clutter, 1, cv2.LINE_AA)
    for _ in range(int(rng.integers(0, 4))):
        center = tuple(int(v) for v in rng.integers(0, SIZE, 2))
        cv2.circle(image, center, int(rng.integers(8, 65)), clutter, 1, cv2.LINE_AA)
    objects = []
    if blank:
        return image, objects
    large = len(glyphs) >= 62 and rng.random() < 0.25
    rows = int(rng.integers(1, 3) if large else rng.integers(3, 5))
    band = (SIZE - 24) / rows
    for row in range(rows):
        count = int(rng.integers(2, 5) if large else rng.integers(5, 8))
        mode = int(rng.integers(0, 3)) if len(glyphs) == 52 else 0
        low, high = (0, len(glyphs)) if mode == 0 else ((0, 26) if mode == 1 else (26, 52))
        if len(glyphs) >= 62:
            mode = int(rng.integers(20))
            low, high = (0, 62) if mode < 14 else ((0, 52) if mode < 19 else (52, 62))
        classes = rng.integers(low, high, count)
        if len(glyphs) == 67:
            # Small punctuation needs enough positive examples alongside letters.
            positions = rng.choice(count, 1 + int(rng.random() < .4), replace=False)
            classes[positions] = rng.integers(62, 67, len(positions))
        if len(glyphs) >= 62:
            gaps = rng.integers(-2, 2 if rng.random() < 0.35 else 9, count - 1)
        else:
            gaps = rng.integers(0, 11, count - 1)
        if rng.random() < 0.4:
            gaps[int(rng.integers(count - 1))] += int(rng.integers(6, 14))
        height = int(rng.integers(60, 111) if large else rng.integers(26, 59))
        thickness = int(rng.integers(1, 3))
        while True:
            letters = [resize_glyph(glyphs[index], height, thickness) for index in classes]
            width = sum(alpha.shape[1] for alpha, offset in letters) + int(gaps.sum())
            if width <= SIZE - 24:
                break
            height -= 1
        assert height >= 26
        x = int(rng.integers(12, SIZE - width - 11))
        top = min(offset for alpha, offset in letters)
        bottom = max(offset + alpha.shape[0] for alpha, offset in letters)
        baseline = int(12 + row * band + rng.integers(0, max(1, int(band - (bottom - top) - 4)))) - top
        for index, (class_id, (alpha, offset)) in enumerate(zip(classes, letters)):
            objects.append(paste_glyph(image, alpha, x, baseline + offset, foreground, class_id, row, glyphs[class_id]["char"]))
            x += alpha.shape[1] + (int(gaps[index]) if index < count - 1 else 0)
    return image, objects


def yolo_labels(objects, width, height, class_count):
    lines = []
    for item in objects:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        values = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]
        assert 0 <= item["class_id"] < class_count and all(0 < value <= 1 for value in values)
        lines.append(f"{item['class_id']} " + " ".join(f"{value:.8f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def draw_preview(image, objects):
    preview = image.copy()
    for item in objects:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 155, 255), 1)
        cv2.putText(preview, item["char"], (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 155, 255), 1)
    return cv2.resize(preview, (256, 256), interpolation=cv2.INTER_AREA)


def make_demo(glyphs, output):
    names = "".join(glyph["char"] for glyph in glyphs)
    texts = ["Spline", "FLEECE TAPE", "ColdMelt", "splice point"] if len(glyphs) == 52 else ["SPLINE", "FLEECE TAPE", "COLDMELT", "SPLICE POINT"]
    if len(glyphs) == 62:
        texts = ["Abc123", "FLEECE TAPE", "xY789", "456O0lI1"]
    elif len(glyphs) == 67:
        texts = ["PNO:SA-C8D", "A/B.12'", "xY-789", "0O:lI1."]
    backgrounds = [(105, 78, 75), (28, 32, 38), (240, 242, 245), (105, 78, 75)]
    image = np.zeros((384, 640, 3), dtype=np.uint8)
    objects, rows = [], []
    for row, (text, background) in enumerate(zip(texts, backgrounds)):
        image[row * 96:(row + 1) * 96] = background
        foreground = np.full(3, 20 if sum(background) > 450 else 250)
        letters = [None if char == " " else resize_glyph(glyphs[names.index(char)], 44, 1 + row % 2) for char in text]
        width = sum(23 if letter is None else letter[0].shape[1] + 3 for letter in letters) - 3
        x, baseline = (640 - width) // 2, row * 96 + 70
        row_objects = []
        for char, letter in zip(text, letters):
            if letter is None:
                x += 23
                continue
            alpha, offset = letter
            item = paste_glyph(image, alpha, x, baseline + offset, foreground, names.index(char), row, char)
            row_objects.append(item)
            objects.append(item)
            x += alpha.shape[1] + 3
        rows.append({"row": row, "text": text, "bbox_xyxy": [min(o["bbox_xyxy"][0] for o in row_objects), min(o["bbox_xyxy"][1] for o in row_objects), max(o["bbox_xyxy"][2] for o in row_objects), max(o["bbox_xyxy"][3] for o in row_objects)]})
    write_image(output / "demo_spline.png", image)
    yolo_labels(objects, 640, 384, len(glyphs))
    truth = {"image": "demo_spline.png", "width": 640, "height": 384, "rows": rows, "objects": objects,
             "evaluation_scope": "Synthetic demonstration using the same source glyph templates as training; not an independent real-world test."}
    (output / "demo_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")


def prepare(source, output, lower_source=None, digit_source=None, punctuation_source=None):
    output.mkdir(parents=True, exist_ok=True)
    glyphs, boxes = extract_glyphs(source, output)
    source_copy = output / "source_alphabet.png"
    if source.resolve() != source_copy.resolve():
        shutil.copy2(source, source_copy)
    sources = [{"path": str(source.resolve()), "copy": source_copy.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "classes": list(LETTERS)}]
    if lower_source:
        lower_glyphs, lower_boxes = extract_glyphs(lower_source, output, string.ascii_lowercase, 26)
        glyphs.extend(lower_glyphs)
        boxes.extend(lower_boxes)
        lower_copy = output / "source_lowercase.png"
        if lower_source.resolve() != lower_copy.resolve():
            shutil.copy2(lower_source, lower_copy)
        sources.append({"path": str(lower_source.resolve()), "copy": lower_copy.name,
                        "sha256": hashlib.sha256(lower_source.read_bytes()).hexdigest(), "classes": list(string.ascii_lowercase)})
    if digit_source:
        if not lower_source:
            raise ValueError("Digit training requires --lower-source for the 62-class alphabet")
        digit_glyphs, digit_boxes = extract_glyphs(digit_source, output, string.digits, 52)
        glyphs.extend(digit_glyphs)
        boxes.extend(digit_boxes)
        digit_copy = output / "source_digits.png"
        if digit_source.resolve() != digit_copy.resolve():
            shutil.copy2(digit_source, digit_copy)
        sources.append({"path": str(digit_source.resolve()), "copy": digit_copy.name,
                        "sha256": hashlib.sha256(digit_source.read_bytes()).hexdigest(), "classes": list(string.digits)})
    if punctuation_source:
        if not digit_source:
            raise ValueError("Punctuation training requires both alphabet cases and digits")
        punctuation_glyphs, punctuation_boxes = extract_glyphs(
            punctuation_source, output, PUNCTUATION, 62, reference_char="/")
        glyphs.extend(punctuation_glyphs)
        boxes.extend(punctuation_boxes)
        punctuation_copy = output / "source_punctuation.png"
        if punctuation_source.resolve() != punctuation_copy.resolve():
            shutil.copy2(punctuation_source, punctuation_copy)
        sources.append({"path": str(punctuation_source.resolve()), "copy": punctuation_copy.name,
                        "sha256": hashlib.sha256(punctuation_source.read_bytes()).hexdigest(),
                        "classes": list(PUNCTUATION)})
    names = [glyph["char"] for glyph in glyphs]
    assert len(names) == len(set(names)) == (67 if punctuation_source else (62 if digit_source else (52 if lower_source else 26)))
    splits = PUNCTUATION_SPLITS if punctuation_source else (ALPHANUMERIC_SPLITS if digit_source else (MIXED_SPLITS if lower_source else SPLITS))
    all_hashes, split_info, previews = set(), {}, []
    for split, (count, seed) in splits.items():
        rng = np.random.default_rng(seed)
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        blank_indices = set(rng.choice(count, round(count * 0.1), replace=False).tolist())
        class_counts = np.zeros(len(glyphs), dtype=int)
        hashes = []
        for index in range(count):
            image, objects = make_scene(glyphs, rng, index in blank_indices)
            name = f"{split}_{index:04d}"
            digest = write_image(image_dir / f"{name}.png", image)
            assert digest not in all_hashes, "Duplicate scene across train/validation/test"
            all_hashes.add(digest)
            hashes.append(digest)
            label_text = yolo_labels(objects, SIZE, SIZE, len(glyphs))
            (label_dir / f"{name}.txt").write_text(label_text, encoding="utf-8")
            for item in objects:
                class_counts[item["class_id"]] += 1
            if split == "train" and len(previews) < 8:
                previews.append(draw_preview(image, objects))
        assert (class_counts > 0).all(), f"Missing classes in {split}"
        split_info[split] = {"images": count, "negative_images": len(blank_indices), "seed": seed,
                             "objects": int(class_counts.sum()), "class_counts": dict(zip(names, class_counts.tolist())),
                             "image_sha256": hashes}
    sheet = np.concatenate([np.concatenate(previews[:4], axis=1), np.concatenate(previews[4:8], axis=1)], axis=0)
    write_image(output / "dataset_preview.png", sheet)
    make_demo(glyphs, output)
    yaml = f"path: {json.dumps(output.resolve().as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {index}: {json.dumps(letter)}\n" for index, letter in enumerate(names))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    # shortcut: one source template per class; add independent real CAD screenshots before claiming deployment performance.
    manifest = {"source": str(source.resolve()), "source_copy": "source_alphabet.png",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "sources": sources, "source_glyph_boxes": boxes,
                "classes": names, "image_size": [SIZE, SIZE], "splits": split_info,
                "glyph_geometry": "Common row cap/ascender height from H/h/0 and punctuation slash. Source-relative small marks, x-height and descenders retained; baseline is the reference character's exclusive stroke bottom. Thin punctuation label boxes have minimum 4px sides to survive augmentation; pixels are unchanged.",
                "scope": "Same-template synthetic feasibility experiment only.",
                "limitations": [f"All splits reuse the same {len(glyphs)} glyph templates from {len(sources)} source image(s).",
                                "Disjoint scene hashes and random seeds do not create independent font or drawing samples.",
                                "Synthetic metrics cannot establish performance on unseen real CAD drawings.",
                                "Only five supplied punctuation marks; no Korean or other fonts are represented." if punctuation_source else ("No punctuation or other fonts are represented." if digit_source else ("No digits, punctuation, or other fonts are represented." if lower_source else "No digits, lowercase letters, punctuation, or other fonts are represented.")),
                                "Demo image also reuses the training glyph templates."],
                "checks": {"glyph_count": len(glyphs), "all_classes_present_per_split": True,
                           "labels_within_image": True, "unique_scene_hashes": len(all_hashes),
                           "scene_splits_disjoint": True}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "splits": {split: {k: v for k, v in info.items() if k in ("images", "negative_images", "objects")} for split, info in split_info.items()}, "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lower-source", type=Path)
    parser.add_argument("--digit-source", type=Path)
    parser.add_argument("--punctuation-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_output = Path(__file__).resolve().parent
    if args.punctuation_source:
        default_output /= "punctuation"
    elif args.digit_source:
        default_output /= "alphanumeric"
    elif args.lower_source:
        default_output /= "mixed"
    prepare(args.source, args.output or default_output, args.lower_source, args.digit_source, args.punctuation_source)
