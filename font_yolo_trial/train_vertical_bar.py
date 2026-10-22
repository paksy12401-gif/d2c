"""Add user-labelled | (class 67); preserve all existing class IDs and weights.

Synthetic splits share glyph templates: they are diagnostics, not new-drawing accuracy.
"""
import argparse
import hashlib
import json
import shutil
import string
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from prepare import extract_glyphs, paste_glyph, read_image, resize_glyph, write_image, yolo_labels
from train import CharacterTrainer

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'punctuation_fixed'
OUT = ROOT / 'vertical_bar'
NAMES = string.ascii_uppercase + string.ascii_lowercase + string.digits + ":-/.'|"
SOURCE = Path('C:/Users/Win11Pro/AppData/Local/Temp/codex-clipboard-66db3522-c3d4-472b-a9bb-a2b744965b17.png')


def extract_bars(image):
    # The grey CAD cursor is not part of the user-labelled white symbols.
    mask = (image.min(axis=2) >= 245).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    bars, boxes = [], []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if h < 20 or h / max(w, 1) < 4 or area / (w*h) < .8:
            continue
        alpha = np.pad((labels[y:y+h, x:x+w] == i).astype(np.float32), 1)
        bars.append(dict(alpha=alpha, char='|', reference_height=h, top_from_baseline=-h-1))
        boxes.append([x, y, x+w, y+h])
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    return [bars[i] for i in order], [boxes[i] for i in order]


def scene(glyphs, bars, seed, negative=False):
    rng = np.random.default_rng(seed)
    light = rng.random() < .5
    bg = np.full(3, 255 if light else 32, dtype=np.uint8)
    image = np.full((384, 384, 3), bg, np.uint8)
    fg = np.full(3, 0 if light else 255)
    if negative:
        # Connected/long CAD lines, not identical short isolated glyphs with false labels.
        for _ in range(3):
            x, y = map(int, rng.integers(15, 180, 2))
            cv2.rectangle(image, (x, y), (x+100, min(375, y+180)), tuple(map(int, fg)), 1)
        return image, []
    objects = []
    for row in range(3):
        cap = int(rng.integers(26, 87))
        ids = rng.integers(0, 67, 5).tolist()
        ids[int(rng.integers(5))] = 67
        if rng.random() < .65:
            ids[int(rng.integers(5))] = int(rng.choice([8, 37, 63, 53, 67]))
        pieces = []
        for cid in ids:
            if cid == 67:
                glyph = bars[int(rng.integers(len(bars)))]
                alpha, offset = resize_glyph(glyph, cap, 1)
                width = max(3, round(cap * rng.uniform(.04, .16)))
                alpha = cv2.resize(alpha, (width+2, alpha.shape[0]))
            else:
                alpha, offset = resize_glyph(glyphs[cid], cap, int(rng.integers(1, 3)))
            pieces.append((alpha, offset))
        gaps = rng.integers(2, 15, 4)
        width = sum(p[0].shape[1] for p in pieces) + int(gaps.sum())
        if width > 360:
            ratio = 350 / width
            pieces = [(cv2.resize(a, (max(3, round(a.shape[1]*ratio)), max(3, round(a.shape[0]*ratio)))), round(o*ratio)) for a, o in pieces]
            gaps = np.maximum(1, np.rint(gaps*ratio).astype(int))
            width = sum(p[0].shape[1] for p in pieces) + int(gaps.sum())
        x = int(rng.integers(10, max(11, 374-width)))
        top = min(o for a, o in pieces)
        baseline = row*122+12-top
        for j, (cid, (alpha, offset)) in enumerate(zip(ids, pieces)):
            obj = paste_glyph(image, alpha, x, baseline+offset, fg, cid, row, NAMES[cid])
            if cid == 67 and obj['bbox_xyxy'][2]-obj['bbox_xyxy'][0] < 4:
                obj['bbox_xyxy'][2] = obj['bbox_xyxy'][0]+4
            objects.append(obj)
            x += alpha.shape[1] + (int(gaps[j]) if j < 4 else 0)
    return image, objects


def prepare(source):
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Dataset already exists; use the existing dataset, do not overwrite.')
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, OUT/'source_vertical_bar.png')
    bars, boxes = extract_bars(read_image(source))
    assert len(bars) == 3, boxes
    glyphs = []
    for name, chars in [('alphabet', string.ascii_uppercase), ('lowercase', string.ascii_lowercase), ('digits', string.digits), ('punctuation', ":-/.'")]:
        found, _ = extract_glyphs(OLD/f'source_{name}.png', OUT, chars, len(glyphs), reference_char='/' if name == 'punctuation' else None)
        glyphs.extend(found)
    for i, bar in enumerate(bars):
        write_image(OUT/f'glyphs/67_vertical_bar_{i}.png', np.rint(bar['alpha']*255).astype(np.uint8))
    splits = {}
    for split, count, seed in [('train', 416, 191000), ('val', 104, 192000), ('test', 130, 193000)]:
        for kind in ('images', 'labels'):
            dest = OUT/kind/split
            dest.mkdir(parents=True, exist_ok=True)
            for path in sorted((OLD/kind/split).glob('*')):
                if path.suffix not in ('.png', '.txt'):
                    continue
                shutil.copy2(path, dest/('replay_'+path.name))
        for i in range(count):
            image, objects = scene(glyphs, bars, seed+i, negative=i % 10 == 0)
            write_image(OUT/f'images/{split}/bar_{i:04d}.png', image)
            (OUT/f'labels/{split}/bar_{i:04d}.txt').write_text(yolo_labels(objects, 384, 384, 68), encoding='utf-8')
        splits[split] = dict(replay=len(list((OUT/f'images/{split}').glob('replay_*.png'))), added=count)
    manifest = dict(classes=list(NAMES), new_class_id=67, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_boxes=boxes, splits=splits,
        scope='All splits share supplied glyph templates. Original screenshot is training material, not independent validation. No drawing text answers or LLM recognition used.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT.resolve()), train='images/train', val='images/val', test='images/test', names=dict(enumerate(NAMES))), sort_keys=False), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not (OUT/'manifest.json').exists():
        prepare(args.source)
    assert json.loads((OUT/'manifest.json').read_text())['classes'] == list(NAMES)
    if (OUT/'training').exists():
        raise FileExistsError('Existing training is preserved, not overwritten.')
    weights = OLD/'training/weights/best.pt'
    initial_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    YOLO(str(weights)).train(trainer=CharacterTrainer, data=str(OUT/'data.yaml'), epochs=args.epochs,
        imgsz=384, batch=16, device=0, workers=0, project=str(OUT), name='training', seed=20261019,
        cls_remap=False, optimizer='AdamW', lr0=.0005, lrf=.2, warmup_epochs=1, patience=12,
        amp=False, cache=False, plots=False, mosaic=0, mixup=0, cutmix=0, copy_paste=0,
        degrees=0, translate=0, scale=0, shear=0, perspective=0, fliplr=0, flipud=0,
        hsv_h=0, hsv_s=0, hsv_v=0)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == initial_sha
    print('VERTICAL BAR TRAINING COMPLETE; original weights unchanged', flush=True)


if __name__ == '__main__':
    main()
