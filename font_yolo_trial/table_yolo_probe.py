"""Probe the existing 62-class model on vector table glyphs, without references."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pymupdf as fitz
import torch
from ultralytics import YOLO

from table_cell_experiment import OUT, SOURCE
from vector_font_probe import bounds, group_glyphs, glyph_canvas, geometric_symbol, parse_linear_path
from vector_extra_symbols import extra_symbol


def connected_runs(runs):
    """Reconnect strokes sharing points before grouping glyphs by horizontal extent."""
    segments = [s for run in runs for s in run]
    a, b, c, d = bounds(segments)
    scale = 8
    image = np.zeros((round((d-b)*scale)+5, round((c-a)*scale)+5), np.uint8)
    points = []
    for x0, y0, x1, y1 in segments:
        p0 = (round((x0-a)*scale)+2, round((y0-b)*scale)+2)
        p1 = (round((x1-a)*scale)+2, round((y1-b)*scale)+2)
        cv2.line(image, p0, p1, 255, 1)
        points.append(p0)
    _, labels = cv2.connectedComponents(image, connectivity=8)
    components = {}
    for segment, (x,y) in zip(segments, points):
        label = int(labels[y,x])
        assert label > 0
        components.setdefault(label, []).append(segment)
    return list(components.values())


def main():
    torch.set_num_threads(4)
    cells = [c for c in json.loads((OUT / 'cells.json').read_text())['cells'] if c['table'] == 'left']
    page = fitz.open(SOURCE)[0]
    tree = ET.fromstring(page.get_svg_image(text_as_path=True))
    table_box = fitz.Rect(5045, 270, 7140, 1240)
    paths = {}
    for node in tree.iter():
        if not node.tag.endswith('path') or 'stroke' not in node.attrib: continue
        matrix = list(map(float, re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?', node.get('transform', 'matrix(1,0,0,1,0,0)'))))
        if len(matrix) != 6: continue
        runs = parse_linear_path(node.get('d', ''), matrix)
        if not runs: continue
        box = fitz.Rect(bounds([s for run in runs for s in run]))
        if not table_box.contains(box): continue
        owners = [c for c in cells if fitz.Rect(c['bbox_pdf']).contains(box)]
        if len(owners) != 1: continue
        owner = owners[0]['id']
        assert owner not in paths
        paths[owner] = dict(groups=group_glyphs(connected_runs(runs)), width=float(node.get('stroke-width', '.1')))
    assert len(paths) == 160
    records = []
    jobs = []
    for cell in cells:
        p = paths[cell['id']]
        groups = p['groups']
        heights = sorted(g['box'][3] - g['box'][1] for g in groups if g['box'][3] - g['box'][1] > .1)
        cap = float(np.median(heights[len(heights) // 3:]))
        baseline = float(np.median([g['box'][3] for g in groups if g['box'][3] - g['box'][1] > cap * .7]))
        record = dict(**cell, cap=cap, glyphs=[])
        for g in groups:
            a, b, c, d = g['box']
            text = geometric_symbol(g, cap, baseline) or extra_symbol(g, cap, baseline)
            if text is None and c - a < cap * .25 and d - b < cap * .25 and b > baseline - cap * .3:
                text = '.'
            if text is None and len(g['segments']) == 1:
                x0, y0, x1, y1 = g['segments'][0]
                if abs(x1 - x0) > cap * .3 and abs(y1 - y0) > cap * .7 and (x1 - x0) * (y1 - y0) < 0:
                    text = '/'
            glyph = dict(box_pdf=g['box'], text=text, confidence=1.0 if text else 0.0,
                         engine='vector_symbol' if text else 'yolo')
            record['glyphs'].append(glyph)
            if text is None: jobs.append((glyph, glyph_canvas(g, cap, 56, p['width'])))
        records.append(record)
    weights = Path(__file__).parent / 'alphanumeric/training/weights/best.pt'
    model = YOLO(str(weights))
    print('TABLE YOLO', len(records), 'GLYPHS', len(jobs), flush=True)
    for start in range(0, len(jobs), 32):
        batch = jobs[start:start + 32]
        predictions = model.predict([image for _, image in batch], imgsz=384, conf=.1,
            iou=.5, end2end=False, device=0, batch=32, verbose=False)
        for (glyph, _), result in zip(batch, predictions):
            boxes = result.boxes.cpu().numpy().data.tolist()
            centered = [b for b in boxes if abs((b[0] + b[2]) / 2 - 192) < 35 and abs((b[1] + b[3]) / 2 - 192) < 35]
            best = max(centered, key=lambda b:b[4]) if centered else None
            glyph.update(text=model.names[int(best[5])] if best else '?', confidence=best[4] if best else 0.0)
        if start % 256 == 0: print('GLYPHS', min(start + 32, len(jobs)), '/', len(jobs), flush=True)
    for record in records:
        record['text'] = ''.join(g['text'] for g in record['glyphs'])
    previous = OUT / 'yolo_predictions.json'
    if previous.exists() and not (OUT / 'yolo_run1.json').exists():
        (OUT / 'yolo_run1.json').write_bytes(previous.read_bytes())
    previous.write_text(json.dumps(dict(llm_inference=False, answer_sheet_used=False,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), results=records), ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__': main()
