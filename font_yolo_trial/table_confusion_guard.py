"""Recover open C glyphs only with source shape, YOLO and two OCR views."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from refine_table_ocr import compact, index_map, BASE, ROOT
from train_table_font_verifier import patch


def open_c_shape(image):
    mask = (image < 180).astype(np.uint8)*255
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return dict(compatible=False, reason='empty_or_tiny')
    ink = mask[ys.min():ys.max()+1, xs.min():xs.max()+1] > 0
    h, w = ink.shape
    if min(h, w) < 8:
        return dict(compatible=False, reason='insufficient_resolution')
    _, hierarchy = cv2.findContours(ink.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else sum(p[3] >= 0 for p in hierarchy[0])
    middle = ink[round(h*.35):round(h*.65)]
    right_ink = float(middle[:, round(w*.8):].mean())
    left_ink = bool(middle[:, :max(1, round(w*.2))].any())
    top = np.where(ink[:max(1, round(h*.2))])[1]
    bottom = np.where(ink[round(h*.8):])[1]
    top_span = float(np.ptp(top)+1)/w if len(top) else 0.
    bottom_span = float(np.ptp(bottom)+1)/w if len(bottom) else 0.
    compatible = holes == 0 and right_ink < .03 and left_ink and min(top_span, bottom_span) >= .5
    return dict(compatible=bool(compatible), holes=int(holes), right_middle_ink=right_ink,
                left_middle_present=left_ink, top_span=top_span, bottom_span=bottom_span)


def supported_c(old, glyph, ocr_candidates, shape, protected=False):
    return (not protected and old in 'O0' and glyph.get('text') == 'C'
            and glyph.get('confidence', 0) >= .9
            and len(ocr_candidates) == 2
            and all(p['text'] == 'C' and p['score'] >= .95 for p in ocr_candidates)
            and shape['compatible'])


def repair_rows(rows, cells, glyphs, english, output):
    results = deepcopy(rows); changes = []
    for row in results:
        row['shape_confusion_updates'] = []
        identity = row['id']; field = glyphs.get(identity)
        if row['table'] != 'left' or field is None:
            continue
        text = compact(row['text'])
        if len(text) != len(field['glyphs']):
            continue
        views = []
        for variant in ('normal', 'more_ink'):
            view = english.get((identity, variant))
            if view is None:
                break
            chars = [c for c in view['ctc']['chars'] if not c['text'].isspace()]
            views.append((variant, chars, index_map(text, ''.join(c['text'] for c in chars))))
        if len(views) != 2:
            continue
        protected = {i for m in re.finditer(r'\d{2,}', text) for i in range(m.start(), m.end())}
        image = None; corrected = list(text)
        for i, (old, glyph) in enumerate(zip(text, field['glyphs'])):
            if old not in 'O0' or glyph.get('text') != 'C' or i in protected:
                continue
            candidates = [dict(variant=v, **chars[mapping[i]]) for v, chars, mapping in views if i in mapping]
            if not supported_c(old, glyph, candidates, dict(compatible=True)):
                continue
            if image is None:
                image = cv2.imdecode(np.fromfile(BASE/'clean'/(identity+'.png'), np.uint8), 0)
            crop = patch(image, cells[identity], glyph, field['cap'])
            shape = open_c_shape(crop)
            if not supported_c(old, glyph, candidates, shape):
                continue
            corrected[i] = 'C'
            update = dict(index=i, before=old, after='C', box_pdf=glyph['box_pdf'],
                          yolo_score=glyph['confidence'], ocr_support=candidates, source_shape=shape)
            row['shape_confusion_updates'].append(update)
            if output is not None:
                cv2.imencode('.png', crop)[1].tofile(output/(identity+f'_char{i}.png'))
        if row['shape_confusion_updates']:
            row['before_shape_guard'] = row['text']
            iterator = iter(corrected)
            row['text'] = ''.join(c if c.isspace() else next(iterator) for c in row['text'])
            changes.append(dict(id=identity, before=row['before_shape_guard'], after=row['text'],
                                updates=row['shape_confusion_updates']))
    return results, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.input.read_text(encoding='utf-8'))
    cells = {r['id']:r for r in json.loads((BASE/'cells.json').read_text())['cells']}
    glyphs = {r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    en_file = ROOT/'output/cad_table_refinement/english_predictions.json'
    english = {(r['id'], r['variant']):r for r in json.loads(en_file.read_text(encoding='utf-8'))['results']}
    results, changes = repair_rows(source['results'], cells, glyphs, english, args.output)
    data = dict(llm_inference=False, answers_loaded=False, training=False, dictionary_correction=False,
                source_predictions_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                english_predictions_sha256=hashlib.sha256(en_file.read_bytes()).hexdigest(),
                results=results, changes=changes)
    (args.output/'predictions.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(processed=len(results), changes=changes), ensure_ascii=True, indent=1))


if __name__ == '__main__':
    main()
