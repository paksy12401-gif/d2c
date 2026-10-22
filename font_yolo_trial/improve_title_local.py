"""Local title reader: source-owned crops and model-supported punctuation, no lexicon."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

import cv2
import numpy as np

from boundary_consensus import spatial_consensus
from ctc_korean_fusion import attach_confidence, choose_spans
from experiment_text_boundaries import proposals_from_cache
from iterate_fusion import fuse_words, lines
from read_table_title_local import spans
from retry_table_title_local import crop_image
from vector_extra_symbols import extra_symbol, supplement

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models_local import _offline
from paddle_ocr_local import model_dirs


def punctuation_kind(group, cap, baseline):
    a, b, c, d = group['box']; width = c-a; height = d-b
    if 0 < width < .35*cap and .12*cap < height < .45*cap and d > baseline:
        return ','
    if len(group['segments']) == 1:
        x0, y0, x1, y1 = group['segments'][0]
        if (x1-x0)*(y1-y0) < 0 and min(width, height) > .55*cap:
            return '/'
    kind = extra_symbol(group, cap, baseline)
    # Internal Hangul bars also satisfy the dash geometry heuristic.
    return kind if kind in ('(', ')') else None


def signature(group, cap):
    a, b, _, _ = group['box']
    return tuple(sorted(tuple(round((v-(a if i%2 == 0 else b))/cap, 3)
                              for i, v in enumerate(s)) for s in group['segments']))


def model_supported_punctuation(field, diagnostics, base):
    candidates = []
    for i, group in enumerate(field['groups']):
        kind = punctuation_kind(group, field['cap'], field['baseline'])
        if kind:
            candidates.append(dict(group_index=i, text=kind, box_pdf=group['box'],
                                   signature=signature(group, field['cap']), support=[]))
    for record in diagnostics:
        text, info = record['raw']['rec_text']
        positions = [x for part in info[2] for x in part]
        image_width = record.get('image_width')
        if image_width is None:
            image_width = cv2.imdecode(np.fromfile(base/(record['variant']+'.png'), np.uint8), 1).shape[1]
        target = int(re.search(r'h(\d+)', record['variant']).group(1))
        pad = {40:.15, 56:.25, 80:.35}[target]
        scale = target/field['cap']; origin = field['box'][0]-pad*field['cap']
        if record['variant'].startswith('native'):
            origin = math.floor(origin*scale)/scale
        origin = record.get('origin', [origin])[0]
        scale = record.get('scale', scale)
        for char, col, score in zip(text, positions, info[4]):
            if char not in ',/()' or score < .5:
                continue
            x = origin+col/info[0]*image_width/scale
            nearest = min((p for p in candidates if p['text'] == char),
                          key=lambda p: abs((p['box_pdf'][0]+p['box_pdf'][2])/2-x), default=None)
            if nearest is not None and abs((nearest['box_pdf'][0]+nearest['box_pdf'][2])/2-x) < field['cap']*.55:
                nearest['support'].append(dict(variant=record['variant'], score=score))
    # Copy a model label only across translation-equivalent original source shapes.
    for p in candidates:
        equivalent = [q for q in candidates if q['text'] == p['text'] and q['signature'] == p['signature']]
        observations = [s for q in equivalent for s in q['support']]
        if len({s['variant'] for s in observations}) < 3:
            continue
        yield dict(text=p['text'], box_pdf=p['box_pdf'], group_index=p['group_index'],
                   score=median(s['score'] for s in observations),
                   engine='model_and_source_punctuation',
                   model_support=observations)


def owned_window(field, window, punctuation):
    box = list(window); cap = field['cap']; cx = (box[0]+box[2])/2
    left = max((p['box_pdf'][2] for p in punctuation if p['box_pdf'][2] < cx), default=None)
    right = min((p['box_pdf'][0] for p in punctuation if p['box_pdf'][0] > cx), default=None)
    if left is not None and abs(left-box[0]) < .6*cap:
        box[0] = left
    if right is not None and abs(right-box[2]) < .6*cap:
        box[2] = right
    excluded = {p['group_index'] for p in punctuation}
    groups = [g for i, g in enumerate(field['groups']) if i not in excluded and
              box[0]-cap*.02 <= (g['box'][0]+g['box'][2])/2 <= box[2]+cap*.02]
    if not groups:
        raise ValueError('Empty isolated input')
    bounds = [min(g['box'][0] for g in groups), min(g['box'][1] for g in groups),
              max(g['box'][2] for g in groups), max(g['box'][3] for g in groups)]
    return dict(field, groups=groups), bounds


def inject_punctuation(glyphs, punctuation, cap):
    kept = [g for g in glyphs if not any(p['box_pdf'][0]-.03*cap <= (g['box_pdf'][0]+g['box_pdf'][2])/2 <= p['box_pdf'][2]+.03*cap
            and p['box_pdf'][1]-.03*cap <= (g['box_pdf'][1]+g['box_pdf'][3])/2 <= p['box_pdf'][3]+.03*cap for p in punctuation)]
    return kept+[dict(p) for p in punctuation]


def automatic_hangul_windows(initial):
    field = initial['field']; cap = field['cap']; candidates = []
    for record in initial['ocr_diagnostics']:
        text, info = record['raw']['rec_text']
        positions = [p for group in info[2] for p in group]
        for match in re.finditer(r'[\uac00-\ud7a3]+', text):
            a, b = match.span()
            x0 = record['origin'][0]+positions[a]/info[0]*record['image_width']/record['scale']-.6*cap
            x1 = record['origin'][0]+positions[b-1]/info[0]*record['image_width']/record['scale']+.6*cap
            candidates.append([max(field['box'][0], x0), field['box'][1],
                               min(field['box'][2], x1), field['box'][3]])
    windows = []
    for box in sorted(candidates):
        old = next((w for w in windows if min(w[2], box[2])-max(w[0], box[0]) > .3*cap), None)
        if old is None:
            windows.append(box.copy())
        else:
            old[0] = min(old[0], box[0]); old[2] = max(old[2], box[2])
    return windows


def improve_field(initial, windows, model, base, output, retry=None):
    output.mkdir(parents=True, exist_ok=True)
    field = initial['field']; cap = field['cap']
    punctuation = list(model_supported_punctuation(field, initial['ocr_diagnostics'], base))
    proposals = proposals_from_cache(initial, retry or dict(diagnostics=[]), True)
    consensus = spatial_consensus(proposals, cap)
    # A crop-only high score must not silently replace a stable full-context word.
    proposals = [dict(p, variant='previous_word') for p in consensus if p['boundary_stable']]
    jobs = []; inputs = []
    for i, window in enumerate(windows):
        owned, box = owned_window(field, window, punctuation)
        inputs.append(dict(index=i, box_pdf=box, groups=len(owned['groups'])))
        for target in (48, 64, 80):
            for stroke in (1., 2., 3.):
                # crop_image has a fixed normalized height; target controls resampling only.
                image, origin, scale = crop_image(owned, box, stroke, 1.)
                factor = target/64
                image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                name = f'owned{i}_h{target}_s{stroke}'
                cv2.imencode('.png', image)[1].tofile(output/(name+'.png'))
                jobs.append((i, name, image, origin, scale*factor, box))
    diagnostics = []; clean_words = defaultdict(list)
    for i, name, image, origin, scale, box in jobs:
        result = next(iter(model.predict(input=image, batch_size=1, return_word_box=True)))
        raw = result.json.get('res', result.json)
        text, info = raw['rec_text']; text = text.strip()
        score = float(np.mean(info[4])) if info[4] else 0.
        diagnostics.append(dict(variant=name, raw=raw, box_pdf=box))
        if re.fullmatch(r'[\uac00-\ud7a3]+', text) and score >= .88:
            clean_words[(i, text)].append(dict(text=text, accepted=text, score=score,
                                              box_pdf=box, variant=name, engine='isolated_source_ocr'))
        print('OWNED_OCR', name, json.dumps(text, ensure_ascii=True), round(score, 3), flush=True)
    for (i, text), observations in clean_words.items():
        heights = {re.search(r'_h(\d+)', p['variant']).group(1) for p in observations}
        if len(heights) < 3:
            continue
        p = deepcopy(max(observations, key=lambda p: p['score']))
        p.update(score=median(p['score'] for p in observations), variants=[p['variant'] for p in observations],
                 boundary_method='all_ink_in_isolated_source_window', input_index=i)
        proposals.append(p)
    selected = choose_spans(proposals, cap)
    combined = []
    for run in initial['results']:
        glyphs = inject_punctuation(run['glyphs'], punctuation, cap)
        glyphs = supplement(glyphs, field, selected)
        fused, suppressed = fuse_words(glyphs, selected)
        combined.append(dict(config=run['config'], fused_lines=lines(fused),
                             glyphs=glyphs, fused_glyphs=fused, suppressed=suppressed))
    report = dict(llm_inference=False, answers_loaded=False, dictionary_correction=False,
                  source_inputs={name: hashlib.sha256((base/name).read_bytes()).hexdigest()
                                 for name in (('recognition.json', 'retry_recognition.json') if retry else ('recognition.json',))},
                  field=field, isolated_inputs=inputs, punctuation=punctuation, diagnostics=diagnostics,
                  korean_spans=selected, results=combined,
                  limitations=['Selected vector text field, not full drawing validation.',
                               'Source-owned windows use local full-context OCR proposals.',
                               'Repeated OCR outputs are correlated, not calibrated confidence.'])
    (output/'recognition.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('IMPROVED_RESULT', json.dumps([dict(config=r['config'], lines=r['fused_lines']) for r in combined], ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with source geometry and previous local predictions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    initial = json.loads((args.input/'recognition.json').read_text(encoding='utf-8'))
    retry = json.loads((args.input/'retry_recognition.json').read_text(encoding='utf-8'))
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _, directory = model_dirs()
        model = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(directory),
                                device='cpu', enable_mkldnn=False, cpu_threads=4)
        attach_confidence(model)
        improve_field(initial, retry['windows'], model, args.input, args.output, retry)


if __name__ == '__main__':
    main()
