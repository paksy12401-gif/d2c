"""Offline table-cell experiment. This inference file never loads answer sheets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'output/cad_table_benchmark'
SOURCE = ROOT / 'font_yolo_trial/cad_without_tables/source.pdf'


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode('.png', image)[1].tobytes())


def lines(profile, threshold):
    indexes = np.flatnonzero(profile > threshold)
    groups = np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
    return [int(np.median(g)) for g in groups if len(g)]


def prepare():
    OUT.mkdir(exist_ok=True)
    page = fitz.open(SOURCE)[0]
    clip = fitz.Rect(5045, 270, 7140, 1240)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    image = rgb[:, :, ::-1].copy()
    write_image(OUT / 'left_source.png', image)
    a = rgb.astype(float)
    blue = (a[:, :, 2] > a[:, :, 0] * 1.1) & (a[:, :, 2] > a[:, :, 1] * 1.1) & (a[:, :, 2] > 90)
    xs = lines(blue.sum(0), image.shape[0] * .7)
    ys = lines(blue.sum(1), image.shape[1] * .7)
    assert len(xs) == 6 and len(ys) == 33, (xs, ys)
    blocks = [b for b in page.get_text('dict')['blocks'] if b['type'] == 1]
    assert len(blocks) == 1
    block = blocks[0]
    right = cv2.imdecode(np.frombuffer(block['image'], np.uint8), cv2.IMREAD_COLOR)
    write_image(OUT / 'right_source.png', right)
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    ink = gray < 70
    rx = lines(ink.sum(0), right.shape[0] * .95)
    ry = lines(ink.sum(1), right.shape[1] * .95)
    # Outer bottom/right edges sit at the image boundary rather than a full black line.
    if ry[-1] < right.shape[0] - 2: ry.append(right.shape[0])
    rx[-1] = right.shape[1]
    assert len(rx) == 11 and len(ry) == 73 and right.shape[:2] == (1584, 1204)
    drawings = page.get_drawings()
    cells = []
    for table, raster, xx, yy in [('left', image, xs, ys), ('right', right, rx, ry)]:
        for row in range(len(yy) - 1):
            for col in range(len(xx) - 1):
                box = [xx[col], yy[row], xx[col + 1], yy[row + 1]]
                margin = 3 if table == 'left' else 2
                raw = raster[box[1] + margin:box[3] - 1, box[0] + margin:box[2] - 1]
                identity = f'{table}_r{row:02d}_c{col}'
                write_image(OUT / 'cells' / (identity + '.png'), raw)
                record = dict(id=identity, table=table, row=row, column=col, bbox_px=box)
                if table == 'left':
                    pdfbox = fitz.Rect(clip.x0 + box[0] / 2, clip.y0 + box[1] / 2,
                                      clip.x0 + box[2] / 2, clip.y0 + box[3] / 2)
                    paths = [d for d in drawings if pdfbox.contains(d['rect'])]
                    assert len(paths) == 1, (identity, len(paths))
                    segments = []
                    for item in paths[0]['items']:
                        if item[0] == 'l': segments.append(item)
                        elif item[0] == 'qu':
                            q = item[1]
                            segments.extend(('l', a, b) for a, b in [(q.ul,q.ur),(q.ur,q.lr),(q.lr,q.ll),(q.ll,q.ul)])
                        else: raise ValueError((identity, item[0]))
                    scale = 4
                    clean = np.full((int(pdfbox.height * scale) + 1, int(pdfbox.width * scale) + 1, 3), 255, np.uint8)
                    for _, a, b in segments:
                        p0 = (round((a.x - pdfbox.x0) * scale), round((a.y - pdfbox.y0) * scale))
                        p1 = (round((b.x - pdfbox.x0) * scale), round((b.y - pdfbox.y0) * scale))
                        cv2.line(clean, p0, p1, (0, 0, 0), max(1, round(paths[0]['width'] * scale)), cv2.LINE_AA)
                    record.update(bbox_pdf=list(pdfbox), vector_segments=len(segments))
                else:
                    clean = raw.copy()
                    x0, y0, x1, y1 = block['bbox']
                    record['bbox_pdf'] = [x0 + box[0] / right.shape[1] * (x1 - x0),
                        y0 + box[1] / right.shape[0] * (y1 - y0),
                        x0 + box[2] / right.shape[1] * (x1 - x0),
                        y0 + box[3] / right.shape[0] * (y1 - y0)]
                write_image(OUT / 'clean' / (identity + '.png'), clean)
                cells.append(record)
    assert len(cells) == 880
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        llm_inference=False, answer_sheet_used=False,
        left=dict(xs=xs, ys=ys, rows=32, columns=5, origin_pdf=list(clip[:2]), scale=2),
        right=dict(xs=rx, ys=ry, rows=72, columns=10, bbox_pdf=block['bbox']), cells=cells)
    (OUT / 'cells.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PREPARED', Counter(c['table'] for c in cells), flush=True)


def recognizer_input(image, tight):
    if tight:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 200)
        if len(xs): image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scale = 32 / image.shape[0]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.copyMakeBorder(image, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    image = cv2.copyMakeBorder(image, 5, 5, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    scale = 48 / image.shape[0]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def recognize():
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    meta = json.loads((OUT / 'cells.json').read_text())
    _, directory = model_dirs()
    jobs = [(c, mode) for mode in ['raw_cell', 'clean_cell'] for c in meta['cells']]
    results = []
    started = time.monotonic()
    prepared = {}
    for cell, mode in jobs:
        folder = 'clean' if mode == 'clean_cell' else 'cells'
        image = cv2.imdecode(np.fromfile(OUT / folder / (cell['id'] + '.png'), dtype=np.uint8), 1)
        image = recognizer_input(image, mode == 'clean_cell')
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        entry = prepared.setdefault(digest, dict(image=image, addresses=[]))
        entry['addresses'].append((cell, mode))
    cached = {}
    previous = OUT / 'predictions.json'
    if previous.exists():
        old = json.loads(previous.read_text(encoding='utf-8'))
        assert old['source_sha256'] == meta['source_sha256']
        by_address = {(r['id'], r['variant']): r for r in old['results']}
        for digest, entry in prepared.items():
            for cell, mode in entry['addresses']:
                value = by_address.get((cell['id'], mode))
                if value is not None:
                    cached[digest] = dict(text=value['text'], score=value['score'])
                    break
    def add_results(digest, value):
        for cell, mode in prepared[digest]['addresses']:
            results.append(dict(**cell, variant=mode, **value, inference_digest=digest))
    for digest, value in cached.items(): add_results(digest, value)
    pending = sorted((d for d in prepared if d not in cached), key=lambda d: prepared[d]['image'].shape[1])
    print('UNIQUE INPUTS', len(prepared), 'CACHED', len(cached), 'PENDING', len(pending), flush=True)
    with _offline(paddle_cache=ROOT / '.model_cache/paddlex'):
        from paddleocr import TextRecognition
        model = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(directory),
                                device='cpu', enable_mkldnn=False, cpu_threads=4)
        for start in range(0, len(pending), 16):
            batch = pending[start:start + 16]
            images = [prepared[d]['image'] for d in batch]
            predictions = list(model.predict(input=images, batch_size=16))
            assert len(predictions) == len(batch)
            for digest, prediction in zip(batch, predictions):
                raw = prediction.json.get('res', prediction.json)
                add_results(digest, dict(text=str(raw.get('rec_text', '')), score=float(raw.get('rec_score', 0))))
            (OUT / 'predictions.json').write_text(json.dumps(dict(source_sha256=meta['source_sha256'],
                llm_inference=False, answer_sheet_used=False, elapsed_seconds=time.monotonic() - started,
                results=results), ensure_ascii=False, indent=2), encoding='utf-8')
            print('OCR', len(results), '/', len(jobs), 'seconds', round(time.monotonic() - started), flush=True)
    assert len(results) == len(jobs) and len({(r['id'], r['variant']) for r in results}) == len(jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'recognize'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else recognize()
