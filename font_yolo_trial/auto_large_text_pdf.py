"""Discover relatively large vector text and run the shared local mixed reader."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

import ezdxf
import pymupdf as fitz

from ctc_korean_fusion import attach_confidence
from improve_title_local import automatic_hangul_windows, improve_field
from preserved_reader import PreservedReader
from read_table_title_local import recognize_field
from vector_font_probe import load_candidates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models_local import _offline
from paddle_ocr_local import model_dirs


def large_text_fields(candidates, ratio=2.):
    if ratio <= 1:
        raise ValueError('Large-text ratio must exceed 1')
    if not candidates:
        return [], dict(body_height=None, large_ratio=ratio, clusters=[])
    clusters = []
    for field in sorted(candidates, key=lambda p:p['cap']):
        if not clusters or field['cap']/clusters[-1][-1]['cap'] > 1.6:
            clusters.append([])
        clusters[-1].append(field)
    body = median(f['cap'] for f in max(clusters, key=len))
    selected = [dict(f, id=f"P{f['path_id']}", relative_height=f['cap']/body)
                for f in candidates if f['cap'] >= body*ratio]
    return selected, dict(body_height=body, large_ratio=ratio,
                           clusters=[dict(count=len(c), median_height=median(f['cap'] for f in c)) for c in clusters])


def existing_field_text(field, model, page_height):
    # Field identity alone is insufficient: require matching placement and scale.
    for e in model.query('TEXT'):
        matches = False
        for app in ('CAD_OCR_REVIEW', 'CAD_TITLE_LOCAL', 'CAD_LARGE_LOCAL'):
            if e.has_xdata(app) and any(t.code == 1000 and t.value == field['id'] for t in e.get_xdata(app)):
                matches = True
        if not matches or e.dxf.halign not in (3, 5):
            continue
        x0, x1 = sorted((e.dxf.insert.x*72, e.dxf.align_point.x*72))
        bottom = page_height-e.dxf.insert.y*72
        b = field['box']; cap = field['cap']
        if (max(abs(x0-b[0]), abs(x1-b[2])) < cap*.1 and
            min(abs(bottom-b[3]), abs(bottom-field['baseline'])) < cap*.2 and
            abs(e.dxf.height*72-cap) < cap*.2):
            return e.dxf.handle
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-dxf', type=Path)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--large-ratio', type=float, default=2.)
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    if not 1 <= args.page <= len(document):
        raise ValueError('PDF page out of range')
    page = document[args.page-1]; short = min(page.rect.width, page.rect.height)
    candidates = load_candidates(page, min_cap=short*.001, max_cap=short*.2)
    fields, sizing = large_text_fields(candidates, args.large_ratio)
    existing = ezdxf.readfile(args.base_dxf).modelspace() if args.base_dxf else None
    routes = []
    for field in fields:
        handle = existing_field_text(field, existing, page.rect.height) if existing is not None else None
        routes.append(dict(field=field, status='existing_editable_text' if handle else 'pending', existing_handle=handle))
    report = dict(source_pdf=str(args.pdf.resolve()), source_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                  page_index=args.page-1, page_height=page.rect.height, candidate_count=len(candidates),
                  size_classification=sizing, manual_rois=False, table_exclusions=[],
                  inference_llm=False, answers_loaded=False,
                  base_dxf=str(args.base_dxf.resolve()) if args.base_dxf else None,
                  base_dxf_sha256=hashlib.sha256(args.base_dxf.read_bytes()).hexdigest() if args.base_dxf else None,
                  routes=routes,
                  limits=['Automatic discovery currently covers linear stroked PDF paths only; raster-only or curved paths need a separate detector.',
                          'Path grouping and relative size classification are heuristics, not complete text detection.',
                          'Same-font recognition remains dependent on separation, line thickness and model coverage.'])
    manifest = args.output/'automatic_results.json'
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('AUTO_DISCOVERY', json.dumps(dict(sizing=sizing, candidates=len(candidates),
                 routes=[dict(id=r['field']['id'], status=r['status'], cap=r['field']['cap']) for r in routes]), ensure_ascii=True), flush=True)
    pending = [r for r in routes if r['status'] == 'pending']
    if args.discover_only or not pending:
        return
    import torch
    torch.set_num_threads(4)
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        reader = PreservedReader()
        from paddleocr import TextRecognition
        _, directory = model_dirs()
        model = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(directory),
                                device='cpu', enable_mkldnn=False, cpu_threads=4)
        attach_confidence(model)
        for route in pending:
            field = route['field']; folder = args.output/field['id']; raw = folder/'initial'
            raw.mkdir(parents=True, exist_ok=True)
            (raw/'source_geometry.json').write_text(json.dumps(dict(source_pdf=str(args.pdf.resolve()),
                      page_index=args.page-1, field=field, llm_inference=False, answers_loaded=False), ensure_ascii=False, indent=2), encoding='utf-8')
            initial = recognize_field(page, field, raw, reader, model)
            windows = automatic_hangul_windows(initial)
            result = improve_field(initial, windows, model, raw, folder/'improved')
            lines = [r['fused_lines'] for r in result['results']]
            agreed = bool(lines) and len(lines[0]) == 1 and all(s == lines[0] for s in lines) and bool(lines[0][0].strip())
            route.update(status='accepted' if agreed else 'review',
                         recognition=str((folder/'improved/recognition.json').resolve()),
                         recognized_text=lines[0][0] if agreed else None,
                         configurations=lines)
            manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('AUTO_COMPLETE', json.dumps(dict(statuses=dict(Counter(r['status'] for r in routes))), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
