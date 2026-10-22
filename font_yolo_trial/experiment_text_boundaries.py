"""Replay saved local OCR/YOLO outputs without recognition or ground-truth loading."""
from copy import deepcopy
from html import escape
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from boundary_consensus import easyocr_content_columns, spatial_consensus
from ctc_korean_fusion import choose_spans
from iterate_fusion import fuse_words, lines
from read_table_title_local import spans
from retry_table_title_local import crop_image
from vector_extra_symbols import supplement

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT/'output/cad_table_title_local'
OUT = ROOT/'output/cad_boundary_experiment'


def proposals_from_cache(initial, retry, correct_easy=False):
    field = initial['field']; proposals = []
    for record in initial['ocr_diagnostics']:
        proposals.extend(dict(p, variant=record['variant']) for p in record['spans'])
    for record in retry['diagnostics']:
        if not correct_easy or not record['variant'].startswith('easy_'):
            proposals.extend(dict(p, variant=record['variant']) for p in record['spans'])
            continue
        match = re.fullmatch(r'easy_word(\d+)_s([\d.]+)_x([\d.]+)', record['variant'])
        index, stroke, stretch = match.groups()
        image, origin, scale = crop_image(field, retry['windows'][int(index)],
                                           float(stroke), float(stretch))
        raw = deepcopy(record['raw'])
        raw['rec_text'][1][0] = easyocr_content_columns(raw['rec_text'][1][0], image.shape)
        proposals.extend(dict(p, variant=record['variant'])
                         for p in spans(raw, image.shape[1], origin, scale, field))
    return proposals


def fuse(initial, proposals, consensus=False):
    field = initial['field']
    candidates = spatial_consensus(proposals, field['cap']) if consensus else proposals
    review = [p for p in candidates if not p.get('boundary_stable', True)]
    usable = [p for p in candidates if p.get('boundary_stable', True)]
    selected = choose_spans(usable, field['cap'])
    results = []
    for run in initial['results']:
        glyphs = supplement(run['glyphs'], field, selected)
        fused, suppressed = fuse_words(glyphs, selected)
        results.append(dict(config=run['config'], fused_lines=lines(fused),
                            glyphs=glyphs, fused_glyphs=fused, suppressed=suppressed))
    return dict(korean_spans=selected, review_candidates=review, results=results)


def main():
    initial = json.loads((BASE/'recognition.json').read_text(encoding='utf-8'))
    retry = json.loads((BASE/'retry_recognition.json').read_text(encoding='utf-8'))
    old = proposals_from_cache(initial, retry)
    corrected = proposals_from_cache(initial, retry, True)
    stages = dict(baseline=fuse(initial, old), padding_only=fuse(initial, corrected),
                  padding_and_consensus=fuse(initial, corrected, True))
    # Cache replay must reproduce the recorded baseline before comparing methods.
    assert [r['fused_lines'] for r in stages['baseline']['results']] == [r['fused_lines'] for r in retry['results']]
    for name, stage in stages.items():
        for before, after in zip(stages['baseline']['results'], stage['results']):
            def key(g):
                return g['text'], tuple(g['box_pdf'])
            suppressed = {key(g) for g in after['suppressed']}
            after['previously_suppressed_now_retained'] = [g for g in before['suppressed']
                                                         if key(g) not in suppressed]
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(inference_llm=False, answers_loaded=False, training=False,
                  source='Saved local OCR/YOLO predictions, same title only',
                  scope='Boundary regression replay, not recognition accuracy or full-page evaluation',
                  input_sha256={name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                for name in ('recognition.json', 'retry_recognition.json')},
                  stages=stages)
    (OUT/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    field = initial['field']
    image = cv2.imdecode(np.fromfile(BASE/'vector_h80.png', np.uint8), 1)
    scale = 80/field['cap']; origin = [field['box'][i]-.35*field['cap'] for i in (0, 1)]
    for name, stage in stages.items():
        overlay = image.copy()
        for p in stage['korean_spans']:
            b = p['box_pdf']; a = (round((b[0]-origin[0])*scale), round((b[1]-origin[1])*scale))
            z = (round((b[2]-origin[0])*scale), round((b[3]-origin[1])*scale))
            cv2.rectangle(overlay, a, z, (0, 120, 220), 2)
        cv2.imencode('.png', overlay)[1].tofile(OUT/(name+'.png'))
        rows.append('<tr><td>'+name+'</td><td><img src="'+name+'.png"></td><td>'+
                    '<br>'.join(escape(r['config']+': '+' / '.join(r['fused_lines'])) for r in stage['results'])+'</td></tr>')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>한글·영문 경계 실험</title>
<style>body{font:16px system-ui;margin:24px;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:10px}img{max-width:100%;min-width:500px}td:nth-child(2){width:65%}</style>
<h1>한글·영문 경계 실험</h1><p>저장된 로컬 OCR·YOLO 결과를 다시 결합한 비교입니다. LLM 추론·정답 문자열 입력·추가 학습은 사용하지 않았습니다. 주황색은 선택된 한글 영역입니다.</p>
<p>baseline: 기존 선택 / padding_only: EasyOCR 내부 오른쪽 여백 보정 / padding_and_consensus: 여백 보정 + 입력 계열별 위치 중앙값. 마지막 단계는 서로 다른 두 입력 계열 이상에서 경계가 모이지 않으면 자동 결합에서 보류합니다.</p>
<p>동일 제목의 경계 회귀 검사이며, 한글 정답률이나 전체 도면 성능 측정은 아닙니다. 원래 인식의 오자·누락·기호 오류는 남을 수 있습니다. 기존 DWG는 변경하지 않았습니다.</p>
<table><tr><th>방법</th><th>한글 영역</th><th>자동 결합 문자열</th></tr>'''+''.join(rows)+'</table><p><a href="results.json">원시 결과</a></p></html>'
    (OUT/'report.html').write_text(html, encoding='utf-8')
    print(json.dumps({name: dict(lines=[r['fused_lines'] for r in s['results']],
                                review=len(s['review_candidates']),
                                boxes=[dict(text=w['text'], box=w['box_pdf']) for w in s['korean_spans']])
                      for name, s in stages.items()}, ensure_ascii=True, indent=1))


if __name__ == '__main__':
    main()
