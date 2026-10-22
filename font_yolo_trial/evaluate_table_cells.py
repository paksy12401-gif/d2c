"""Score frozen cell predictions; references are confined to this evaluation file."""
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_real_validation import alignment, canonical

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output/cad_table_benchmark'


def metrics(records):
    n = len(records)
    characters = sum(r['score']['reference_characters'] for r in records)
    errors = sum(r['score']['edit_distance'] for r in records)
    return dict(cells=n, exact_cells=sum(r['score']['exact'] for r in records),
                exact_rate=sum(r['score']['exact'] for r in records) / n if n else None,
                reference_characters=characters, edits=errors, cer=errors / characters if characters else None,
                substitutions=sum(r['score']['substitutions'] for r in records),
                deletions=sum(r['score']['deletions'] for r in records),
                insertions=sum(r['score']['insertions'] for r in records))


def main():
    ref_path = OUT / 'reference.json'
    frozen = json.loads((OUT / 'reference_freeze.json').read_text())
    assert hashlib.sha256(ref_path.read_bytes()).hexdigest() == frozen['sha256']
    reference = json.loads(ref_path.read_text(encoding='utf-8'))
    answers = {f'{t}_r{int(row):02d}_c{col}': text for t in ['left', 'right']
               for row, cells in reference[t].items() for col, text in enumerate(cells)}
    selection = json.loads((OUT / 'evaluation_selection.json').read_text())
    assert list(answers) == selection['ids'] and len(answers) == 150
    data = json.loads((OUT / 'predictions.json').read_text(encoding='utf-8'))
    assert data['llm_inference'] is False and data['answer_sheet_used'] is False
    assert len(data['results']) == 1760
    # Guard against accidentally forgiving code-character mistakes.
    assert alignment('O', '0')['edit_distance'] == 1
    assert alignment('A_B', 'AB')['edit_distance'] == 1
    assert alignment('A', 'a')['edit_distance'] == 1
    assert alignment('A B', 'AB')['edit_distance'] == 0
    scored = [{**r, 'reference': answers[r['id']], 'ocr_score': r['score'], 'score': alignment(answers[r['id']], r['text'])}
              for r in data['results'] if r['id'] in answers]
    groups = defaultdict(list)
    for r in scored:
        groups[r['table'], r['variant']].append(r)
    assert all(len(groups[t, v]) == (40 if t == 'left' else 110)
               for t in ['left', 'right'] for v in ['raw_cell', 'clean_cell'])
    results = []
    for (table, variant), rows in groups.items():
        columns = {str(c): metrics([r for r in rows if r['column'] == c])
                   for c in range(5 if table == 'left' else 10)}
        # Report repeated strings separately rather than treating repetition as independent evidence.
        unique = {}
        for r in rows: unique.setdefault(canonical(r['reference']), r)
        results.append(dict(table=table, variant=variant, **metrics(rows), columns=columns,
                            distinct_reference_strings=len(unique), distinct_first_cell_metrics=metrics(list(unique.values()))))
    report = dict(source_sha256=data['source_sha256'], reference_sha256=frozen['sha256'],
                  all_cells=dict(left=160, right=720), evaluation_cells=dict(left=40, right=110),
                  normalization='Whitespace only. Case, punctuation and O/0 retained.',
                  limitation=reference['selection'] + ' ' + reference['policy'],
                  llm_inference=False, automatic_reference_correction=False,
                  metrics=results, scored_cells=scored)
    yolo_file = OUT / 'yolo_predictions.json'
    yolo_note = ''
    if yolo_file.exists():
        yolo = json.loads(yolo_file.read_text(encoding='utf-8'))
        assert not yolo['llm_inference'] and not yolo['answer_sheet_used']
        yolo_scored = [{**r, 'reference': answers[r['id']], 'score': alignment(answers[r['id']], r['text'])}
                       for r in yolo['results'] if r['id'] in answers]
        assert len(yolo_scored) == 40
        yolo_metrics = metrics(yolo_scored)
        report['existing_yolo_probe'] = dict(**yolo_metrics, weights_sha256=yolo['weights_sha256'],
            adopted=False, scored_cells=yolo_scored,
            note='Probe combines connected vector-stroke grouping, existing 62-class model and geometric punctuation. It measures this combination, not the model in isolation. No retraining.')
        yolo_note = f'<p>기존 62종 YOLO 문자 모델도 왼쪽 표에 별도 시험했습니다. 연결된 획을 묶어 분리한 뒤 읽었지만 표본 40칸 중 {yolo_metrics["exact_cells"]}칸 ({100*yolo_metrics["exact_rate"]:.1f}%)만 일치해 채택하지 않았습니다. 학습을 추가한 실험은 아니며, 글꼴 차이와 획 분할의 영향을 함께 받는 결과입니다. 상세 결과는 검사 기록에 포함했습니다.</p>'
    (OUT / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    predictions = {(r['id'], r['variant']): r for r in data['results']}
    cells = json.loads((OUT / 'cells.json').read_text())['cells']
    grid = json.loads((OUT / 'cells.json').read_text())['left']
    original = cv2.imdecode(np.fromfile(OUT / 'left_source.png', np.uint8), 1)
    red = (original[:, :, 2] > 180) & (original[:, :, 1] < 130) & (original[:, :, 0] < 130)
    crossed = [row for row in range(32) if red[grid['ys'][row]+2:grid['ys'][row+1]-1,
               grid['xs'][0]+2:grid['xs'][-1]-1].mean(axis=1).max() > .7]
    report['crossed_out_left_geometric_rows'] = crossed
    (OUT / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'cell_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['table','geometric_row','column','text','crossed_out_row_candidate','reference_available'])
        for cell in cells:
            writer.writerow([cell['table'],cell['row'],cell['column'],predictions[cell['id'],'clean_cell']['text'],
                cell['table']=='left' and cell['row'] in crossed, cell['id'] in answers])
    for table in ['left', 'right']:
        for variant in ['raw_cell', 'clean_cell']:
            with (OUT / f'{table}_{variant}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                table_cells = [c for c in cells if c['table'] == table]
                cols = 5 if table == 'left' else 10
                for start in range(0, len(table_cells), cols):
                    writer.writerow([predictions[c['id'], variant]['text'] for c in table_cells[start:start + cols]])
    esc = html.escape
    table_names = {'left':'왼쪽 부자재 LIST', 'right':'오른쪽 회로 표'}
    method_names = {'raw_cell':'원본 칸 OCR', 'clean_cell':'입력 정리 후 OCR'}
    summary = ''.join(f'<tr><td>{table_names[r["table"]]}</td><td>{method_names[r["variant"]]}</td><td>{r["exact_cells"]}/{r["cells"]} ({100*r["exact_rate"]:.1f}%)</td><td>{100*r["cer"]:.2f}% ({r["edits"]}/{r["reference_characters"]})</td></tr>' for r in sorted(results, key=lambda r:(r['table'],r['variant'] != 'raw_cell')))
    rows = []
    scores = {(r['id'], r['variant']): r for r in scored}
    for c in cells:
        identity = c['id']
        raw = predictions[identity, 'raw_cell']
        clean = predictions[identity, 'clean_cell']
        ref = answers.get(identity)
        status = ('일치' if scores[identity, 'clean_cell']['score']['exact'] else '문자 차이 있음') if ref is not None else '미검증'
        if c['table'] == 'left' and c['row'] in crossed: status += ' · 취소선 행'
        cls = 'bad' if ref is not None and not scores[identity, 'clean_cell']['score']['exact'] else ''
        rows.append(f'<tr class="{cls}" id="{identity}"><td>{identity}<br>{status}</td><td><img loading="lazy" src="cells/{identity}.png"><details><summary>정리한 입력</summary><img loading="lazy" src="clean/{identity}.png"></details></td><td>{esc(ref) if ref is not None else "정답지 미작성"}</td><td>{esc(raw["text"])}</td><td>{esc(clean["text"])}</td></tr>')
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>표 칸별 OCR 실험</title><style>body{{font:16px system-ui;line-height:1.65;margin:28px;color:#203040}}table{{border-collapse:collapse;width:100%;table-layout:fixed;margin:20px 0}}td,th{{border:1px solid #cbd3dc;padding:10px;vertical-align:top;overflow-wrap:anywhere}}img{{max-width:100%;max-height:120px;object-fit:contain}}.bad{{background:#fff0ed}}.note{{background:#eef5fc;padding:16px}}</style>
<h1>표 칸별 OCR 실험</h1><p class="note">LLM 없이 로컬 PaddleOCR로 왼쪽 160칸과 오른쪽 720칸을 읽었습니다. 실제 정답 대조는 사전에 선택한 왼쪽 40칸·오른쪽 110칸입니다. 전체 880칸 정확도를 뜻하지 않습니다.</p>
<p>raw_cell: 원본 칸을 잘라 여백을 넣고 OCR. clean_cell: 왼쪽은 원본 벡터 획만 검정색으로 다시 그려 표선·취소선을 제외하고, 오른쪽은 이미지 글자 영역을 잘라 글자 높이와 여백을 맞춘 뒤 OCR. 정답 문자열로 인식 결과를 보정하지 않았습니다. 공백만 제외하고 대소문자·기호·O/0 차이는 오류로 계산했습니다.</p>
<p>반복 칸은 입력 픽셀 해시가 같을 때만 인식을 재사용했습니다. 셀 완전일치율은 한 글자라도 다르면 해당 칸을 오답 처리합니다. CER은 치환+삭제+추가 글자 수를 정답 글자 수로 나눈 값이며 낮을수록 좋습니다. 표 구조 자체는 셀 분리 수를 확인했으며 학습 모델의 구조 인식 정확도 측정은 아닙니다.</p>
<table><tr><th>표</th><th>방법</th><th>칸 완전일치</th><th>문자 오류율 CER</th></tr>{summary}</table>
{yolo_note}
<p>표본은 동일 PDF에서 선택한 행이며 무작위 표본·다른 도면의 독립 시험이 아닙니다. 일부 코드 O/0의 의미는 윤곽만으로 보장할 수 없습니다. 반복 문자열의 영향과 열별 결과는 <a href="evaluation.json">검사 기록</a>에서 따로 확인할 수 있습니다.</p>
<p><a href="left_clean_cell.csv">왼쪽 CSV</a> · <a href="right_clean_cell.csv">오른쪽 CSV</a> · <a href="left_reference_sample.png">왼쪽 원본 표본</a> · <a href="right_reference_sample.png">오른쪽 원본 표본</a></p>
<p>CSV는 전체 칸의 인식 후보이며 최종 승인 데이터가 아닙니다. 왼쪽 PC NO 27 행에는 빨간 취소선이 있습니다. 획만 분리하면 취소선 없이 글자가 읽히므로, 취소선 여부를 별도 판별해 <a href="cell_review.csv">칸별 검토 CSV</a>와 해당 행에 표시했습니다. 이 행은 이번 150칸 정답 대조 표본에는 포함되지 않았습니다.</p>
<p><label><input type="checkbox" id="onlyErrors"> 검증한 오답만 보기</label></p>
<table id="cells"><tr><th style="width:11%">위치</th><th style="width:31%">원본 칸</th><th>정답</th><th>원본 칸 OCR</th><th>입력 정리 후 OCR</th></tr>{''.join(rows)}</table>
<script>document.querySelector('#onlyErrors').onchange=e=>document.querySelectorAll('#cells tr[id]').forEach(r=>r.hidden=e.target.checked&&!r.classList.contains('bad'));</script></html>'''
    (OUT / 'report.html').write_text(document, encoding='utf-8')
    print(json.dumps(results, ensure_ascii=True), flush=True)


if __name__ == '__main__': main()
