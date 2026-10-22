"""Retrospective regression evaluation, separate from answer-free inference."""
import hashlib
from html import escape
import json
from pathlib import Path

from evaluate_real_validation import alignment
from evaluate_table_cells import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'output/cad_table_confusion_fixed'
REFERENCES = ROOT/'output/cad_table_font_training'


def main():
    frozen = json.loads((REFERENCES/'split_freeze.json').read_text())
    assert hashlib.sha256((REFERENCES/'holdout.json').read_bytes()).hexdigest() == frozen['holdout.json']
    new_frozen = json.loads((REFERENCES/'new_check_freeze.json').read_text())
    assert hashlib.sha256((REFERENCES/'new_check_reference.json').read_bytes()).hexdigest() == new_frozen['sha256']
    refs = json.loads((REFERENCES/'holdout.json').read_text(encoding='utf-8'))['answers']
    refs += json.loads((REFERENCES/'new_check_reference.json').read_text(encoding='utf-8'))['answers']
    assert len(refs) == 308
    before = json.loads((REFERENCES/'combined_predictions.json').read_text(encoding='utf-8'))
    after = json.loads((OUT/'predictions.json').read_text(encoding='utf-8'))
    assert after['answers_loaded'] is False and after['llm_inference'] is False
    predictions = {'before':{r['id']:r for r in before['results']}, 'after':{r['id']:r for r in after['results']}}
    scored = {}
    summary = []
    for stage, rows in predictions.items():
        scored[stage] = [dict(id=a['id'], table=a['table'], reference=a['reference'], text=rows[a['id']]['text'],
                              score=alignment(a['reference'], rows[a['id']]['text'])) for a in refs]
        for table in ('left', 'right'):
            summary.append(dict(stage=stage, table=table, **metrics([r for r in scored[stage] if r['table']==table])))
    old_scores = {r['id']:r['score']['edit_distance'] for r in scored['before']}
    regressions = [r['id'] for r in scored['after'] if r['score']['edit_distance'] > old_scores[r['id']]]
    changed = [key for key in predictions['before'] if predictions['before'][key]['text'] != predictions['after'][key]['text']]
    result = dict(scope='Retrospective regression on the same 308 previously inspected cells. Not unseen-drawing accuracy.',
                  all_cells=880, evaluated_cells=len(refs), changed_cells=changed, regressions=regressions,
                  summary=summary, scored=scored)
    (OUT/'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ''.join('<tr><td>'+r['stage']+'</td><td>'+r['table']+'</td><td>'+str(r['exact_cells'])+'/'+str(r['cells'])+'</td><td>'+str(r['edits'])+'</td></tr>' for r in summary)
    changes = ''.join('<p><code>'+escape(c['before'])+'</code> → <code>'+escape(c['after'])+'</code></p>'+''.join(
        '<img style="width:128px;image-rendering:pixelated" src="'+escape(c['id']+f'_char{u["index"]}.png')+'">' for u in c['updates']) for c in after['changes'])
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>C/0 혼동 보정 검사</title><style>body{font:17px system-ui;line-height:1.7;max-width:1200px;margin:32px auto}td,th{padding:9px;border:1px solid #bbb}table{border-collapse:collapse}code{background:#eee;padding:3px}</style>
<h1>C/0 혼동 보정 검사</h1><p>YOLO의 C 검출, EasyOCR 두 입력의 C 후보, 원본의 오른쪽이 열린 획이 모두 일치할 때만 0/O를 C로 변경했습니다. 연속 숫자 구간은 보호했습니다. 단어 사전·정답 문자열·LLM 추론·추가 학습은 사용하지 않았습니다.</p>'''+changes+'''
<table><tr><th>단계</th><th>표</th><th>정확히 일치한 칸</th><th>문자 오류</th></tr>'''+rows+'''</table>
<p>전체 880칸 중 변경된 칸은 '''+str(len(changed))+'''개입니다. 기존 정답이 있는 308칸의 오류가 증가한 칸은 '''+str(len(regressions))+'''개입니다. 이미 확인했던 같은 도면의 회귀 검사이며, 새로운 도면에서의 정확도가 아닙니다. 나머지 572칸은 정답률을 계산하지 않았습니다.</p>
<p><a href="predictions.json">정답 미사용 추론 결과</a> · <a href="evaluation.json">별도 평가 결과</a></p></html>'''
    (OUT/'report.html').write_text(html, encoding='utf-8')
    print(json.dumps(dict(summary=summary, regressions=regressions, changed_cells=changed), ensure_ascii=True))


if __name__ == '__main__':
    main()
