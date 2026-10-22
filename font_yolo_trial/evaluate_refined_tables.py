"""Evaluate table refinements against frozen references, without changing inference."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re

from evaluate_real_validation import alignment
from evaluate_table_cells import metrics
from refine_table_ocr import BASE, OUT, ROOT, compact


def main():
    sources=[('original',BASE/'reference.json',BASE/'reference_freeze.json'),
             ('additional',OUT/'additional_reference.json',OUT/'additional_reference_freeze.json')]
    answers={};hashes={}
    for group,file,freeze in sources:
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest==json.loads(freeze.read_text())['sha256']
        hashes[group]=digest
        ref=json.loads(file.read_text(encoding='utf-8'))
        for table in ['left','right']:
            for row,values in ref[table].items():
                for col,value in enumerate(values):
                    identity=f'{table}_r{int(row):02d}_c{col}'
                    assert identity not in answers
                    answers[identity]=dict(group=group,text=value)
    assert len(answers)==290
    prediction=OUT/'refined_predictions.json'
    data=json.loads(prediction.read_text(encoding='utf-8'))
    assert not data['llm_inference'] and not data['answers_loaded'] and len(data['results'])==880
    scored=[]
    for r in data['results']:
        if r['id'] not in answers:continue
        answer=answers[r['id']]
        scored.append(dict(**r,group=answer['group'],reference=answer['text'],
            score=alignment(answer['text'],r['text']),baseline_score=alignment(answer['text'],r['baseline'])))
    summaries=[]
    for group in ['original','additional','combined']:
        for table in ['left','right']:
            rows=[r for r in scored if r['table']==table and (group=='combined' or r['group']==group)]
            previous=[{**r,'score':r['baseline_score']} for r in rows]
            summaries.append(dict(group=group,table=table,before=metrics(previous),after=metrics(rows)))
    regressions=[r for r in scored if r['score']['edit_distance']>r['baseline_score']['edit_distance']]
    improvements=[r for r in scored if r['score']['edit_distance']<r['baseline_score']['edit_distance']]
    lost_correct=[r for r in regressions if r['baseline_score']['exact']]
    result=dict(reference_hashes=hashes,predictions_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
        source_sha256=data['source_sha256'],llm_inference=False,reference_used_in_inference=False,
        evaluated_cells=290,summary=summaries,improved_cells=len(improvements),regressed_cells=len(regressions),
        previously_correct_now_wrong=len(lost_correct),regressions=regressions,scored_cells=scored,
        limitation='Same-PDF exploratory comparisons. Original 150 cells are development data; additional 140 source-transcribed cells were selected before inspecting the new OCR. Repeated strings and shared source drawing prevent an independent generalization claim. Unknown color codes are flagged, not blindly forced. No neural model fine-tuning was performed.')
    (OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    records={r['id']:r for r in data['results']}
    cells=json.loads((BASE/'cells.json').read_text())['cells']
    crossed=json.loads((BASE/'evaluation.json').read_text())['crossed_out_left_geometric_rows']
    with (OUT/'cell_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['table','geometric_row','column','baseline','refined','reason','review_flags','crossed_out_row_candidate'])
        for cell in cells:
            r=records[cell['id']]
            writer.writerow([r['table'],r['row'],r['column'],r['baseline'],r['text'],r['reason'],';'.join(r['review_flags']),r['table']=='left' and r['row'] in crossed])
    for table,ncols in [('left',5),('right',10)]:
        selected=[c for c in cells if c['table']==table]
        with (OUT/f'{table}_refined.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            for i in range(0,len(selected),ncols):writer.writerow([records[c['id']]['text'] for c in selected[i:i+ncols]])
    esc=html.escape;tn={'left':'왼쪽 부자재 LIST','right':'오른쪽 회로 표'}
    gn={'original':'기존 150칸','additional':'추가 140칸','combined':'합계 290칸'}
    summary=''.join(f'<tr><td>{gn[r["group"]]}</td><td>{tn[r["table"]]}</td><td>{r["before"]["exact_cells"]}/{r["before"]["cells"]} ({r["before"]["exact_rate"]*100:.1f}%)</td><td>{r["after"]["exact_cells"]}/{r["after"]["cells"]} ({r["after"]["exact_rate"]*100:.1f}%)</td><td>{r["before"]["cer"]*100:.2f}% → {r["after"]["cer"]*100:.2f}%</td></tr>' for r in summaries)
    scored_by={r['id']:r for r in scored};rows=[]
    for cell in cells:
        r=records[cell['id']];s=scored_by.get(cell['id'])
        status=('정답 일치' if s['score']['exact'] else '오답') if s else '정답 미검증'
        classes=[]
        if s and not s['score']['exact']:classes.append('bad')
        if compact(r['text'])!=compact(r['baseline']):classes.append('changed')
        if r['table']=='left' and r['row'] in crossed:status+=' · 취소선 행'
        vote='<details><summary>바꾼 글자의 판독 근거</summary><pre>'+esc(json.dumps(r['character_updates'],ensure_ascii=False,indent=2))+'</pre></details>' if r['character_updates'] else ''
        rows.append(f'<tr id="{r["id"]}" class="{" ".join(classes)}"><td>{r["id"]}<br>{status}</td><td><img loading="lazy" src="../cad_table_benchmark/cells/{r["id"]}.png"></td><td>{esc(s["reference"]) if s else "정답지 미작성"}</td><td>{esc(r["baseline"])}</td><td>{esc(r["text"])}</td><td>{esc(r["reason"])}<br>{esc("; ".join(r["review_flags"]))}{vote}</td></tr>')
    doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>표 OCR 개선 검증</title><style>body{{font:16px system-ui;line-height:1.65;margin:28px;color:#203040}}table{{border-collapse:collapse;width:100%;table-layout:fixed;margin:20px 0}}td,th{{border:1px solid #cbd3dc;padding:10px;vertical-align:top;overflow-wrap:anywhere}}img{{max-width:100%;max-height:120px;object-fit:contain}}.bad{{background:#fff0ed}}.note{{background:#eef5fc;padding:16px}}pre{{white-space:pre-wrap;font:13px monospace}}select{{padding:8px}}</style>
<h1>표 OCR 개선 검증</h1><p class="note">전체 880칸의 인식 후보를 만들고, 기존 150칸과 추가 140칸을 정답 대조했습니다. LLM과 검증 정답은 인식·보정에 사용하지 않았습니다. 기존 DWG와 이전 표 실험 결과를 보존했습니다.</p>
<table><tr><th>검사 범위</th><th>표</th><th>이전 입력 정리 OCR</th><th>개선 후</th><th>문자 오류율 CER</th></tr>{summary}</table>
<p>문자 차이가 줄어든 칸 {len(improvements)}개, 늘어난 칸 {len(regressions)}개, 기존 정답이 오답으로 바뀐 칸 {len(lost_correct)}개입니다. 공백만 제외하며 대소문자·기호·O/0 차이는 그대로 오류로 계산했습니다. 검토 표시가 있는 칸도 정확도 계산에서 제외하지 않았습니다.</p>
<p><b>적용한 방법:</b> 숫자 열에 허용 문자를 제한하고 PIN ID2의 S/J-1·S/J-2 같은 혼합 표기는 보존했습니다. 영문 EasyOCR 모델을 따로 비교했고, PaddleOCR의 두 입력과 부분 단어 재판독을 합쳐 글자별 후보를 모았습니다. 적어도 세 판독 근거가 지지하는 헷갈리는 글자만 바꾸고, 이미 이어진 숫자 구간은 영문으로 바꾸지 않았습니다. 부분 판독은 정답 문자열 없이 OCR 위치를 이용해 잘랐습니다.</p>
<p>오른쪽 색상은 두 기존 입력에서 일치한 고신뢰 토큰을 후보로 모았습니다. 새로운 색상을 무조건 틀렸다고 간주하지 않으며, 재판독과 실제 CTC 대체 후보가 지지하는 경우에만 바꿨습니다. 품명은 도면 본문에서 기존 YOLO가 읽은 동일 번호의 코드와 교차 확인했고, 번호가 다르거나 후보가 충돌하면 보정하지 않았습니다. 이는 같은 PDF의 다른 위치를 비교한 것으로 독립 정답은 아닙니다.</p>
<p>영문 모델 단독 교체는 성적이 낮아 채택하지 않았습니다. 신경망 추가 학습은 하지 않았습니다. 단계별 원시 결과와 글자 후보·출처는 JSON에 보존했습니다. 이번 표본은 같은 PDF에서 선정했으므로 다른 도면의 정확도를 보장하지 않습니다. 특히 반복되는 코드와 일부 O/0의 의미에는 원본 윤곽만으로 판단하기 어려운 한계가 있습니다.</p>
<p><a href="left_refined.csv">왼쪽 표 CSV</a> · <a href="right_refined.csv">오른쪽 표 CSV</a> · <a href="cell_review.csv">취소선·검토 상태 포함 CSV</a> · <a href="evaluation.json">정답 대조 기록</a> · <a href="refined_predictions.json">인식 및 보정 근거</a> · <a href="../cad_table_benchmark/report.html">이전 실험</a></p>
<p>CSV는 검토용 후보입니다. 왼쪽 PC NO 27 취소선 행의 글자도 읽히므로 실제 반영 시 취소 상태를 함께 사용해야 합니다.</p>
<label>표시 <select id="filter"><option value="all">전체 880칸</option><option value="bad">검증한 오답</option><option value="changed">바뀐 칸</option></select></label>
<table id="cells"><tr><th style="width:9%">위치</th><th style="width:25%">원본</th><th>검증 정답</th><th>이전</th><th>개선 후</th><th>근거 / 검토 사항</th></tr>{''.join(rows)}</table>
<script>document.querySelector('#filter').onchange=e=>document.querySelectorAll('#cells tr[id]').forEach(r=>r.hidden=e.target.value!=='all'&&!r.classList.contains(e.target.value));</script></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    links=re.findall(r'(?:src|href)="([^"]+)"',doc)
    assert all((OUT/r).exists() for r in links)
    print(json.dumps(dict(summary=summaries,improvements=len(improvements),regressions=len(regressions),previously_correct_now_wrong=len(lost_correct)),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
