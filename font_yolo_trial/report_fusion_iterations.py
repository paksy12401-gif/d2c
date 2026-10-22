"""Evaluate saved outputs only; answers cannot influence inference or training."""
import hashlib
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

from evaluate_real_validation import alignment
from infer_real_validation import inside
from iterate_fusion import lines, owned_by_field

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'fusion_iterations'
OUT=ROOT.parent/'output/cad_fusion_iterations'


def evaluate_split(split):
    predictions=WORK/f'{split}_predictions.json'
    if not predictions.exists():return [],[]
    data=json.loads(predictions.read_text(encoding='utf-8'))
    if split=='development':
        answer_file=ROOT/'real_font_validation/answers.json'
        manifest=json.loads((ROOT/'real_font_validation/regions.json').read_text(encoding='utf-8'))
        kinds={r['id']:r['kind'] for r in manifest['regions']}
    else:
        answer_file=WORK/'audit_answers.json'
        kinds={r['id']:r['kind'] for r in json.loads(answer_file.read_text(encoding='utf-8'))['answers']}
    answers={r['id']:r for r in json.loads(answer_file.read_text(encoding='utf-8'))['answers']}
    fields={f['id']:f for f in data['fields']}
    rows=[]
    def add(identity,config,prediction,count,raw=None):
        answer=answers[identity]
        rows.append(dict(split=split,id=identity,kind=kinds[identity],config=config,
            reference=answer['reference'],prediction=prediction,raw_prediction=raw,
            box_count=count,score=alignment(answer['reference'],prediction,answer.get('ambiguities'))))
    old=json.loads((ROOT/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    for identity,field in fields.items():
        glyphs=[dict(text=g['text'],score=g['score'],box_pdf=g['box_pdf']) for g in old['detections'] if owned_by_field(g,field)]
        add(identity,'previous_pipeline','\n'.join(lines(glyphs)),len(glyphs))
    for record in data['records']:
        add(record['id'],record['config'],'\n'.join(record['fused_lines']),len(record['fused_glyphs']),'\n'.join(record['raw_lines']))
    automatic=OUT/'automatic_results.json'
    if automatic.exists():
        auto=json.loads(automatic.read_text(encoding='utf-8'))
        for identity,field in fields.items():
            glyphs=[g for g in auto['detections'] if owned_by_field(g,field)]
            add(identity,'automatic_selected','\n'.join(lines(glyphs)),len(glyphs))
    grouped=defaultdict(list)
    for row in rows:grouped[row['config'],row['kind']].append(row)
    summaries=[]
    for (config,kind),items in grouped.items():
        characters=sum(r['score']['reference_characters'] for r in items)
        errors=sum(r['score']['edit_distance'] for r in items)
        summaries.append(dict(split=split,config=config,kind=kind,regions=len(items),
            exact=sum(r['score']['exact'] for r in items),characters=characters,errors=errors,
            cer=errors/characters if characters else None,
            negative_false_boxes=sum(r['box_count'] for r in items) if kind=='drawing_negative' else None))
    return rows,summaries


def main():
    OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
    rows=[];summaries=[]
    for split in ['development','audit']:
        r,s=evaluate_split(split);rows+=r;summaries+=s
        for item in {r['id']:r for r in r}.values():
            source=(ROOT/'real_font_validation' if split=='development' else WORK)/f"{item['id']}_source.png"
            shutil.copy2(source,assets/f"{split}_{source.name}")
    report=dict(scope='Known-region diagnostics on one PDF. Development set repeatedly inspected; audit on same PDF, not new drawing/font. Full-page localization and CAD replacement not established.',
        inference_llm=False,reference_policy='User authorized visual review for evaluation only. Spaces ignored; explicitly annotated O/0 ambiguity allowed. No answers loaded in inference or training.',
        scoring_change='Field extent includes original detector bounds as well as measured cap band. This retains baseline underscores excluded by the earlier cap-only region selection. All previous-pipeline boxes reselected with the same field extent.',
        experiments=['Original models at 56px cap height','12-epoch 68-class synthetic fine-tune, 520 train scenes, no real evaluation data','Original-vector groups separated onto white canvases at 40/56/80px, existing YOLO; all detections retained','Word-level Korean OCR, digit/Hangul acceptance, geometric overlap fusion, native PDF text extraction'],
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*predictions.json')},
        summaries=summaries,rows=rows)
    (OUT/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in [WORK/'development_predictions.json',WORK/'audit_predictions.json',WORK/'audit_regions.json',WORK/'audit_answers.json',WORK/'audit_answers_initial.json',WORK/'audit_reference_review.json',WORK/'H01_fourth_source.png',WORK/'synthetic_regression.json',WORK/'selected_config.json']:
        if p.exists():shutil.copy2(p,OUT/p.name)
    payload=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>YOLO + OCR 반복 실험</title>
<style>body{font:15px system-ui;max-width:1550px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#192635}h1{font-size:26px}p{line-height:1.65}select{padding:8px;margin:10px}table{border-collapse:collapse;width:100%;background:white;table-layout:fixed}th,td{border:1px solid #ccd5df;padding:10px;vertical-align:top;overflow-wrap:anywhere}img{width:100%;max-height:200px;object-fit:contain}pre{white-space:pre-wrap}.ok{background:#e3f4e8}.bad{background:#ffebeb}#summary{padding:15px;background:white;margin:12px 0}</style>
<h1>YOLO + 한글 OCR 반복 실험</h1>
<p>인식·학습·문자열 보정에 LLM을 사용하지 않았습니다. 검증 정답은 사용자가 허용한 원본 육안 확인으로 작성했으며 학습/인식 코드에 전달하지 않았습니다.<br>
이 결과는 <b>알려진 영역을 읽는 실험</b>입니다. 전체 도면을 빠짐없이 찾았다는 의미가 아닙니다. 개발 영역과 같은 PDF에서 추가한 확인 영역을 구분합니다.<br>
이전 검사의 높이 범위가 밑줄을 제외하는 문제가 있어, 이번에는 원래 영역과 글자 높이 영역을 합친 동일 범위로 이전 검출과 새 검출을 비교했습니다. 경계에 일부만 걸친 짧은 기호는 제외합니다. 공백은 제외하고 지정된 O/0 위치만 동등 평가합니다.<br>
추가 확인 H01의 넷째 숫자는 원본 벡터 획을 재확인해 8에서 9로 정정했습니다. 초기 정답과 변경 기록·원본 획 이미지를 함께 보관했습니다. 모델 출력은 변경하지 않았습니다.<br>
자동 검색(automatic_selected)은 검증 위치를 주지 않고 표 밖의 선형 벡터 후보에서 읽은 결과입니다. 여기 표시하는 점수는 고정 검사 위치에서의 비교이며, 도면 전체 누락률이나 전체 오검출률은 아닙니다. 일부만 잘린 OCR 후보 6곳은 정답 문자열을 확정하지 못해 문자 점수에서 제외했으며 audit_regions.json에 남겼습니다.</p>
<select id="split"><option value="development">개발 24개 영역</option><option value="audit">추가 확인 8개 영역</option></select><select id="config"></select><select id="kind"><option value="all">전체</option><option value="outlined_latin">영문·숫자</option><option value="korean">한글</option><option value="mixed">한영 혼합</option><option value="drawing_negative">문자 없는 도형</option><option value="native_font_control">PDF 내장 문자</option></select>
<div id="summary"></div><table><thead><tr><th style="width:8%">영역</th><th style="width:38%">원본 / 정답</th><th style="width:34%">원시 출력 / 병합 출력</th><th>오류</th></tr></thead><tbody id="rows"></tbody></table>
<script>const D=__DATA__,S=document.querySelector('#split'),C=document.querySelector('#config'),K=document.querySelector('#kind');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const v of [...new Set(D.rows.map(r=>r.config))]){const o=document.createElement('option');o.value=v;o.textContent=v;C.append(o)}
if([...C.options].some(o=>o.value==='automatic_selected'))C.value='automatic_selected';
function show(){const rows=D.rows.filter(r=>r.split===S.value&&r.config===C.value&&(K.value==='all'||r.kind===K.value)),positive=rows.filter(r=>r.kind!=='drawing_negative');const n=positive.reduce((a,r)=>a+r.score.reference_characters,0),e=positive.reduce((a,r)=>a+r.score.edit_distance,0);document.querySelector('#summary').textContent=`문자열 일치 ${positive.filter(r=>r.score.exact).length}/${positive.length}, 문자 오류 ${e}/${n}, 도형 오검출 ${rows.filter(r=>r.kind==='drawing_negative').reduce((a,r)=>a+r.box_count,0)}개`;
document.querySelector('#rows').innerHTML=rows.map(r=>`<tr><td>${esc(r.id)}<br>${esc(r.kind)}</td><td><pre>${esc(r.reference||'(문자 없음)')}</pre><img loading="lazy" src="assets/${r.split}_${r.id}_source.png"></td><td>${r.raw_prediction!==null?`<details><summary>병합 전 YOLO</summary><pre>${esc(r.raw_prediction)}</pre></details>`:''}<pre>${esc(r.prediction||'(검출 없음)')}</pre></td><td class="${r.score.exact?'ok':'bad'}">오류 ${r.score.edit_distance}<br>누락 ${r.score.deletions}<br>치환 ${r.score.substitutions}<br>추가 ${r.score.insertions}</td></tr>`).join('')};S.onchange=show;C.onchange=show;K.onchange=show;show();</script></html>'''.replace('__DATA__',payload)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    for s in summaries:print(json.dumps(s,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
