"""Group-specific evaluation; overlapping groups are never pooled as accuracy."""
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz

from evaluate_real_validation import alignment
from iterate_fusion import field_box,owned_by_field,lines

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output/cad_fusion_refined'
REF=ROOT/'ocr_refinement'


def main():
    OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
    base_path=ROOT.parent/'output/cad_fusion_iterations/automatic_results.json'
    base=json.loads(base_path.read_text(encoding='utf-8'))
    current_path=OUT/'combined_results.json';current=json.loads(current_path.read_text(encoding='utf-8'))
    legacy=json.loads((ROOT/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    page=fitz.open(current['source'])[0]
    fields=[]
    for group,directory in [('fixed_development',ROOT/'real_font_validation'),('previous_audit_now_regression',ROOT/'fusion_iterations')]:
        manifest=json.loads((directory/('regions.json' if group=='fixed_development' else 'audit_regions.json')).read_text(encoding='utf-8'))
        answers={r['id']:r for r in json.loads((directory/('answers.json' if group=='fixed_development' else 'audit_answers.json')).read_text(encoding='utf-8'))['answers']}
        for region in manifest['regions']:
            answer=answers[region['id']];box=field_box(region)
            fields.append(dict(group=group,id=region['id'],box_pdf=box,cap=region.get('reference_cap_height_pdf',box[3]-box[1]),
                reference=answer['reference'],ambiguities=answer.get('ambiguities',[]),kind=region.get('kind',answer.get('kind','mixed'))))
    base_records={r['id']:r for r in base['records']}
    for answer in json.loads((REF/'full_caption_answers.json').read_text(encoding='utf-8'))['answers']:
        record=base_records[answer['id']]
        fields.append(dict(group='expanded_captions',id=answer['id'],box_pdf=record['box_pdf'],cap=record['cap'],reference=answer['reference'],ambiguities=[],kind='caption'))
    for answer in json.loads((REF/'prefix_answers.json').read_text(encoding='utf-8'))['answers']:
        fields.append(dict(group='branch_prefixes',id=answer['id'],box_pdf=answer['box_pdf'],cap=answer['cap'],reference=answer['reference'],ambiguities=[],kind='prefix'))
    frame_answers={r['id']:r['reference'] for r in json.loads((REF/'frame_answers.json').read_text(encoding='utf-8'))['answers']}
    for record in current['frame_labels']:
        fields.append(dict(group='border_labels',id=record['id'],box_pdf=record['box_pdf'],cap=record['cap'],reference=frame_answers[record['id']],ambiguities=[],kind='border'))
    legacy_glyphs=[dict(text=g['text'],score=g['score'],box_pdf=g['box_pdf']) for g in legacy['detections']]
    methods={'original_pipeline':legacy_glyphs,'previous_iteration':base['detections'],'refined':current['detections']}
    rows=[]
    current_records={r['id']:r for r in current['records']}
    for field in fields:
        image=f"{field['group']}_{field['id']}.png"
        target=assets/image
        if not target.exists():
            b=fitz.Rect(field['box_pdf']);margin=min(2.,field['cap']*.15);scale=min(10.,55/field['cap'])
            page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=b+(-margin,-margin,margin,margin),alpha=False).save(target)
        for method,glyphs in methods.items():
            if field['group']=='expanded_captions' and method!='original_pipeline':
                record=(base_records if method=='previous_iteration' else current_records)[field['id']]
                prediction='\n'.join(record['fused_lines']);count=len(record['fused_glyphs'])
            else:
                selected=[g for g in glyphs if owned_by_field(g,field)]
                prediction='\n'.join(lines(selected));count=len(selected)
            rows.append({**field,'method':method,'prediction':prediction,'box_count':count,'image':image,
                'score':alignment(field['reference'],prediction,field['ambiguities'])})
    grouped=defaultdict(list)
    for row in rows:grouped[row['group'],row['method'],row['kind']].append(row)
    summaries=[]
    for (group,method,kind),items in grouped.items():
        characters=sum(r['score']['reference_characters'] for r in items)
        errors=sum(r['score']['edit_distance'] for r in items)
        summaries.append(dict(group=group,method=method,kind=kind,regions=len(items),exact=sum(r['score']['exact'] for r in items),characters=characters,errors=errors,cer=errors/characters if characters else None,false_boxes=sum(r['box_count'] for r in items) if kind=='drawing_negative' else None))
    report=dict(scope='Same-PDF regression and expanded development. Groups overlap and must not be pooled. Full-page recall still not established.',
        inference_llm=False,answers_used_for_inference_or_training=False,
        reference_scope='20 complete captions and 13 repeated branch-name prefixes visually reviewed after baseline inspection: expanded development, not blind holdout. Border references fixed before model evaluation; full stroke extent repaired after initial crop failure. Previous audit now used as regression.',
        notes=['Whitespace ignored, old explicit O/0 annotations retained in fixed regions.','Native PDF text is already editable and is reported separately.','Confidence is an OCR/model score, not a calibrated probability of correctness.','No expected grid sequence is used to correct border labels.','No final CAD replacement performed.'],
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base_path,current_path,REF/'full_caption_answers.json',REF/'prefix_answers.json',REF/'frame_answers.json']},
        summaries=summaries,rows=rows)
    (OUT/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['group','id','method','reference','prediction','character_errors','exact'])
        for r in rows:writer.writerow([r['group'],r['id'],r['method'],r['reference'],r['prediction'],r['score']['edit_distance'],r['score']['exact']])
    payload=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>YOLO + OCR 확장 검증</title><style>body{font:15px system-ui;background:#f4f6f8;color:#192635;margin:25px}h1{font-size:26px}p{line-height:1.7}select{padding:9px;margin:5px}table{width:100%;border-collapse:collapse;background:white;table-layout:fixed}td,th{padding:10px;border:1px solid #ccd5df;vertical-align:top;overflow-wrap:anywhere}img{max-width:100%;max-height:150px;object-fit:contain}pre{white-space:pre-wrap}.ok{background:#e3f4e8}.bad{background:#ffebeb}#summary{background:white;padding:14px;margin:12px 0}</style>
<h1>YOLO + 한글 OCR: 전체 문구와 짧은 표기 확장 검증</h1><p>문자별 CTC 위치로 한글 범위만 합치고, 기존에 읽은 단어를 보호합니다. 주변 선이 섞이지 않도록 벡터 획만 정규화해 재인식하는 입력도 비교했습니다.<br>
이 보고서는 동일 PDF에서의 회귀/개발 검사입니다. 전체 문구 20개와 반복된 분기명 표기 13개는 기존 결과를 본 뒤 추가했으므로 독립 시험이 아닙니다. 테두리 104개 정답은 모델 평가 전에 원본으로 고정했습니다.<br>
검사 그룹은 일부 겹칩니다. 점수를 모두 합쳐 전체 정확도라고 해석하면 안 됩니다. 공백은 제외하고 기존에 지정한 O/0 위치만 동등 평가합니다. 새 보고서에서도 남은 오류를 그대로 표시합니다.<br>
<a href="../cad_fusion_iterations/report.html">직전 실험 보고서</a> · <a href="recognized_text.csv">현재 문자·좌표 CSV</a> · <a href="recognized_lines.txt">현재 문구 출력</a></p>
<select id="group"><option value="expanded_captions">전체 문구 20개</option><option value="branch_prefixes">분기명 표기 13개</option><option value="border_labels">테두리 표기 104개</option><option value="fixed_development">기존 개발 24개</option><option value="previous_audit_now_regression">기존 추가 8개 (현재 회귀용)</option></select><select id="method"><option value="refined">개선 후</option><option value="previous_iteration">직전 자동 검색</option><option value="original_pipeline">최초 파이프라인</option></select><div id="summary"></div><table><thead><tr><th style="width:12%">영역</th><th style="width:43%">원본 / 검증 정답</th><th style="width:30%">기계 출력</th><th>차이</th></tr></thead><tbody id="rows"></tbody></table>
<script>const D=__DATA__,G=document.querySelector('#group'),M=document.querySelector('#method');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function show(){const a=D.rows.filter(r=>r.group===G.value&&r.method===M.value);const p=a.filter(r=>r.kind!=='drawing_negative'),n=p.reduce((s,r)=>s+r.score.reference_characters,0),e=p.reduce((s,r)=>s+r.score.edit_distance,0);document.querySelector('#summary').textContent=`문자열 일치 ${p.filter(r=>r.score.exact).length}/${p.length} · 문자 오류 ${e}/${n} · 도형 오검출 ${a.filter(r=>r.kind==='drawing_negative').reduce((s,r)=>s+r.box_count,0)}개`;document.querySelector('#rows').innerHTML=a.map(r=>`<tr><td>${esc(r.id)}<br>${esc(r.kind)}</td><td><pre>${esc(r.reference||'(문자 없음)')}</pre><img loading="lazy" src="assets/${r.image}"></td><td><pre>${esc(r.prediction||'(검출 없음)')}</pre></td><td class="${r.score.exact?'ok':'bad'}">오류 ${r.score.edit_distance}<br>누락 ${r.score.deletions}<br>치환 ${r.score.substitutions}<br>추가 ${r.score.insertions}</td></tr>`).join('')};G.onchange=show;M.onchange=show;show();</script></html>'''.replace('__DATA__',payload)
    (OUT/'report.html').write_text(html,encoding='utf-8')
    for name in ['full_caption_answers.json','prefix_answers.json','frame_answers.json']:
        shutil.copy2(REF/name,OUT/name)
    for row in summaries:print(json.dumps(row,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
