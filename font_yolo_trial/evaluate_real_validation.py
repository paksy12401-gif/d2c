"""Score frozen model outputs against separate, reviewable validation references."""
import csv
import hashlib
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

from infer_real_validation import as_lines, inside

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'real_font_validation'
OUT=ROOT.parent/'output/cad_font_validation_no_llm'


def canonical(text):
    return ''.join(text.split())


def alignment(reference, prediction, ambiguities=None):
    ref,pred=canonical(reference),canonical(prediction)
    allowed={a['index']:set(a['allowed']) for a in (ambiguities or [])}
    n,m=len(ref),len(pred)
    table=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):table[i][0]=i
    for j in range(m+1):table[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=0 if pred[j-1] in allowed.get(i-1,{ref[i-1]}) else 1
            table[i][j]=min(table[i-1][j]+1,table[i][j-1]+1,table[i-1][j-1]+cost)
    i,j=n,m;counts=dict(correct=0,substitutions=0,deletions=0,insertions=0)
    while i or j:
        cost=0 if i and j and pred[j-1] in allowed.get(i-1,{ref[i-1]}) else 1
        if i and j and table[i][j]==table[i-1][j-1]+cost:
            counts['substitutions' if cost else 'correct']+=1;i-=1;j-=1
        elif i and table[i][j]==table[i-1][j]+1:
            counts['deletions']+=1;i-=1
        else:counts['insertions']+=1;j-=1
    return dict(reference_characters=n,predicted_characters=m,edit_distance=table[n][m],
                exact=table[n][m]==0,character_error_rate=table[n][m]/n if n else None,**counts)


def supported_score(answer,prediction,supported):
    ref=canonical(answer['reference']);pred=canonical(prediction)
    indices=[i for i,c in enumerate(ref) if c in supported]
    mapped={old:new for new,old in enumerate(indices)}
    ambiguities=[{**a,'index':mapped[a['index']]} for a in answer.get('ambiguities',[]) if a['index'] in mapped]
    return alignment(''.join(ref[i] for i in indices),''.join(c for c in pred if c in supported),ambiguities)


def evaluate():
    OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
    manifest=json.loads((WORK/'regions.json').read_text(encoding='utf-8'))
    answers_file=WORK/'answers.json'
    answers={r['id']:r for r in json.loads(answers_file.read_text(encoding='utf-8'))['answers']}
    probes=json.loads((WORK/'probes_predictions.json').read_text(encoding='utf-8'))
    names={int(k):v for k,v in probes['names'].items()};supported=set(names.values())
    old=json.loads((ROOT/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    before=json.loads((ROOT/'cad_without_tables/ocr_progress.json').read_text(encoding='utf-8'))
    regions={r['id']:r for r in manifest['regions']};table=[]
    def add(region,config,prediction,box_count=None,image=None):
        answer=answers[region['id']]
        score=alignment(answer['reference'],prediction,answer.get('ambiguities'))
        trained=supported_score(answer,prediction,supported) if region['kind']!='korean' else score
        unsupported=sorted(set(canonical(answer['reference']))-supported) if region['kind']!='korean' else []
        table.append(dict(id=region['id'],kind=region['kind'],config=config,reference=answer['reference'],
                          prediction=prediction,box_count=box_count,unsupported_characters=unsupported,
                          ambiguous_positions=answer.get('ambiguities',[]),score=score,supported_score=trained,image=image))
    for region in regions.values():
        for config,items in [('old_before_dedup',before),('old_raw',old['detections'])]:
            selected=[r for r in items if inside(r['box_pdf'],region['box_pdf'],region['reference_cap_height_pdf']*.035)]
            lookup={};boxes=[]
            for item in selected:
                index=next((k for k,v in lookup.items() if v==item['text']),None)
                if index is None:index=len(lookup);lookup[index]=item['text']
                boxes.append([*item['box_pdf'],item['score'],index])
            add(region,config,'\n'.join(as_lines(boxes,lookup)),len(boxes))
    for record in probes['records']:
        if record['engine']=='korean_ocr' and record['view']=='context':continue
        region=regions[record['id']]
        config=f"crop_{record['view']}_h{record['target_height']}"
        image=f"{record['id']}_{record['view']}_{record['target_height']}_detected.png" if record['engine']=='yolo' else None
        add(region,config,'\n'.join(record['raw_lines']),record.get('selected_box_count'),image)
    scan_file=WORK/'scan_predictions.json'
    if scan_file.exists():
        scan=json.loads(scan_file.read_text(encoding='utf-8'))
        for record in scan['records']:
            for region in regions.values():
                if region['kind']=='korean':continue
                boxes=[b for b in record['boxes'] if inside(b,region['box_pdf'],region['reference_cap_height_pdf']*.035)]
                add(region,f"direct_scan_s{record['scale']}",'\n'.join(as_lines(boxes,names)),len(boxes))
    vector_file=WORK/'vector_predictions.json'
    if vector_file.exists():
        vector=json.loads(vector_file.read_text(encoding='utf-8'))
        for record in vector['records']:
            count=sum(bool(g.get('text')) for g in record['glyphs'])
            add(regions[record['id']],f"vector_glyph_h{record['target_height']}",'\n'.join(record['raw_lines']),count)
    summaries=[]
    grouped=defaultdict(list)
    for row in table:grouped[row['config'],row['kind']].append(row)
    for (config,kind),rows in grouped.items():
        n=sum(r['score']['reference_characters'] for r in rows)
        sn=sum(r['supported_score']['reference_characters'] for r in rows)
        summaries.append(dict(config=config,kind=kind,regions=len(rows),
            exact_regions=sum(r['score']['exact'] for r in rows),
            trained_characters_exact_regions=sum(r['supported_score']['exact'] for r in rows),
            reference_characters=n,character_errors=sum(r['score']['edit_distance'] for r in rows),
            character_error_rate=sum(r['score']['edit_distance'] for r in rows)/n if n else None,
            trained_character_error_rate=sum(r['supported_score']['edit_distance'] for r in rows)/sn if sn else None,
            negative_false_detections=sum(r['box_count'] or 0 for r in rows) if kind=='drawing_negative' else None))
    report=dict(scope='Diagnostic on 24 fixed real regions: 16 outlined Latin/number fields, 2 Korean lines, 4 native-font controls, 2 drawing negatives. Not independent deployment accuracy.',
        inference_llm=False,validation_reference='Assistant visual review authorized by user; native numeric controls from embedded PDF text.',
        reference_not_used_for_inference_or_training=True,training=False,word_correction=False,
        ambiguity_policy='Only explicitly listed O/0 positions allow both; this does not resolve their identity.',
        unsupported_policy='Full-field scores include missing underscore classes; trained-character-only scores are separately labeled.',
        source_sha256=manifest['source_sha256'],answers_sha256=hashlib.sha256(answers_file.read_bytes()).hexdigest(),
        vector_probe_limitations='Linear SVG paths only; heuristic character grouping, not guaranteed character boundaries. Underscores use geometry rules, not a learned class; isolated YOLO uses the 62-class letter model, so punctuation is not yet covered. Regions are controlled diagnostic crops, not complete CAD conversion.',
        frozen_prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'probes_predictions.json',scan_file,vector_file] if p.exists()},
        summaries=summaries,rows=table)
    (OUT/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['id','kind','config','reference','prediction','exact','character_errors','deletions','insertions','substitutions','trained_character_errors','unsupported_characters']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in table:
            writer.writerow({**{k:r[k] for k in ['id','kind','config','reference','prediction']},
                'exact':r['score']['exact'],'character_errors':r['score']['edit_distance'],
                **{k:r['score'][k] for k in ['deletions','insertions','substitutions']},
                'trained_character_errors':r['supported_score']['edit_distance'],
                'unsupported_characters':''.join(r['unsupported_characters'])})
    for region in regions.values():shutil.copy2(WORK/f"{region['id']}_source.png",assets)
    for row in table:
        if row['image'] and (WORK/row['image']).exists():shutil.copy2(WORK/row['image'],assets)
    for name in ['answers.json','regions.json','probes_predictions.json','scan_predictions.json','vector_predictions.json','reference_review_log.json','L15_zoom.png']:
        if (WORK/name).exists():shutil.copy2(WORK/name,OUT/name)
    html_report(report,regions)
    print(json.dumps(summaries,ensure_ascii=True),flush=True)


def html_report(report,regions):
    data=json.dumps({'rows':report['rows'],'summaries':report['summaries'],'regions':list(regions.values())},ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실제 CAD 글자 검증</title>
<style>body{font:15px system-ui;margin:28px;color:#182432;background:#f4f6f8}h1{font-size:25px}p{line-height:1.6}select{font:inherit;padding:8px;margin:0 12px 16px 0}table{border-collapse:collapse;background:white;width:100%;table-layout:fixed}th,td{border:1px solid #d7dee5;padding:10px;vertical-align:top;overflow-wrap:anywhere}th{background:#e8edf2}pre{white-space:pre-wrap;margin:0}.good{background:#e3f4e7}.bad{background:#ffebeb}img{max-width:100%;max-height:210px;object-fit:contain;background:white}small{color:#566}details{margin-top:10px}#summary{padding:14px;background:white;margin-bottom:15px}button{font:inherit}</style>
<h1>실제 도면 24개 영역 · YOLO / 한글 OCR 검증</h1>
<p>영문·숫자는 YOLO 직접 검색, 한글은 별도 OCR입니다. 인식·학습·출력 보정에 LLM을 사용하지 않았습니다.<br>
정답지는 사용자가 허용한 원본 육안 대조로 작성했으며, 숫자 대조군은 PDF 내장 문자를 사용했습니다. 정답지는 인식에 전달하지 않았습니다.<br>
밑줄(_)은 현재 YOLO 학습 클래스에 없습니다. O/0가 구별되지 않는 지정 위치는 둘 다 허용해 별도 표시했습니다. 24개 진단 결과를 전체 도면 정확도로 해석하면 안 됩니다.</p>
<p lang="en">Vector experiment: linear paths only, heuristic character splitting; underscore is a geometry rule. The 62-class isolated glyph model does not cover punctuation. These crops do not establish full drawing accuracy.</p><select id="config"></select><select id="kind"><option value="all">전체 24개</option><option value="outlined_latin">선으로 된 영문·숫자 16개</option><option value="korean">한글 2개</option><option value="native_font_control">기존 문자 / 다른 숫자체 대조군 4개</option><option value="drawing_negative">문자 없는 커넥터 2개</option></select>
<div id="summary"></div><table><thead><tr><th style="width:8%">영역</th><th style="width:32%">원본 / 검증용 정답</th><th style="width:34%">모델 원출력</th><th>차이</th></tr></thead><tbody id="rows"></tbody></table>
<script>const D=__DATA__;const C=document.querySelector('#config'),K=document.querySelector('#kind');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={old_raw:'이전 파이프라인 원출력',old_before_dedup:'이전 중복 제거 전',crop_isolated_h32:'단독 영역 / 32px',crop_isolated_h40:'단독 영역 / 40px',crop_isolated_h56:'단독 영역 / 56px',crop_context_h32:'주변 도형 포함 / 32px',crop_context_h40:'주변 도형 포함 / 40px',crop_context_h56:'주변 도형 포함 / 56px','direct_scan_s3.2':'OCR 영역 검출 없이 본문 직접 검색 / 배율 3.2','direct_scan_s4.0':'OCR 영역 검출 없이 본문 직접 검색 / 배율 4.0','direct_scan_s5.6':'OCR 영역 검출 없이 본문 직접 검색 / 배율 5.6'};
Object.assign(labels,{vector_glyph_h32:'Vector glyphs / 32px',vector_glyph_h40:'Vector glyphs / 40px',vector_glyph_h56:'Vector glyphs / 56px'});
[...new Set(D.rows.map(r=>r.config))].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=labels[v]||v;C.append(o)});C.value='vector_glyph_h40';
function show(){const a=D.rows.filter(r=>r.config===C.value&&(K.value==='all'||r.kind===K.value));const positive=a.filter(r=>r.kind!=='drawing_negative');let n=positive.reduce((s,r)=>s+r.score.reference_characters,0),e=positive.reduce((s,r)=>s+r.score.edit_distance,0);document.querySelector('#summary').textContent=`표시 ${a.length}개 · 문자열 일치 ${positive.filter(r=>r.score.exact).length}/${positive.length} · 문자 오류 ${e}/${n} (공백 제외, 지정 O/0 허용). 학습한 문자만의 점수는 아래에 별도 표시합니다.`;
document.querySelector('#rows').innerHTML=a.map(r=>`<tr><td>${esc(r.id)}<br><small>${esc(r.kind)}</small></td><td><pre>${esc(r.reference||'(문자 없음)')}</pre><img src="assets/${r.id}_source.png"><br><small>미학습 문자: ${esc(r.unsupported_characters.join(' ')||'없음')} ${r.ambiguous_positions.length?'· O/0 미확정 위치 있음':''}</small></td><td><pre>${esc(r.prediction||'(검출 없음)')}</pre>${r.image?`<details><summary>검출 박스 보기</summary><img loading="lazy" src="assets/${r.image}"></details>`:''}</td><td class="${r.score.exact?'good':'bad'}">${r.kind==='drawing_negative'?`오검출 후보 ${r.box_count||0}개`:`전체 문자 오류 ${r.score.edit_distance}<br>누락 ${r.score.deletions} / 치환 ${r.score.substitutions} / 추가 ${r.score.insertions}<br><small>학습한 문자만 비교: 오류 ${r.supported_score.edit_distance}</small>`}</td></tr>`).join('')};C.onchange=show;K.onchange=show;show();</script></html>'''.replace('__DATA__',data)
    (OUT/'report.html').write_text(page,encoding='utf-8')


if __name__=='__main__':evaluate()
