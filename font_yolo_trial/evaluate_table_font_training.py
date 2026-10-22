"""Score server OCR and font adaptation on training-excluded source cells."""
import csv
import hashlib
import html
import json
import re

import cv2
import numpy as np

from evaluate_real_validation import alignment
from evaluate_table_cells import metrics
from refine_table_ocr import ROOT,BASE,compact
from train_table_font_verifier import patch

OUT=ROOT/'output/cad_table_font_training'


def main():
    frozen=json.loads((OUT/'split_freeze.json').read_text())
    for name,digest in frozen.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    new_freeze=json.loads((OUT/'new_check_freeze.json').read_text())
    assert hashlib.sha256((OUT/'new_check_reference.json').read_bytes()).hexdigest()==new_freeze['sha256']
    held=json.loads((OUT/'holdout.json').read_text(encoding='utf-8'))
    new=json.loads((OUT/'new_check_reference.json').read_text(encoding='utf-8'))['answers']
    assert len(held['answers'])==284 and len(new)==24
    answers={r['id']:{**r,'group':'heldout'} for r in held['answers']}
    assert not(set(answers)&{r['id'] for r in new})
    answers.update({r['id']:{**r,'group':'new_check'} for r in new})
    final=json.loads((OUT/'combined_predictions.json').read_text(encoding='utf-8'))
    font=json.loads((OUT/'font_predictions.json').read_text(encoding='utf-8'))
    server=json.loads((OUT/'server_predictions.json').read_text(encoding='utf-8'))
    previous=json.loads((ROOT/'output/cad_table_refinement/refined_predictions.json').read_text(encoding='utf-8'))
    assert not final['heldout_labels_loaded'] and not final['llm_inference']
    assert len(final['results'])==880 and len(server['results'])==1760
    stages={'previous':{r['id']:r for r in previous['results']},'font_only':{r['id']:r for r in font['results']},
        'server_normal':{r['id']:r for r in server['results'] if r['variant']=='normal'},
        'server_more_ink':{r['id']:r for r in server['results'] if r['variant']=='more_ink'},
        'combined':{r['id']:r for r in final['results']}}
    scored=[];summary=[]
    for stage,records in stages.items():
        for identity,answer in answers.items():
            row=records[identity]
            scored.append(dict(id=identity,stage=stage,table=answer['table'],group=answer['group'],
                reference=answer['reference'],text=row['text'],score=alignment(answer['reference'],row['text'])))
        for group in ['heldout','new_check','all_evaluated']:
            for table in ['left','right']:
                rows=[r for r in scored if r['stage']==stage and r['table']==table and (group=='all_evaluated' or r['group']==group)]
                if rows:summary.append(dict(stage=stage,group=group,table=table,**metrics(rows)))
    by_score={(r['id'],r['stage']):r for r in scored}
    regressions=[i for i in answers if by_score[i,'combined']['score']['edit_distance']>by_score[i,'previous']['score']['edit_distance']]
    errors=[r for r in scored if r['stage']=='combined' and not r['score']['exact']]
    manifest=json.loads((OUT/'training_manifest.json').read_text())
    training_hashes={r['pixel_sha256'] for r in manifest['records']}
    cells={r['id']:r for r in json.loads((BASE/'cells.json').read_text())['cells']}
    glyphs={r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    pixel_total=pixel_repeated=0
    for identity,answer in answers.items():
        if answer['table']!='left':continue
        field=glyphs[identity];text=compact(answer['reference'])
        assert len(text)==len(field['glyphs']),identity
        source=cv2.imdecode(np.fromfile(BASE/'clean'/(identity+'.png'),np.uint8),0)
        for char,glyph in zip(text,field['glyphs']):
            if char not in 'O0Il1':continue
            value=patch(source,cells[identity],glyph,field['cap']);pixel_total+=1
            pixel_repeated+=hashlib.sha256(value.tobytes()).hexdigest() in training_hashes
    result=dict(training_cells=5,excluded_training_and_duplicate_cells=len(held['excluded']),
        heldout_left=74,heldout_right=210,new_left_check=24,neural_training=True,
        summary=summary,scored_cells=scored,remaining_errors=errors,regressed_cells=regressions,
        glyph_pixel_overlap=dict(evaluated_ambiguous_glyphs=pixel_total,pixel_identical_to_training=pixel_repeated),
        limitation='One source PDF and one adapted table font. Whole training strings are excluded, but repeated font glyphs are expected and some patches are pixel-identical to training. This is same-font adaptation, not unseen-font or independent-drawing accuracy. I/l are one visual class; context chooses identity. Additional 24 cells are not used to retune the inference. A source-only reference transcription correction is documented separately.',
        frozen_split_hashes=frozen,weights_sha256=hashlib.sha256((OUT/'font_verifier.pt').read_bytes()).hexdigest(),
        llm_inference=False)
    (OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    final_by=stages['combined'];ordered=list(cells.values())
    crossed=json.loads((BASE/'evaluation.json').read_text())['crossed_out_left_geometric_rows']
    with (OUT/'cell_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','previous','font_only','combined','glyph_updates','context_updates','training_cell_or_duplicate','crossed_out_row_candidate'])
        excluded={r['id'] for r in held['excluded']}
        for c in ordered:
            r=final_by[c['id']]
            writer.writerow([r['id'],r['previous_text'],r['font_only_text'],r['text'],len(r['verifier_updates']),len(r['server_context_updates']),r['id'] in excluded,r['table']=='left' and r['row'] in crossed])
    for table,cols in [('left',5),('right',10)]:
        selected=[c for c in ordered if c['table']==table]
        with (OUT/f'{table}_trained.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            for start in range(0,len(selected),cols):writer.writerow([final_by[c['id']]['text'] for c in selected[start:start+cols]])
    esc=html.escape
    titles={'previous':'이전 결합 방식','font_only':'학습한 글자 검증 추가','server_normal':'서버 OCR 단독 · 기본','server_more_ink':'서버 OCR 단독 · 입력 변경','combined':'글자 검증 + 서버 문맥'}
    groups={'heldout':'학습 제외 기존 검증','new_check':'새로 확인한 24칸','all_evaluated':'검증 합계'}
    def summary_table(group):
        rows=[r for r in summary if r['group']==group]
        body=''.join(f'<tr><td>{titles[r["stage"]]}</td><td>{"왼쪽 표" if r["table"]=="left" else "오른쪽 표"}</td><td>{r["exact_cells"]}/{r["cells"]} ({r["exact_rate"]*100:.1f}%)</td><td>{r["cer"]*100:.2f}%</td></tr>' for r in rows)
        return f'<h2>{groups[group]}</h2><table><tr><th>방법</th><th>대상</th><th>칸 완전일치</th><th>문자 오류율 CER</th></tr>{body}</table>'
    source_rows=[]
    for c in ordered:
        identity=c['id'];r=final_by[identity];answer=answers.get(identity)
        status=('일치' if by_score[identity,'combined']['score']['exact'] else '오답') if answer else ('학습 또는 중복 제외' if identity in excluded else '정답 미검증')
        css='bad' if answer and not by_score[identity,'combined']['score']['exact'] else ''
        changes=esc(json.dumps(dict(glyph=r['verifier_updates'],context=r['server_context_updates']),ensure_ascii=False,indent=2))
        source_rows.append(f'<tr id="{identity}" class="{css}"><td>{identity}<br>{status}</td><td><img loading="lazy" src="../cad_table_benchmark/cells/{identity}.png"></td><td>{esc(answer["reference"]) if answer else "정답 대조 제외"}</td><td>{esc(r["previous_text"])}</td><td>{esc(r["text"])}<details><summary>변경 근거</summary><pre>{changes}</pre></details></td></tr>')
    doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>서버 OCR·표 글꼴 학습 검증</title><style>body{{font:16px system-ui;line-height:1.65;margin:28px;color:#203040}}table{{border-collapse:collapse;table-layout:fixed;width:100%;margin:18px 0}}td,th{{border:1px solid #ccd5df;padding:10px;vertical-align:top;overflow-wrap:anywhere}}img{{max-width:100%;max-height:130px;object-fit:contain}}.bad{{background:#fff0ed}}.note{{background:#eef5fc;padding:16px}}pre{{white-space:pre-wrap;font:13px monospace}}</style>
<h1>서버 OCR 비교와 실제 표 글꼴 학습</h1><p class="note">학습용 5칸과 동일 문구가 있는 1칸을 기존 290칸에서 제외했습니다. 왼쪽 74칸·오른쪽 210칸으로 비교한 뒤, 기존에 정답 대조하지 않은 왼쪽 24칸을 추가 확인했습니다. 전체 880칸의 정확도를 뜻하지 않습니다.</p>
<p>추가 학습은 기존 OCR 전체를 미세조정한 것이 아니라, 실제 표에서 잘라낸 글자 패치를 구별하는 작은 CNN 보조 모델입니다. 고유 패치 23개를 기울기·크기·위치·획 두께·노이즈로 변형해 균형 학습했습니다. 고정 600회 학습했고 검증 정답으로 학습 종료나 가중치를 선택하지 않았습니다. LLM은 인식·보정에 사용하지 않았습니다.</p>
<p>학습하는 모양은 O, 0, I/l 공통 세로획, 1의 네 그룹입니다. 이 표 글꼴에서는 I와 l의 획이 같으므로 이미지 모델만으로 둘을 구별하지 않습니다. 다섯 위치 변형에서 모양 판단이 일치하고 확률이 0.99 이상인 경우만 보조 판단에 사용했습니다.</p>
{summary_table('heldout')}{summary_table('new_check')}
<p>검증한 308칸 중 남은 오답은 {len(errors)}칸이고, 이전보다 문자 차이가 늘어난 칸은 {len(regressions)}칸입니다. 남은 TYCO의 C→0 혼동은 이번에 학습한 네 모양 그룹 밖의 오류입니다. 추가 24칸 결과를 본 뒤 모델이나 추론 규칙을 다시 조정하지 않았습니다.</p>
<p>서버급 모델 전체 교체는 성적이 낮아 채택하지 않았습니다. 학습한 모양 모델에 서버 OCR의 두 문맥 판독이 일치하는 경우만 결합했습니다. 대소문자 보정도 원본 벡터의 글자 높이가 해당 대소문자 모양과 맞을 때만 적용했습니다. 오른쪽 표는 이전 결과를 보존했습니다.</p>
<p><b>평가 범위:</b> 한 PDF·한 표 글꼴에 적응한 결과입니다. 평가 대상 혼동 글자 {pixel_total}개 중 {pixel_repeated}개는 정규화 후 학습 패치와 픽셀이 같습니다. 같은 글꼴이 모양을 재사용하기 때문이며 새 글꼴·다른 도면 성능을 입증하지 않습니다. 전체 문구 중복은 학습/검증에서 제외했지만 글꼴 모양 자체를 독립 분리한 시험은 아닙니다.</p>
<p>추가 정답의 카탈로그 번호는 원본 획을 분리해 12296694로 재확인했습니다. <a href="new_check_reference_review.json">정답 전사 수정 기록</a>과 <a href="new_check_catalog_source_parts.png">원본 획</a>을 보존했고, 이 수정은 모델 개선으로 계산하지 않았습니다.</p>
<p><a href="left_trained.csv">왼쪽 표 CSV</a> · <a href="right_trained.csv">오른쪽 표 CSV</a> · <a href="cell_review.csv">학습 제외·취소선·변경 기록 CSV</a> · <a href="evaluation.json">검증 기록</a> · <a href="training_report.json">학습 기록</a> · <a href="training_glyphs.png">학습 글자 모양</a> · <a href="font_verifier.pt">보조 모델 가중치</a> · <a href="../cad_table_refinement/report.html">이전 결과</a></p>
<p>CSV는 검토용 후보입니다. PC NO 27의 취소 상태는 별도 CSV에서 확인해야 합니다. 기존 DWG는 수정하지 않았습니다.</p>
<p><label><input id="onlyErrors" type="checkbox">검증한 오답만 보기</label></p><table id="cells"><tr><th style="width:10%">위치</th><th style="width:30%">원본</th><th>검증 정답</th><th>이전</th><th>학습·결합 후</th></tr>{''.join(source_rows)}</table><script>document.querySelector('#onlyErrors').onchange=e=>document.querySelectorAll('#cells tr[id]').forEach(r=>r.hidden=e.target.checked&&!r.classList.contains('bad'));</script></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    assert all((OUT/r).exists() for r in re.findall(r'(?:src|href)="([^"]+)"',doc))
    print(json.dumps(dict(summary=[r for r in summary if r['stage'] in ['previous','font_only','combined']],remaining_errors=errors,regressions=regressions,pixel_overlap=result['glyph_pixel_overlap']),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
