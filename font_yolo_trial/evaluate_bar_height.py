"""Evaluate the longer-| assumption, separately from equal-height old | labels.

Uses source templates and deterministic diagnostics, not independent drawing accuracy.
"""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_preserved_reader import match
from prepare import extract_glyphs,paste_glyph,resize_glyph,read_image,write_image
from train_vertical_bar import ROOT,OLD,NAMES
from vertical_bar_reader import VerticalBarReader,merge_bars

OUT=ROOT.parent/'output/cad_bar_height_rule'
RATIO=1.15


def write_report(report):
    rows=[]
    for name,s in report['diagnostics']['summary'].items():
        rows.append(f'<tr><td>{name}</td><td>{s["tp"]}/60</td><td>{s["fp"]}</td><td>{s["fn"]}</td></tr>')
    table_path=OUT/'tables/analysis.json'
    tables=json.loads(table_path.read_text(encoding='utf-8')) if table_path.exists() else None
    sections=[]
    if tables:
        sections.append(f'<p>표 880칸의 YOLO 후보 검사: 길이 조건 후 | 후보 {tables["accepted_bars"]}개 / {tables["changed_cells"]}칸. 기존 정답지 308칸에서는 새 | 후보가 추가되지 않았습니다. 이 수치는 표 전체 OCR의 정확도가 아닙니다.</p>')
    correction_path=OUT/'vector_corrections/predictions.json'
    if correction_path.exists():
        corrections=json.loads(correction_path.read_text(encoding='utf-8'))
        sections.append('<h2>원본 선 좌표로 긴 세로줄 확인</h2><p>같은 필드의 일반 문자 높이 10.20, 긴 세로줄 13.99: 약 1.37배. 문자별 위치가 일대일로 대응하고 세로획이 끊김 없이 이어진 경우만 바꿨습니다. 세 셀의 I 7개를 |로 수정했습니다.</p>')
        for change in corrections['changes']:
            sections.append('<p>'+escape(change['id'])+'</p><pre>'+escape(change['before'])+'\n→ '+escape(change['after'])+'</pre>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>| 길이 기준 검사</title>
<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:10px}pre{white-space:pre-wrap}img{max-width:100%}</style>
<h1>|가 더 길다는 조건 적용</h1><p>기존 학습 가중치는 유지하고 결합 단계에 길이 판정을 추가했습니다. 같은 줄의 기준 글자보다 원본 세로획이 15% 이상 길어야 |로 채택합니다. 박스 길이만 쓰지 않고 실제 잉크 또는 PDF 선 좌표를 확인합니다. 기준 글자 두 개를 확보하지 못하면 보류합니다.</p>
<h2>길이가 다른 I와 |를 넣은 진단 60장</h2><table><tr><th>조건</th><th>긴 | 검출</th><th>| 오탐</th><th>긴 | 누락</th></tr>'''+''.join(rows)+'''</table>
<p>학습과 같은 글꼴 템플릿 및 직접 만든 세로획을 이용한 합성 진단입니다. 다른 실제 도면의 정확도를 뜻하지 않습니다. |를 I로 잘못 추가하는 것과 별도로, I 자체가 다른 문자로 읽히는 기존 오류까지 해결됐다는 뜻도 아닙니다.</p>
<p>기존 문자 합성 260장과 실제 도면 22곳×3크기의 저장된 모델 출력을 재검사했으며, 기존 결과는 변하지 않았습니다. 예전 | 합성 자료는 글자와 높이가 같게 제작되어 이번 길이 가정을 만족하지 않습니다.</p>'''+''.join(sections)+'''
<h2>DWG</h2><p><a href="../cad_bar_height_fixed_dwg/integrated_bar_height_fixed.dwg">길이 기준을 반영한 DWG</a> · <a href="../cad_bar_height_fixed_dwg/conversion_report.json">DWG 보존 검사</a></p>
<p>기존 정답지 308칸은 그대로입니다. 새로 바꾼 7개는 이 정답지 밖에 있으며, 사용자가 지정한 길이 조건과 원본 선 형상으로 처리했습니다. LLM·단어 치환·특정 도면 좌표 규칙을 쓰지 않았습니다.</p>
<p>주변 글자 없이 |만 있는 첨부 그림은 검출 후보는 찾지만, 길이 비교 기준을 확보할 수 없어 이 추가 판정은 보류합니다.</p>
<p><a href="results.json">합성/기존 검사 수치</a> · <a href="tables/analysis.json">표 후보 검사</a> · <a href="vector_corrections/predictions.json">선 좌표 기반 변경 기록</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')


def diagnostic(seed,glyphs):
    rng=np.random.default_rng(seed)
    image=np.full((240,384,3),255,np.uint8)
    height=int(rng.integers(35,65));baseline=180
    # Random reference letters plus two visually plain stems with different heights.
    ids=[int(rng.choice([0,1,2,3,4,7,10,12,13,14,15,17,18,19,20,21,22,23,24,25])),8,67,int(rng.choice([0,3,4,7,12,14,15,17,20,21,24]))]
    objects=[]
    for x,cid in zip([20,125,195,270],ids):
        if cid in (8,67):
            tall=height if cid==8 else round(height*rng.uniform(1.25,1.55))
            width=max(3,round(height*.12))
            image[baseline-tall:baseline,x:x+width]=0
            objects.append(dict(class_id=cid,bbox_xyxy=[x,baseline-tall,x+width,baseline]))
        else:
            alpha,offset=resize_glyph(glyphs[cid],height,2)
            item=paste_glyph(image,alpha,x,baseline+offset,np.zeros(3),cid,0,NAMES[cid])
            objects.append(item)
    if seed%2:image=255-image
    return image,objects


def replay_existing():
    source=ROOT.parent/'output/cad_vertical_bar_branch_scaled_audit/frozen_predictions.json'
    payload=json.loads(source.read_text(encoding='utf-8'))
    summary={};decisions=Counter()
    for rec in payload['records']:
        image=read_image(Path(rec['path']))
        d=rec['bar_decisions'];proposed=[r['bar'] for r in d['accepted']]+d['rejected']
        merged,detail=merge_bars(image,rec['dual_old'],proposed,min_height_ratio=RATIO)
        group=rec['group'];counts=summary.setdefault(group,dict(images=0,changed=0,accepted=0))
        counts['images']+=1;counts['accepted']+=len(detail['accepted'])
        counts['changed']+=int(bool(detail['accepted']))
        decisions.update(r['reason'] for r in detail['height_checks'])
    return dict(summary=summary,height_decisions=dict(decisions),
        note='Old synthetic | templates were resized to normal character height. They violate the new longer-| assumption and are reported separately, not relabelled.')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    reader=VerticalBarReader(ROOT/'vertical_bar_branch/training/weights/best.pt',adapt_large=True,min_height_ratio=RATIO)
    glyphs,_=extract_glyphs(OLD/'source_alphabet.png',OUT)
    records=[]
    for i in range(60):
        image,truth=diagnostic(203000+i,glyphs)
        result,baseline,details=reader.predict(image,f'diagnostic_{i}','tiles',384)
        new=result.boxes.cpu().numpy().data.tolist()
        prior=details['vertical_bar']
        proposed=[r['bar'] for r in prior['accepted']]+prior['rejected']
        # Recreate the unconstrained reader from the same frozen detector candidates.
        from preserved_reader import merge_predictions
        old,_=merge_predictions(baseline.boxes.cpu().numpy().data.tolist(),details['symbols'],details['cleaned'])
        unconstrained,_=merge_bars(image,old.tolist(),proposed)
        labels=[(o['class_id'],np.asarray(o['bbox_xyxy'],dtype=float)) for o in truth]
        scores={}
        for name,pred in [('without_height',unconstrained),('with_height',new)]:
            bar_pred=[b for b in pred if int(b[5])==67]
            hits,fp,fn=match(bar_pred,[t for t in labels if t[0]==67])
            scores[name]=dict(tp=len(hits),fp=fp,fn=fn)
        records.append(dict(id=i,truth=truth,scores=scores,height_checks=prior['height_checks']))
        if i<8:
            write_image(OUT/f'diagnostic_{i}.png',image)
            write_image(OUT/f'diagnostic_{i}_detected.png',result.plot(line_width=1,font_size=10))
    summary={name:{k:sum(r['scores'][name][k] for r in records) for k in ['tp','fp','fn']} for name in ['without_height','with_height']}
    source=read_image(ROOT/'vertical_bar/source_vertical_bar.png')
    result,_,detail=reader.predict(source)
    report=dict(min_height_ratio=RATIO,method='Actual ink height / local same-row cap-height references; at least two unambiguous reference letters. Unknown reference defers.',
        scope='Same-template and constructed longer-bar diagnostics, plus replay of prior frozen predictions. Not independent drawing accuracy.',
        diagnostics=dict(images=len(records),summary=summary,records=records),existing=replay_existing(),source_height_checks=detail['vertical_bar']['height_checks'])
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_report(report)
    print(json.dumps(dict(diagnostics=summary,existing=report['existing'],source_height_checks=report['source_height_checks'])),flush=True)


if __name__=='__main__':main()
