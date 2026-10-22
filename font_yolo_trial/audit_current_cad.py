"""Audit saved CAD text against separate references and document confirmed omissions."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import ezdxf
import pymupdf as fitz

from evaluate_real_validation import alignment,canonical

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_text_audit'
REF=ROOT/'font_yolo_trial/ocr_refinement'


def boundaries(text):
    result=[];n=0
    for match in re.finditer(r'\S+|\s+',text):
        token=match.group()
        if token.isspace():
            before=text[match.start()-1] if match.start() else ''
            after=text[match.end()] if match.end()<len(text) else ''
            if before.isalnum() and after.isalnum():result.append(n)
        else:n+=len(token)
    return set(result)


def main():
    OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
    directory=ROOT/'output/cad_readability_fixed'
    manifest=json.loads((directory/'conversion_report.json').read_text(encoding='utf-8'))
    dwg=directory/'non_table_readable_yolo_ocr.dwg'
    assert hashlib.sha256(dwg.read_bytes()).hexdigest()==manifest['dwg_sha256']
    recognition=Path(manifest['recognition_source']);source_data=json.loads(recognition.read_text(encoding='utf-8'))
    assert hashlib.sha256(recognition.read_bytes()).hexdigest()==manifest['recognition_sha256']
    actual=ezdxf.readfile(directory/'dwg_roundtrip.dxf');actual_text={e.dxf.handle:e.dxf.text for e in actual.modelspace().query('TEXT')}
    assert all(actual_text[r['handle']]==r['text'] for r in manifest['texts'])
    answers={}
    reference_files=[REF/n for n in ['full_caption_answers.json','connector_caption_answers.json','branch_caption_answers.json']]
    for file in reference_files:
        for row in json.loads(file.read_text(encoding='utf-8'))['answers']:
            if row['id'] in answers:raise ValueError('Overlapping full-caption references')
            answers[row['id']]=row
    body=[r for r in manifest['texts'] if r['id'].startswith('P')]
    assert len(body)==len(answers)==48
    page=fitz.open(source_data['source'])[0]
    rows=[];spacing=[]
    for record in body:
        reference=answers[record['id']]['reference'];text=actual_text[record['handle']]
        score=alignment(reference,text,answers[record['id']].get('ambiguities',[]))
        missing=[]
        if canonical(reference)==canonical(text):
            joined=canonical(reference)
            for index in sorted(boundaries(reference)-boundaries(text)):
                missing.append(dict(index=index,reference_context=joined[max(0,index-7):index]+' '+joined[index:index+7],actual_context=joined[max(0,index-7):index+7]))
        if missing:spacing.append(dict(id=record['id'],reference=reference,actual=text,missing_word_separators=missing))
        image=record['id']+'_source.png';target=assets/image
        if not target.exists():
            box=fitz.Rect(record['box_pdf']);scale=65/max(.1,box.height)
            page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=box+(-1,-1,1,1),alpha=False).save(target)
        rows.append(dict(id=record['id'],box_pdf=record['box_pdf'],reference=reference,actual=text,score=score,
                         spacing_issue=bool(missing),source_image=image,parts_image=record['id']+'_parts.png' if (assets/(record['id']+'_parts.png')).exists() else None))
    frames={r['id']:r['reference'] for r in json.loads((REF/'frame_answers.json').read_text(encoding='utf-8'))['answers']}
    frame_errors=[r for r in manifest['texts'] if r['id'].startswith('F') and actual_text[r['handle']]!=frames[r['id']]]
    base=ezdxf.readfile(ROOT/'font_yolo_trial/cad_without_tables/non_table_geometry.dxf')
    native=[e for e in base.modelspace().query('MTEXT')]
    native_errors=[e.dxf.handle for e in native if actual.entitydb[e.dxf.handle].text!=e.text]
    native_pdf=[w[4] for w in page.get_text('words') if not any(t[0]<(w[0]+w[2])/2<t[2] and t[1]<(w[1]+w[3])/2<t[3] for t in manifest['table_exclusions_pdf'])]
    assert Counter(native_pdf)==Counter(e.text for e in native)
    circles=[c for c in json.loads((OUT/'circles.json').read_text(encoding='utf-8')) if c['radius']>9]
    checks=json.loads((OUT/'circle_path_checks.json').read_text(encoding='utf-8'))
    check_by_handle={r['circle_handle']:r for r in checks}
    # Reference labels visually transcribed from the source-only contact sheet; no inference consumes them.
    labels=dict(zip(['BB15','BB16','BB17','BB18','BB19','BB1A','BB1B','BB1C','BB1D','BB1E','BB1F','BB20','BB21','BB24','BB25','BB26','BB27','BB28','BB2A','BB2B','BB2C','BB2D','BB2E','BB2F','BB30','BB32','BB33','BB34','BB35','BB36','BB37','BB38','BB39','BB3A','BB3B','BB3C','BB3E','BB3F','BB41','BB43'],['6','18','19','15','7','23','3','22','8','1','13','14','29','11','10','26','4','24','14','14','14','14','11','11','13','17','25','21','5','13','7','7','7','7','7','7','7','7','7','12']))
    labels.update(BB22='31',BB23='12',BB29='14',BB31='20',BB3D='7',BB40='7',BB42='9',BB44='7')
    omissions=[]
    for c in circles:
        check=check_by_handle[c['handle']]
        assert check['inside_paths'] and not check['editable_text_inside']
        circle=actual.entitydb[c['handle']]
        assert circle.dxftype() in ('CIRCLE','LWPOLYLINE')
        omissions.append(dict(**c,source_label=labels[c['handle']],editable_text_present=False,
                              source_outline_preserved=True,image='circle_'+c['handle']+'.png'))
    errors=[r for r in rows if r['score']['edit_distance']]
    filter_evidence=json.loads((OUT/'omission_filter_evidence.json').read_text(encoding='utf-8'))
    assert filter_evidence['confirmed']==48 and all(not r['in_candidates'] for r in filter_evidence['paths'])
    result=dict(scope='Audit of the latest delivered DWG. 48 selected complete captions reviewed, 104 border labels checked, 59 native text objects preserved. 48 circular callout labels confirmed still unconverted (40 circle entities and 8 closed polylines). Not a certified complete-page recall benchmark.',
        source_dwg=str(dwg.resolve()),dwg_sha256=manifest['dwg_sha256'],recognition_sha256=manifest['recognition_sha256'],
        dwg_text_matches_recognition=True,reference_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files},
        confirmed_character_error_captions=len(errors),caption_count=48,word_spacing_issue_captions=len(spacing),
        border_labels=104,border_errors=frame_errors,native_text=59,native_errors=native_errors,
        confirmed_unconverted_circular_labels=len(omissions),
        omission_filter_conditions=filter_evidence['condition_counts'],
        ambiguity_note='Outline-only O/0 in some alphanumeric codes cannot establish the original semantic character. Same-PDF references, including reference corrections, are not an independent accuracy benchmark.',
        finding_notes=['Character scoring ignores whitespace; missing word separators are listed separately.',
                       'Circular labels remain visible as source polylines, but are not editable text. Candidate width/group/height filters exclude them. See omission_filter_evidence.json for all 48 paths and exact condition counts.',
                       'Dark original callout color remains on those unconverted paths; the recent Arial/ACI display fix affected converted text layers only.',
                       'No CAD or recognition output was rewritten by this audit. Other unselected very small or image-only text is not proven absent.'],
        captions=rows,character_errors=errors,spacing_issues=spacing,omissions=omissions)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    esc=html.escape
    caption_rows=''.join(f'<tr id="{r["id"]}"><td>{r["id"]}</td><td><img loading="lazy" src="assets/{r["source_image"]}"><pre>{esc(r["reference"])}</pre>'+ (f'<details><summary>원본 선 조각 분리 보기</summary><img loading="lazy" src="assets/{r["parts_image"]}"></details>' if r['parts_image'] else '')+f'</td><td><pre>{esc(r["actual"])}</pre></td><td>{"띄어쓰기 점검" if r["spacing_issue"] else "확정 글자 오류 없음"}<br>문자 차이 {r["score"]["edit_distance"]}</td></tr>' for r in rows)
    spacing_rows=''.join(f'<tr><td><a href="#{r["id"]}">{r["id"]}</a></td><td>{esc(" / ".join(x["reference_context"] for x in r["missing_word_separators"]))}</td><td>{esc(" / ".join(x["actual_context"] for x in r["missing_word_separators"]))}</td></tr>' for r in spacing)
    omission_rows=''.join(f'<tr><td>{r["handle"]}</td><td>{r["source_label"]}</td><td>({r["x"]:.1f}, {r["y"]:.1f})</td><td><img loading="lazy" src="assets/{r["image"]}"></td><td>원본 선만 남음 · TEXT 없음</td></tr>' for r in omissions)
    page_html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>최신 DWG 문자 점검</title><style>body{font:16px system-ui;margin:25px;line-height:1.7;color:#203040}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #cbd3dc;padding:12px;vertical-align:top;overflow-wrap:anywhere}th{background:#eef2f7}pre{white-space:pre-wrap;font:inherit}img{max-width:100%;max-height:150px;object-fit:contain}details img{max-height:260px}a{color:#175ba6}.finding{background:#fff3dd;padding:16px}#circles img{width:135px;height:135px}</style>
<h1>최신 DWG 문자 점검</h1><p>전달한 가독성 수정 DWG를 기준으로 원본 PDF와 대조했습니다. DWG 해시와 재내보내기한 문자가 당시 인식 결과와 일치함을 확인했습니다. 인식 결과나 DWG를 수정하지 않았습니다.</p>
<div class="finding"><b>확정 누락: 원 안 번호 48곳이 편집 가능한 문자로 변환되지 않았습니다.</b><br>번호는 원본 선으로 남아 보여도 TEXT가 아닙니다. 원 외곽은 CIRCLE 40개와 닫힌 폴리라인 8개로 나뉘어 있었습니다. 예: 원 안 26, 25, 14. 짧은 번호가 글자 후보의 가로폭·조각 수·높이 조건에 걸렸습니다. 모델에 보여주기 전 후보 선정 단계의 누락이며, 경로별 탈락 조건을 근거 파일에 기록했습니다.</div>
<p>선택된 본문 48문구에서 공백을 제외한 확정 치환·누락은 찾지 못했습니다. 띄어쓰기 누락은 5문구에서 확인했습니다. 테두리 104표기는 정답지와 일치했고, 기존 PDF 문자 59개는 보존됐습니다. 이 결과가 전체 도면 100% 정확도를 의미하지 않습니다. 일부 코드 O/0는 윤곽만으로 원래 의미를 확정할 수 없습니다.</p>
<p><a href="#spacing">띄어쓰기 문제</a> · <a href="#circles">원 안 번호 누락 48곳</a> · <a href="#captions">본문 48문구 대조</a> · <a href="audit.json">검사 원시 기록</a></p>
<h2 id="spacing">띄어쓰기 누락</h2><table><thead><tr><th>위치</th><th>원본 대조 기준</th><th>현재 DWG</th></tr></thead><tbody>__SPACING__</tbody></table>
<h2 id="circles">문자로 변환되지 않은 원 안 번호 48곳</h2><p><a href="missing_number_locations.png">누락 위치 지도</a> · <a href="circled_number_sources.png">48곳 원본 확대 모음</a> · <a href="omission_filter_evidence.json">후보 필터 탈락 근거</a></p><p>번호는 중복될 수 있습니다. 좌표는 원본 PDF 기준이며, CAD에서는 X/72, (2383.93994140625-Y)/72로 대응합니다. 원의 외곽과 내부 숫자는 원본 선 객체로 보존됐습니다.</p><table><thead><tr><th>원 객체 핸들</th><th>원 안 번호</th><th>PDF 중심 좌표</th><th>원본</th><th>검사 결과</th></tr></thead><tbody>__OMISSIONS__</tbody></table>
<h2 id="captions">본문 전체 48문구 대조</h2><p>동일 PDF의 수동 대조 자료입니다. 일부 숫자는 원본 획 분리 재검토로 정답지를 수정한 이력이 있습니다. 정답지 수정은 인식 성능 향상으로 계산하지 않았습니다. 검증용 자료는 인식 프로그램에 전달하지 않습니다.</p><table><thead><tr><th style="width:8%">위치</th><th style="width:42%">원본 / 대조 기준</th><th style="width:35%">DWG 문자</th><th>결과</th></tr></thead><tbody>__CAPTIONS__</tbody></table></html>'''
    (OUT/'report.html').write_text(page_html.replace('__SPACING__',spacing_rows).replace('__OMISSIONS__',omission_rows).replace('__CAPTIONS__',caption_rows),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['caption_count','confirmed_character_error_captions','word_spacing_issue_captions','border_labels','native_text','confirmed_unconverted_circular_labels']},ensure_ascii=True),flush=True)
    print(json.dumps(spacing,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
