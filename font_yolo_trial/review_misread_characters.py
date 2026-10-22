"""Review saved DWG characters only; never alter recognition or CAD files."""
import hashlib
import html
import json
from pathlib import Path

import ezdxf
import pymupdf as fitz
from PIL import Image, ImageDraw

from vector_font_probe import glyph_canvas, load_candidates

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output/cad_text_audit'


def main():
    audit = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
    directory = ROOT / 'output/cad_readability_fixed'
    manifest = json.loads((directory / 'conversion_report.json').read_text(encoding='utf-8'))
    dwg = Path(audit['source_dwg'])
    assert hashlib.sha256(dwg.read_bytes()).hexdigest() == audit['dwg_sha256'] == manifest['dwg_sha256']
    recognition = Path(manifest['recognition_source'])
    assert hashlib.sha256(recognition.read_bytes()).hexdigest() == manifest['recognition_sha256']
    saved = ezdxf.readfile(directory / 'dwg_roundtrip.dxf')
    for row in manifest['texts']:
        assert saved.entitydb[row['handle']].dxf.text == row['text']
    rows = audit['captions']
    assert len(rows) == 48 and len(manifest['texts']) == 152
    # Compare literal characters: no case folding or O/0 equivalence is allowed.
    compact = lambda value: ''.join(value.split())
    differences = [r for r in rows if compact(r['reference']) != compact(r['actual'])]
    page = fitz.open(ROOT / 'font_yolo_trial/cad_without_tables/source.pdf')[0]
    candidates = {f'P{p["path_id"]}': p for p in load_candidates(page)}
    focused = [
        ('P39236', 'bll15914', '두 l은 아래쪽 짧은 가로획이 있고, 숫자 1은 위쪽 사선과 받침이 있습니다. 현재 문자열과 원본 획이 일치합니다.'),
        ('P39105', 'SA-C8D-CBD-BK', '8은 위아래 고리, B는 왼쪽 직선이 있습니다. 하이픈 3개와 P의 아래쪽 열린 획도 원본과 대조했습니다.'),
        ('P39252', "HS\'G", '작은 기호는 제공받은 작은따옴표 모양과 대응합니다. 원본에서 기호를 구성하는 선을 따로 표시했습니다.'),
        ('P38557', "TAP\'G", '작은따옴표 위치와 원본 획을 대조했습니다.'),
        ('P39233', 'OPU88567', '숫자 88567은 원본과 일치합니다. 맨 앞 글자의 의미가 알파벳 O인지 숫자 0인지는 이 윤곽 대조만으로 확정하지 않았습니다.'),
    ]
    checks = []
    for field, token, note in focused:
        p = candidates[field]
        groups = p['groups']
        strip = Image.new('RGB', (144 * min(10, len(groups)), 160 * ((len(groups) + 9) // 10)), 'white')
        draw = ImageDraw.Draw(strip)
        for i, group in enumerate(groups):
            tile = Image.fromarray(glyph_canvas(group, p['cap'], 80, p['width'])).crop((124, 124, 260, 260))
            x, y = (i % 10) * 144, (i // 10) * 160
            strip.paste(tile, (x + 4, y + 20))
            draw.text((x + 6, y + 3), str(i + 1), fill='#444444')
        name = field + '_misread_parts.png'
        strip.save(OUT / 'assets' / name)
        checks.append(dict(id=field, token=token, note=note, parts_image=name))
    result = dict(source_dwg=str(dwg), dwg_sha256=audit['dwg_sha256'],
                  scope='Converted characters only. Circular callout numbers, other omissions and whitespace are excluded.',
                  body_captions=48, border_labels=104, preserved_native_text=59,
                  literal_reference_differences=differences,
                  confirmed_misread_captions=len(differences), focused_checks=checks,
                  limitation='Same-PDF manual review, not an independent benchmark. Some code O/0 identities remain unverified. Recognition and CAD are unchanged.')
    (OUT / 'misread_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    esc = html.escape
    detail = ''.join(f'<article><h2>{r["id"]} · {esc(r["token"])}</h2><p>{r["note"]}</p><img src="assets/{r["id"]}_source.png"><details><summary>원본 획을 떨어뜨려 보기</summary><img src="assets/{r["parts_image"]}"></details></article>' for r in checks)
    table = ''.join(f'<tr><td>{r["id"]}</td><td><img loading="lazy" src="assets/{r["source_image"]}"></td><td>{esc(r["actual"])}</td></tr>' for r in rows)
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>변환 문자 오인식 재검사</title>
<style>body{{font:16px system-ui;line-height:1.65;max-width:1500px;margin:30px auto;padding:0 20px;color:#203040}}article{{border:1px solid #ccd5df;padding:18px;margin:18px 0}}img{{max-width:100%;height:auto}}td,th{{border:1px solid #ccd5df;padding:12px;vertical-align:top;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}.result{{background:#eef5fb;padding:18px}}details{{margin-top:15px}}</style>
<h1>변환된 문자에서 틀리게 읽힌 부분 검사</h1><p>대상: non_table_readable_yolo_ocr.dwg. 저장된 DWG의 재내보내기 문자와 인식 결과가 일치하는지 다시 확인했습니다.</p>
<p class="result"><b>본문 48문구 대조에서 확정된 문자 오인식: {len(differences)}곳.</b><br>동그라미 안 숫자, 미변환 영역과 띄어쓰기는 이번 검사에서 제외했습니다. 테두리 104표기와 기존 문자 59개는 앞선 검사에서 차이가 없었습니다.</p>
<p>정답지 비교뿐 아니라 아래 의심 항목은 원본 확대본과 분리한 획으로 다시 확인했습니다. <b>일부 코드 O/0의 의미는 미확정</b>이며, 이 결과를 전체 도면 100% 정확도로 해석할 수 없습니다. 같은 PDF를 대상으로 한 수동 검토이며, 인식 결과나 CAD 파일을 수정하지 않았습니다.</p>
{detail}<h2>본문 48문구 대조</h2><table><tr><th style="width:9%">위치</th><th style="width:48%">원본 PDF</th><th>실제 DWG 문자</th></tr>{table}</table><p><a href="misread_review.json">검사 기록</a> · <a href="report.html">이전 전체 검사</a></p></html>'''
    (OUT / 'misread_review.html').write_text(document, encoding='utf-8')
    assert hashlib.sha256(dwg.read_bytes()).hexdigest() == audit['dwg_sha256']
    assert len(checks) == 5 and all((OUT / 'assets' / r['parts_image']).exists() for r in checks)
    print(json.dumps(dict(confirmed_misreads=len(differences), focused_checks=len(checks), dwg_unchanged=True)))


if __name__ == '__main__':
    main()
