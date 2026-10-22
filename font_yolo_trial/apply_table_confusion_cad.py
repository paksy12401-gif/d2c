"""Apply saved answer-free table corrections to a copy of the latest CAD."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

import ezdxf
import numpy as np

import export_fusion_review_cad as cad
from export_auto_large_cad import geometry
from export_integrated_review_cad import sha
from export_tables_review_cad import font_bounds
from export_title_review_cad import texts

STEM = 'integrated_ocr_confusion_fixed'


def ink_bounds(e):
    ax, ay, bx, by = font_bounds(e.dxf.text)
    h, w = e.dxf.height, e.dxf.width
    x, y, _ = e.dxf.insert
    return [x+ax*h*w, y+ay*h, x+bx*h*w, y+by*h]


def prepare(args):
    path = args.output/'conversion_report.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        assert sha(args.base) == report['base_sha256'] and sha(args.predictions) == report['predictions_sha256']
        return report
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    assert predictions['answers_loaded'] is False and predictions['llm_inference'] is False
    doc = ezdxf.readfile(args.base); model = doc.modelspace(); updates = []
    entities = {}
    for e in model.query('TEXT'):
        if e.has_xdata('CAD_TABLE_OCR'):
            for tag in e.get_xdata('CAD_TABLE_OCR'):
                if tag.code == 1000:
                    if tag.value in entities:
                        raise ValueError('Duplicate table cell identity')
                    entities[tag.value] = e
    for change in predictions['changes']:
        e = entities[change['id']]
        if e.dxf.text != change['before'] or e.dxf.rotation != 0 or e.dxf.halign != 0:
            raise ValueError('CAD text differs from the correction baseline')
        assert doc.styles.get(e.dxf.style).dxf.font.lower() == 'arial.ttf'
        bounds = ink_bounds(e); x0, y0, x1, y1 = bounds
        ax, ay, bx, by = font_bounds(change['after'])
        height = (y1-y0)/(by-ay); width = (x1-x0)/((bx-ax)*height)
        e.dxf.text = change['after']; e.dxf.height = height; e.dxf.width = width
        e.dxf.insert = (x0-ax*height*width, y0-ay*height, 0)
        np.testing.assert_allclose(ink_bounds(e), bounds, rtol=0, atol=1e-8)
        updates.append(dict(cell_id=change['id'], handle=e.dxf.handle, before=change['before'],
                            after=change['after'], ink_bounds_cad=bounds))
    if not updates:
        raise ValueError('No supported corrections to export')
    dxf = args.output/(STEM+'.dxf'); doc.saveas(dxf)
    report = dict(base_sha256=sha(args.base), predictions_sha256=sha(args.predictions),
                  inference_llm=False, answers_used=False, updated=updates, dxf_sha256=sha(dxf))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def verify(args, report, path):
    old = ezdxf.readfile(args.base); new = ezdxf.readfile(path)
    a, b = old.modelspace(), new.modelspace(); changed = {r['handle']:r for r in report['updated']}
    assert not new.audit().has_errors and len(a) == len(b)
    old_text, new_text = texts(a), texts(b)
    assert len(old_text) == len(new_text)
    for h, value in old_text.items():
        assert new_text[h] == (changed[h]['after'] if h in changed else value)
    for e in a:
        actual = new.entitydb[e.dxf.handle]
        assert e.dxf.layer == actual.dxf.layer and e.dxf.color == actual.dxf.color
        if e.dxf.handle not in changed:
            assert geometry(e) == geometry(actual)
    for item in report['updated']:
        np.testing.assert_allclose(ink_bounds(new.entitydb[item['handle']]), item['ink_bounds_cad'], rtol=0, atol=1e-7)
    for layer in old.layers:
        assert new.layers.get(layer.dxf.name).dxf.color == layer.dxf.color
    assert Counter(e.dxf.name for e in a.query('INSERT')) == Counter(e.dxf.name for e in b.query('INSERT'))
    return dict(entities=len(b), editable_text=len(new_text), modified_text=len(changed),
                other_entities_preserved=len(a)-len(changed), symbols=len(b.query('INSERT')),
                centerlines=sum(e.dxf.layer.startswith('GREEN_CENTERLINE_') for e in b))


def preview(args, report):
    from ezdxf.addons.drawing import RenderContext, Frontend, config, layout, pymupdf
    from ezdxf.math import BoundingBox2d
    settings = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                                    color_policy=config.ColorPolicy.BLACK)
    sections = []
    for i, item in enumerate(report['updated']):
        x0, y0, x1, y1 = item['ink_bounds_cad']; pad = (y1-y0)*.5
        for label, file in [('before', args.base), ('after', args.output/'dwg_roundtrip.dxf')]:
            doc = ezdxf.readfile(file); backend = pymupdf.PyMuPdfBackend()
            Frontend(RenderContext(doc), backend, config=settings).draw_layout(doc.modelspace(), finalize=True,
                       filter_func=lambda e:e.dxf.handle == item['handle'])
            name = f'cell{i}_{label}.png'
            (args.output/name).write_bytes(backend.get_pixmap_bytes(layout.Page(380, 30, margins=layout.Margins.all(2)),
                  dpi=150, render_box=BoundingBox2d([(x0-pad, y0-pad), (x1+pad, y1+pad)])))
        sections.append('<p>'+escape(item['before'])+' → '+escape(item['after'])+'</p><img src="cell'+str(i)+'_before.png"><img src="cell'+str(i)+'_after.png">')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>표 C/0 오류 수정 DWG</title><style>body{font:17px system-ui;line-height:1.7;max-width:1300px;margin:32px auto}img{max-width:100%;border:1px solid #ddd}</style>
<h1>표 C/0 오류 수정 DWG</h1><p>단어 정답을 넣지 않고 YOLO C 후보·EasyOCR 두 입력·오른쪽이 열린 원본 획을 모두 확인해 반영했습니다. 추가 학습이나 LLM 추론은 사용하지 않았습니다.</p>'''+''.join(sections)+'''
<p><a href="'''+STEM+'''.dwg">DWG 받기</a> · <a href="conversion_report.json">DWG 검증 기록</a></p>
<p>기존 308칸 회귀 검사에서 오답 1칸이 수정됐고 악화된 칸은 없었습니다. 새 도면 검증이나 나머지 572칸의 정답률 측정은 아닙니다. 아직 정답이 확인되지 않은 별도 기호는 학습하지 않았습니다.</p></html>'''
    if STEM != 'integrated_ocr_confusion_fixed':
        html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>표 문자 보정 DWG</title>
<style>body{font:17px system-ui;max-width:1300px;margin:32px auto;line-height:1.7}img{max-width:100%;border:1px solid #ddd}</style>
<h1>표 문자 보정 DWG</h1><p>저장된 로컬 인식·기하 검증 결과를 반영했습니다. 아래는 변경한 셀의 전후 비교입니다.</p>'''+''.join(sections)+f'''
<p><a href="{STEM}.dwg">DWG 받기</a> · <a href="conversion_report.json">변경 내용과 보존 검사</a></p>
<p>수정한 셀 외의 문자·심볼·중앙선·원본 기하는 그대로 보존했습니다.</p></html>'''
    (args.output/'report.html').write_text(html, encoding='utf-8')


def main():
    global STEM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dwg', action='store_true')
    parser.add_argument('--stem',default=STEM)
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if not args.stem or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.stem):
        raise ValueError('Output stem must use letters, digits, underscores or hyphens')
    STEM=args.stem
    report = prepare(args); report['verification'] = verify(args, report, args.output/(STEM+'.dxf'))
    (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TABLE_CORRECTION_DXF', json.dumps(report['verification']), flush=True)
    if args.dwg:
        cad.OUT = args.output; dwg = args.output/(STEM+'.dwg'); roundtrip = args.output/'dwg_roundtrip.dxf'
        if dwg.exists() or roundtrip.exists():
            raise FileExistsError('Preserve previous DWG')
        cad.run_cad(args.output/(STEM+'.dxf'), 'save_correction', ['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
        cad.run_cad(dwg, 'verify_correction', ['_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
        report.update(dwg_reopened=True, dwg_sha256=sha(dwg), verification=verify(args, report, roundtrip))
        (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        preview(args, report)
        print('TABLE_CORRECTION_DWG', json.dumps(report['verification']), flush=True)


if __name__ == '__main__':
    main()
