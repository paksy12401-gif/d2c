"""Add the saved local title result to a copy of the integrated CAD drawing."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment
import pymupdf as fitz

import export_fusion_review_cad as cad
from export_integrated_review_cad import sha

STEM = 'integrated_with_editable_title'
TEXT_LAYER = 'TITLE_OCR_TEXT'
SOURCE_LAYER = 'TITLE_SOURCE_OUTLINES'


def texts(model):
    return {e.dxf.handle: e.dxf.text if e.dxftype() == 'TEXT' else e.text
            for e in model.query('TEXT MTEXT')}


def prepare(args):
    out = args.output; manifest = out/'conversion_report.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        assert sha(args.recognition) == data['recognition_sha256']
        assert sha(args.base) == data['base_sha256']
        assert sha(out/(STEM+'.dxf')) == data['dxf_sha256']
        return data
    result = json.loads(args.recognition.read_text(encoding='utf-8'))
    assert result['llm_inference'] is False and result['answers_loaded'] is False
    outputs = [r['fused_lines'] for r in result['results']]
    if not outputs or any(lines != outputs[0] for lines in outputs) or len(outputs[0]) != 1:
        raise ValueError('Reader configurations disagree; do not automatically export')
    text = outputs[0][0]; field = result['field']
    geometry = json.loads(args.geometry.read_text(encoding='utf-8'))
    page = fitz.open(geometry['source_pdf'])[0]; page_height = page.rect.height
    segments = [s for g in field['groups'] for s in g['segments']]
    index = cad.StrokeIndex({field['id']: segments})
    doc = ezdxf.readfile(args.base); model = doc.modelspace()
    original_text = texts(model); count = len(model)
    original_inserts = Counter(e.dxf.name for e in model.query('INSERT'))
    centers = Counter(e.dxf.layer for e in model if e.dxf.layer.startswith('GREEN_CENTERLINE_'))
    if SOURCE_LAYER in doc.layers or TEXT_LAYER in doc.layers:
        raise ValueError('Title layers already exist')
    doc.layers.new(SOURCE_LAYER, dxfattribs={'color':8})
    doc.layers.new(TEXT_LAYER, dxfattribs={'color':6})
    moved = []; matched = set()
    for e in model:
        if e.dxftype() == 'LWPOLYLINE':
            vertices = list(e.get_points('xyb'))
            if any(abs(p[2]) > 1e-10 for p in vertices):
                continue
            points = [(p[0]*72, page_height-p[1]*72) for p in vertices]
            identity, edges = index.match(points, e.closed)
        elif e.dxftype() == 'LINE':
            points = [(p.x*72, page_height-p.y*72) for p in (e.dxf.start, e.dxf.end)]
            identity, edges = index.match(points)
        else:
            continue
        if identity:
            moved.append(dict(handle=e.dxf.handle, previous_layer=e.dxf.layer))
            e.dxf.layer = SOURCE_LAYER; e.dxf.color = 256
            e.dxf.discard('true_color')
            matched.update(edges)
    if matched != set(index.edges):
        raise ValueError(f'Title source geometry coverage incomplete: {len(matched)}/{len(index.edges)}')
    doc.layers.get(SOURCE_LAYER).off()
    style = 'TITLE_LOCAL_KOREAN'
    if style not in doc.styles:
        doc.styles.new(style, dxfattribs={'font':'malgun.ttf'})
    b = field['box']; baseline = field['baseline']
    title = model.add_text(text, dxfattribs=dict(layer=TEXT_LAYER, style=style,
                                                height=field['cap']/72, color=256))
    title.set_placement((b[0]/72, (page_height-baseline)/72),
                        (b[2]/72, (page_height-baseline)/72), align=TextEntityAlignment.FIT)
    if 'CAD_TITLE_LOCAL' not in doc.appids:
        doc.appids.new('CAD_TITLE_LOCAL')
    title.set_xdata('CAD_TITLE_LOCAL', [(1000, field['id'])])
    dxf = out/(STEM+'.dxf'); doc.saveas(dxf)
    reread = ezdxf.readfile(dxf); space = reread.modelspace()
    assert not reread.audit().has_errors
    assert len(space) == count+1
    assert all(texts(space)[k] == v for k, v in original_text.items())
    assert len(texts(space)) == len(original_text)+1
    assert Counter(e.dxf.name for e in space.query('INSERT')) == original_inserts
    assert Counter(e.dxf.layer for e in space if e.dxf.layer.startswith('GREEN_CENTERLINE_')) == centers
    report = dict(inference_llm=False, answers_used=False, review_only=True,
                  source_pdf_sha256=sha(Path(geometry['source_pdf'])),
                  base=str(args.base.resolve()), base_sha256=sha(args.base),
                  recognition=str(args.recognition.resolve()), recognition_sha256=sha(args.recognition),
                  dxf_sha256=sha(dxf), title=text, title_handle=title.dxf.handle,
                  field_id=field['id'], box_pdf=b, page_height=page_height,
                  original_entities=count, original_text_count=len(original_text),
                  added_editable_text=1, title_source_edges=len(index.edges),
                  matched_title_source_edges=len(matched), hidden_originals=moved,
                  symbol_inserts=dict(original_inserts), centerlines=dict(centers),
                  limitations=['Only the selected title was improved; other recognition strings unchanged.',
                               'Same-title agreement is not unseen-drawing accuracy.',
                               'Editable text uses Malgun Gothic fitted to the source width.'])
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TITLE_DXF_READY', json.dumps(dict(title=text, hidden=len(moved), edges=len(matched)), ensure_ascii=True), flush=True)
    return report


def convert(args, report):
    out = args.output; cad.OUT = out
    dwg = out/(STEM+'.dwg'); roundtrip = out/'dwg_roundtrip.dxf'
    if dwg.exists() or roundtrip.exists():
        raise FileExistsError('Preserve the previously verified DWG')
    cad.run_cad(out/(STEM+'.dxf'), 'save_title', ['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
    if not dwg.is_file() or dwg.stat().st_size < 1000:
        raise RuntimeError('DWG not saved')
    cad.run_cad(dwg, 'verify_title', ['_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
    actual = ezdxf.readfile(roundtrip); space = actual.modelspace()
    source = ezdxf.readfile(args.base)
    old = texts(source.modelspace()); now = texts(space)
    assert not actual.audit().has_errors
    assert len(space) == report['original_entities']+1
    assert len(now) == len(old)+1 and all(now[h] == t for h, t in old.items())
    assert now[report['title_handle']] == report['title']
    assert Counter(e.dxf.name for e in space.query('INSERT')) == Counter(report['symbol_inserts'])
    assert Counter(e.dxf.layer for e in space if e.dxf.layer.startswith('GREEN_CENTERLINE_')) == Counter(report['centerlines'])
    assert actual.layers.get(SOURCE_LAYER).is_off() and not actual.layers.get(TEXT_LAYER).is_off()
    assert all(actual.entitydb[r['handle']].dxf.layer == SOURCE_LAYER for r in report['hidden_originals'])
    report.update(dwg_reopened=True, dwg_text_verified=True, dwg_sha256=sha(dwg),
                  roundtrip_sha256=sha(roundtrip), final_editable_text_count=len(now))
    (out/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TITLE_DWG_VERIFIED', len(now), 'editable strings', flush=True)


def preview(args, report):
    from ezdxf.addons.drawing import RenderContext, Frontend, config, layout, pymupdf
    from ezdxf.math import BoundingBox2d
    source = ezdxf.readfile(args.base)
    actual = ezdxf.readfile(args.output/'dwg_roundtrip.dxf')
    b = report['box_pdf']; h = report['page_height']; pad = (b[3]-b[1])*.3
    viewport = BoundingBox2d([((b[0]-pad)/72, (h-b[3]-pad)/72),
                              ((b[2]+pad)/72, (h-b[1]+pad)/72)])
    settings = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                                    color_policy=config.ColorPolicy.BLACK, lineweight_scaling=.6)
    for name, doc, handles in [('title_before.png', source, {r['handle'] for r in report['hidden_originals']}),
                               ('title_after.png', actual, {report['title_handle']})]:
        backend = pymupdf.PyMuPdfBackend()
        Frontend(RenderContext(doc), backend, config=settings).draw_layout(doc.modelspace(), finalize=True,
                   filter_func=lambda e: e.dxf.handle in handles)
        (args.output/name).write_bytes(backend.get_pixmap_bytes(layout.Page(390, 34, margins=layout.Margins.all(2)),
                                                              dpi=150, render_box=viewport))
    old = json.loads((args.geometry.parent/'recognition.json').read_text(encoding='utf-8'))
    before = old['results'][1]['fused_lines'][0]
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>제목 문자화 결과</title>
<style>body{font:17px system-ui;max-width:1500px;margin:28px auto;padding:12px;line-height:1.7;color:#223}img{max-width:100%;border:1px solid #ddd}code{background:#eee;padding:3px}</style>
<h1>제목 문자화 결과</h1><p>기존 인식: <code>'''+escape(before)+'''</code></p>
<p>개선 인식: <code>'''+escape(report['title'])+'''</code></p>
<p>로컬 PaddleOCR 재인식, 기존 YOLO 영문 결과, 모델과 원본 획이 함께 지지한 기호를 결합했습니다. 정답 문자열·사전·LLM 추론으로 바꿔 넣지 않았습니다.</p>
<h2>원래 제목 선</h2><img src="title_before.png"><h2>DWG에서 다시 읽어 렌더링한 편집 가능 문자</h2><img src="title_after.png">
<p><a href="'''+STEM+'''.dwg">DWG</a> · <a href="'''+STEM+'''.dxf">DXF</a> · <a href="conversion_report.json">검증 기록</a></p>
<p>원래 제목 선은 TITLE_SOURCE_OUTLINES 레이어에 보존했고 기본으로 껐습니다. 새 문자는 TITLE_OCR_TEXT 레이어입니다. 다른 문자 '''+str(report['original_text_count'])+'''개, 심볼 블록, 중앙선은 유지했습니다.</p>
<p>같은 제목에서 입력 크기를 바꾼 세 결과가 일치했습니다. 다른 도면의 정확도를 검증한 결과는 아닙니다. 글꼴 모양·간격은 원본과 달라집니다.</p></html>'''
    (args.output/'report.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recognition', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dwg', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    report = prepare(args)
    if args.dwg:
        convert(args, report)
        preview(args, report)


if __name__ == '__main__':
    main()
