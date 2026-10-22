"""Export automatically selected large text to a copy of the corresponding CAD."""
import argparse
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment

import export_fusion_review_cad as cad
from export_integrated_review_cad import sha, normalized
from export_title_review_cad import texts

STEM = 'integrated_auto_large_text'
TEXT_LAYER = 'AUTO_LARGE_TEXT'
SOURCE_LAYER = 'AUTO_LARGE_SOURCE'


def geometry(entity):
    keys = ('start', 'end', 'center', 'radius', 'major_axis', 'ratio', 'start_param', 'end_param',
            'insert', 'align_point', 'height', 'width', 'rotation', 'name', 'xscale', 'yscale', 'zscale')
    result = {key: normalized(getattr(entity.dxf, key)) for key in keys if entity.dxf.is_supported(key)}
    if entity.dxftype() == 'LWPOLYLINE':
        result.update(vertices=normalized(entity.get_points()), closed=entity.closed)
    return result


def prepare(args):
    manifest = args.output/'conversion_report.json'
    run = json.loads(args.run.read_text(encoding='utf-8'))
    if run['manual_rois'] is not False or run['inference_llm'] is not False or run['answers_loaded'] is not False:
        raise ValueError('Unexpected discovery/recognition source')
    base = Path(run['base_dxf'])
    assert sha(base) == run['base_dxf_sha256']
    assert sha(Path(run['source_pdf'])) == run['source_sha256']
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        assert sha(args.run) == data['automatic_run_sha256']
        assert sha(args.output/(STEM+'.dxf')) == data['dxf_sha256']
        return data
    accepted = [r for r in run['routes'] if r['status'] == 'accepted']
    if not accepted:
        raise ValueError('No new agreed text to export; preserve the existing CAD')
    fields = {r['field']['id']: [s for g in r['field']['groups'] for s in g['segments']] for r in accepted}
    index = cad.StrokeIndex(fields)
    doc = ezdxf.readfile(base); model = doc.modelspace(); original_count = len(model)
    found = []; matched = defaultdict(set)
    for e in model:
        if e.dxftype() == 'LWPOLYLINE':
            vertices = list(e.get_points('xyb'))
            if any(abs(p[2]) > 1e-10 for p in vertices):
                continue
            points = [(p[0]*72, run['page_height']-p[1]*72) for p in vertices]
            identity, edges = index.match(points, e.closed)
        elif e.dxftype() == 'LINE':
            points = [(p.x*72, run['page_height']-p.y*72) for p in (e.dxf.start, e.dxf.end)]
            identity, edges = index.match(points)
        else:
            continue
        if identity:
            found.append(dict(handle=e.dxf.handle, field_id=identity, previous_layer=e.dxf.layer))
            matched[identity].update(edges)
    for identity, segments in fields.items():
        expected = set(cad.StrokeIndex({identity:segments}).edges)
        if matched[identity] != expected:
            raise ValueError(f'{identity}: incomplete source ownership; do not hide unrelated geometry')
    for name, color in ((TEXT_LAYER, 6), (SOURCE_LAYER, 8)):
        if name in doc.layers:
            raise ValueError('Automatic large-text layers already exist')
        doc.layers.new(name, dxfattribs={'color':color})
    for item in found:
        e = doc.entitydb[item['handle']]; e.dxf.layer = SOURCE_LAYER; e.dxf.color = 256
        e.dxf.discard('true_color')
    doc.layers.get(SOURCE_LAYER).off()
    if 'AUTO_LARGE_LOCAL_FONT' not in doc.styles:
        doc.styles.new('AUTO_LARGE_LOCAL_FONT', dxfattribs={'font':'malgun.ttf'})
    if 'CAD_LARGE_LOCAL' not in doc.appids:
        doc.appids.new('CAD_LARGE_LOCAL')
    added = []
    for route in accepted:
        f = route['field']; b = f['box']; text = route['recognized_text']
        evidence = json.loads(Path(route['recognition']).read_text(encoding='utf-8'))
        assert evidence['llm_inference'] is False and evidence['answers_loaded'] is False
        assert all(r['fused_lines'] == [text] for r in evidence['results'])
        e = model.add_text(text, dxfattribs=dict(layer=TEXT_LAYER, style='AUTO_LARGE_LOCAL_FONT', height=f['cap']/72))
        e.set_placement((b[0]/72, (run['page_height']-f['baseline'])/72),
                        (b[2]/72, (run['page_height']-f['baseline'])/72), align=TextEntityAlignment.FIT)
        e.set_xdata('CAD_LARGE_LOCAL', [(1000, f['id'])])
        added.append(dict(field_id=f['id'], text=text, handle=e.dxf.handle, box_pdf=b,
                          relative_height=f['relative_height'], recognition_sha256=sha(Path(route['recognition']))))
    dxf = args.output/(STEM+'.dxf'); doc.saveas(dxf)
    reread = ezdxf.readfile(dxf)
    assert len(reread.modelspace()) == original_count+len(added)
    assert not reread.audit().has_errors
    report = dict(inference_llm=False, manual_rois=False, answers_used=False,
                  automatic_run=str(args.run.resolve()), automatic_run_sha256=sha(args.run),
                  base=str(base), base_sha256=sha(base), dxf_sha256=sha(dxf),
                  original_entities=original_count, original_editable_text=len(texts(model))-len(added),
                  size_classification=run['size_classification'], candidate_count=run['candidate_count'],
                  routing_counts=dict(Counter(r['status'] for r in run['routes'])),
                  added=added, hidden_originals=found,
                  matched_source_edges=sum(len(v) for v in matched.values()),
                  source_edges=len(index.edges), page_height=run['page_height'],
                  limitations=run['limits'])
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('AUTO_DXF_READY', json.dumps(dict(added=len(added), hidden=len(found), edges=report['source_edges']), ensure_ascii=True), flush=True)
    return report


def verify(args, report, dxf):
    before = ezdxf.readfile(report['base']); after = ezdxf.readfile(dxf)
    old = before.modelspace(); new = after.modelspace()
    assert not after.audit().has_errors
    assert len(new) == len(old)+len(report['added'])
    old_text = texts(old); new_text = texts(new)
    assert len(new_text) == len(old_text)+len(report['added'])
    assert all(new_text[h] == text for h, text in old_text.items())
    assert all(new_text[r['handle']] == r['text'] for r in report['added'])
    assert Counter(e.dxf.name for e in old.query('INSERT')) == Counter(e.dxf.name for e in new.query('INSERT'))
    assert Counter(e.dxf.layer for e in old if e.dxf.layer.startswith('GREEN_CENTERLINE_')) == Counter(e.dxf.layer for e in new if e.dxf.layer.startswith('GREEN_CENTERLINE_'))
    hidden = {r['handle'] for r in report['hidden_originals']}
    for e in old:
        actual = after.entitydb[e.dxf.handle]
        assert geometry(actual) == geometry(e), f'Changed source geometry {e.dxf.handle}'
        assert actual.dxf.layer == (SOURCE_LAYER if e.dxf.handle in hidden else e.dxf.layer)
    assert after.layers.get(SOURCE_LAYER).is_off() and not after.layers.get(TEXT_LAYER).is_off()
    return dict(original_geometry_verified=len(old), editable_text_verified=len(new_text),
                symbols=len(new.query('INSERT')), centerlines=sum(1 for e in new if e.dxf.layer.startswith('GREEN_CENTERLINE_')))


def write_report(args, report):
    from ezdxf.addons.drawing import RenderContext, Frontend, config, layout, pymupdf
    from ezdxf.math import BoundingBox2d
    doc = ezdxf.readfile(args.output/'dwg_roundtrip.dxf')
    settings = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                                    color_policy=config.ColorPolicy.BLACK, lineweight_scaling=.6)
    rows = []
    run = json.loads(args.run.read_text(encoding='utf-8'))
    (args.output/'automatic_results.json').write_bytes(args.run.read_bytes())
    routing_rows = ''.join('<tr><td>'+escape(r['field']['id'])+'</td><td>'+str(round(r['field']['cap'], 2))+
                         '</td><td>'+str(round(r['field']['relative_height'], 2))+'</td><td>'+
                         ('기존 문자 보존' if r['status']=='existing_editable_text' else
                          '추가 문자화' if r['status']=='accepted' else '검토 필요')+'</td></tr>' for r in run['routes'])
    for i, item in enumerate(report['added']):
        b = item['box_pdf']; h = report['page_height']; pad = (b[3]-b[1])*.3
        backend = pymupdf.PyMuPdfBackend()
        Frontend(RenderContext(doc), backend, config=settings).draw_layout(doc.modelspace(), finalize=True,
                      filter_func=lambda e:e.dxf.handle == item['handle'])
        image = f'large_text_{i}.png'
        (args.output/image).write_bytes(backend.get_pixmap_bytes(layout.Page(390, 34, margins=layout.Margins.all(2)), dpi=150,
                    render_box=BoundingBox2d([((b[0]-pad)/72, (h-b[3]-pad)/72), ((b[2]+pad)/72, (h-b[1]+pad)/72)])))
        rows.append('<p>'+escape(item['field_id'])+' · '+str(round(item['relative_height'], 2))+'배 · '+escape(item['text'])+'</p><img src="'+image+'">')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>큰 글씨 자동 처리 DWG</title>
<style>body{font:17px system-ui;line-height:1.7;max-width:1500px;margin:28px auto;padding:12px}img{max-width:100%}code{background:#eee;padding:3px}</style>
<h1>큰 글씨 자동 처리 DWG</h1><p>PDF 전체의 선형 벡터 문자 후보를 탐색하고, 도면에서 가장 많이 나타나는 크기 군집의 중앙값을 기준으로 큰 글씨를 분류했습니다. 제목 좌표·정답 문구·표 제외 좌표를 입력하지 않았습니다.</p>
<p>후보 '''+str(report['candidate_count'])+'''곳 중 큰 글씨 '''+str(sum(report['routing_counts'].values()))+'''곳을 분류했습니다. 이미 편집 가능한 문자는 위치·높이·식별자를 확인해 중복 생성을 피했습니다.</p>
<p>크기 분포: <code>'''+escape(json.dumps(report['size_classification'], ensure_ascii=False))+'''</code></p>
<table border="1" cellpadding="8"><tr><th>자동 영역 ID</th><th>글자 높이(PDF pt)</th><th>일반 글씨 대비</th><th>처리</th></tr>'''+routing_rows+'''</table>
<p><a href="'''+STEM+'''.dwg">DWG 받기</a> · <a href="'''+STEM+'''.dxf">DXF</a> · <a href="conversion_report.json">검증 기록</a> · <a href="automatic_results.json">자동 탐색 기록</a> · <a href="사용법.txt">다른 입력으로 실행</a></p>'''+''.join(rows)+'''
<p>원래 제목 선은 AUTO_LARGE_SOURCE 레이어에 보존했고 기본으로 껐습니다. 추가 문자는 AUTO_LARGE_TEXT 레이어입니다. 로컬 YOLO·PaddleOCR만으로 처리했고 정답 문자열을 넣지 않았습니다.</p>
<p>현재 자동 탐색은 선형 벡터 경로를 대상으로 합니다. 이미지로만 된 글자, 곡선 경로, 원본에서 글자별 경로가 흩어진 경우의 전체 탐색은 보장하지 않습니다. 다른 도면 인식 정확도는 아직 검증하지 않았습니다.</p></html>'''
    (args.output/'report.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dwg', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    report = prepare(args)
    report['dxf_verification'] = verify(args, report, args.output/(STEM+'.dxf'))
    (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.dwg:
        cad.OUT = args.output; dwg = args.output/(STEM+'.dwg'); roundtrip = args.output/'dwg_roundtrip.dxf'
        if dwg.exists() or roundtrip.exists():
            raise FileExistsError('Preserve previous DWG verification')
        cad.run_cad(args.output/(STEM+'.dxf'), 'save_auto', ['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
        cad.run_cad(dwg, 'verify_auto', ['_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
        report.update(dwg_sha256=sha(dwg), dwg_reopened=True, verification=verify(args, report, roundtrip))
        (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        write_report(args, report)
        print('AUTO_DWG_VERIFIED', json.dumps(report['verification']), flush=True)


if __name__ == '__main__':
    main()
