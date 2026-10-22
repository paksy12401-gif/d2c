"""Refresh only generated centerlines from a saved source trace; preserve other CAD objects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import ezdxf
import numpy as np

import export_fusion_review_cad as cad
from export_auto_large_cad import geometry
from export_integrated_review_cad import sha
from export_title_review_cad import texts

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trace_green_centerline import pdf_points

STEM = 'integrated_centerlines_corrected'
PREFIX = 'GREEN_CENTERLINE_'


def prepare(args):
    out = args.output; manifest = out/'conversion_report.json'
    if manifest.exists():
        report = json.loads(manifest.read_text(encoding='utf-8'))
        assert sha(args.base) == report['base_sha256'] and sha(args.trace) == report['trace_sha256']
        return report
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    doc = ezdxf.readfile(args.base); model = doc.modelspace()
    old_count = len(model); old_centers = [e for e in model if e.dxf.layer.startswith(PREFIX)]
    unused = {e.dxf.handle:e for e in old_centers}; matched = []; added = []
    for status in ('confirmed', 'inferred'):
        layer = PREFIX+status.upper()
        if layer not in doc.layers:
            doc.layers.new(layer)
        doc.layers.get(layer).dxf.color = 4 if status == 'confirmed' else 30
        doc.layers.get(layer).dxf.discard('true_color')
        for row in trace[status]:
            points = np.array([(x/72, (trace['page_height']-y)/72) for x, y in pdf_points(row)])
            same = next((e for e in unused.values() if e.dxftype()=='LWPOLYLINE' and
                        len(e.get_points('xy')) == 2 and
                        (np.allclose(e.get_points('xy'), points, rtol=0, atol=1e-7) or
                         np.allclose(e.get_points('xy'), points[::-1], rtol=0, atol=1e-7))), None)
            if same is not None:
                same.dxf.layer = layer; same.dxf.color = 256; same.dxf.discard('true_color')
                matched.append(same.dxf.handle); del unused[same.dxf.handle]
            else:
                e = model.add_lwpolyline(points, dxfattribs={'layer':layer, 'color':256})
                added.append(dict(handle=e.dxf.handle, source=row, layer=layer))
    removed = []
    for e in unused.values():
        removed.append(dict(handle=e.dxf.handle, layer=e.dxf.layer, points=list(e.get_points('xy'))))
        model.delete_entity(e)
    dxf = out/(STEM+'.dxf'); doc.saveas(dxf)
    report = dict(base=str(args.base.resolve()), base_sha256=sha(args.base), trace_sha256=sha(args.trace),
                  inference_llm=False, original_entities=old_count, original_centerlines=len(old_centers),
                  original_editable_text=len(texts(model)), kept_centerlines=matched,
                  removed_generated_centerlines=removed, added_centerlines=added,
                  confirmed=len(trace['confirmed']), inferred=len(trace['inferred']),
                  reason='Green-dominant yellow-green source strokes were excluded by the old pure-green predicate.',
                  dxf_sha256=sha(dxf))
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def verify(args, report, path):
    old = ezdxf.readfile(args.base); new = ezdxf.readfile(path)
    old_space = old.modelspace(); model = new.modelspace()
    removed = {r['handle'] for r in report['removed_generated_centerlines']}
    assert not new.audit().has_errors
    assert len(model) == len(old_space)-len(removed)+len(report['added_centerlines'])
    assert texts(old_space) == texts(model)
    assert Counter(e.dxf.name for e in old_space.query('INSERT')) == Counter(e.dxf.name for e in model.query('INSERT'))
    preserved = 0
    for e in old_space:
        if e.dxf.handle in removed:
            continue
        actual = new.entitydb[e.dxf.handle]
        assert geometry(e) == geometry(actual)
        if not e.dxf.layer.startswith(PREFIX):
            assert actual.dxf.layer == e.dxf.layer and actual.dxf.color == e.dxf.color
            preserved += 1
    centers = [e for e in model if e.dxf.layer.startswith(PREFIX)]
    assert len(centers) == report['confirmed']+report['inferred']
    assert sum(e.dxf.layer == PREFIX+'CONFIRMED' for e in centers) == report['confirmed']
    for item in report['added_centerlines']:
        e = new.entitydb[item['handle']]
        source = json.loads(args.trace.read_text(encoding='utf-8'))
        expected = [(x/72, (source['page_height']-y)/72) for x, y in pdf_points(item['source'])]
        np.testing.assert_allclose(e.get_points('xy'), expected, rtol=0, atol=1e-7)
    assert new.layers.get(PREFIX+'CONFIRMED').dxf.color == 4
    return dict(non_centerline_entities_preserved=preserved, editable_text=len(texts(model)),
                symbols=len(model.query('INSERT')), confirmed_centerlines=report['confirmed'],
                inferred_centerlines=report['inferred'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dwg', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    report = prepare(args)
    report['verification'] = verify(args, report, args.output/(STEM+'.dxf'))
    (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('CENTERLINE_DXF_VERIFIED', json.dumps(report['verification']), flush=True)
    if args.dwg:
        cad.OUT = args.output; dwg = args.output/(STEM+'.dwg'); roundtrip = args.output/'dwg_roundtrip.dxf'
        if dwg.exists() or roundtrip.exists():
            raise FileExistsError('Preserve previous DWG')
        cad.run_cad(args.output/(STEM+'.dxf'), 'save_fixed', ['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
        cad.run_cad(dwg, 'verify_fixed', ['_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
        report.update(dwg_reopened=True, dwg_sha256=sha(dwg), verification=verify(args, report, roundtrip))
        (args.output/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('CENTERLINE_DWG_VERIFIED', json.dumps(report['verification']), flush=True)


if __name__ == '__main__':
    main()
