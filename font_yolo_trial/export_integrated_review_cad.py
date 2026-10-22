"""Merge existing symbol geometry/centerlines with saved local OCR, without recognition.

No LLM, answer sheets, source strings, or older caption objects are used.
Original flat symbol geometry is retained on an off layer after geometric matching.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import ezdxf
from ezdxf.addons import Importer
from ezdxf.disassemble import recursive_decompose
from ezdxf.lldxf.tagwriter import TagCollector
from ezdxf.path import make_path
import numpy as np
from scipy.spatial import cKDTree

import export_fusion_review_cad as cad

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_integrated_local'
BASE=ROOT/'output/cad_with_tables_ocr'
SOURCE=ROOT/'green_centerline_trace/symbols_60_with_full_centerlines.dwg'
STEM='integrated_symbols_centerlines_yolo_ocr'
HIDDEN='SYMBOL_SOURCE_GEOMETRY'
KINDS={'LWPOLYLINE','ARC','CIRCLE','ELLIPSE'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def points(entity):
    return np.asarray([(p.x,p.y) for p in make_path(entity).flattening(.0001,segments=8)])


def sample(vertices,step):
    first,last=vertices[:-1],vertices[1:]
    counts=np.maximum(1,np.ceil(np.linalg.norm(last-first,axis=1)/step).astype(int))
    return np.concatenate([np.linspace(a,b,int(n)+1) for a,b,n in zip(first,last,counts)])


def normalized(value):
    if isinstance(value,float):return round(value,8)
    if isinstance(value,(str,int,bytes)):return value
    try:return tuple(normalized(v) for v in value)
    except TypeError:return str(value)


def tags(entity,ignore_layer=False):
    return [(t.code,normalized(t.value)) for t in TagCollector.dxftags(entity)
            if not (ignore_layer and t.code==8)]


def match_geometry(base,blocks):
    parts=list(recursive_decompose(blocks))
    assert len(parts)==11357 and all(e.dxftype() in KINDS for e in parts)
    bp=np.asarray([p[:2] for e in base.modelspace().query('LWPOLYLINE') for p in e.get_points('xy')])
    sp=np.asarray([p[:2] for e in parts if e.dxftype()=='LWPOLYLINE' for p in e.get_points('xy')])[::20]
    _,near=cKDTree(bp).query(sp)
    delta=bp[near]-sp
    close=(np.abs(delta[:,0])<1e-5)&(np.abs(delta[:,1])<.005)
    assert close.mean()>.8
    shift=np.median(delta[close],axis=0)
    shift[np.abs(shift)<1e-6]=0
    assert np.linalg.norm(shift)<.005
    print('GEOMETRY_ALIGNMENT',shift.tolist(),flush=True)
    edges=set()
    for entity in parts:
        entity.translate(float(shift[0]),float(shift[1]),0)
        v=points(entity)
        for first,last in zip(v,v[1:]):
            a,b=tuple(np.round(first,8)),tuple(np.round(last,8))
            if a!=b:edges.add(tuple(sorted((a,b))))
    source_samples=np.concatenate([sample(np.asarray(edge),.001) for edge in edges])
    tree=cKDTree(source_samples)
    matches=[]
    for entity in base.modelspace():
        if entity.dxftype() not in KINDS:continue
        v=points(entity)
        if len(v)<2:continue
        distances,_=tree.query(v,distance_upper_bound=.0012)
        if not np.all(np.isfinite(distances)):continue
        distances,_=tree.query(sample(v,.01),distance_upper_bound=.0012)
        if np.all(np.isfinite(distances)):
            matches.append(dict(handle=entity.dxf.handle,type=entity.dxftype(),
                layer=entity.dxf.layer,max_distance=float(distances.max())))
    assert len(matches)>11000
    assert all(not base.layers.get(r['layer']).is_off() for r in matches)
    target_samples=np.concatenate([sample(points(base.entitydb[r['handle']]),.001) for r in matches])
    reverse,_=cKDTree(target_samples).query(source_samples)
    coverage=float((reverse<=.0012).mean())
    print('SYMBOL_SOURCE_MATCH',len(matches),'reverse_coverage',coverage,flush=True)
    # The two PDF imports encode some curves differently, and some original
    # entities cross a block boundary. Keep all unmatched base geometry visible.
    # This reverse coverage is diagnostic; it is not a whole-drawing accuracy score.
    return shift,matches,dict(tolerance_cad_units=.0012,tolerance_pdf_points=.0864,
        source_sampling_step=.001,target_test_step=.01,reverse_sample_coverage=coverage,
        maximum_reverse_distance=float(reverse.max()),source_samples=len(source_samples),
        policy='Hide only whole base entities fully covered by source block geometry. Retain unmatched/partial originals; possible overlap remains.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cad.OUT=OUT
    dwg=OUT/(STEM+'.dwg');roundtrip=OUT/'dwg_roundtrip.dxf'
    if dwg.exists() or roundtrip.exists():raise FileExistsError('Preserve previous delivery')
    base_manifest=json.loads((BASE/'conversion_report.json').read_text(encoding='utf-8'))
    assert sha(BASE/'drawing_with_tables_yolo_ocr.dwg')==base_manifest['dwg_sha256']
    source_dxf=OUT/'symbol_centerline_source.dxf'
    if not source_dxf.exists():
        cad.run_cad(SOURCE,'extract_symbol_centerlines',[
            '_.DXFOUT',source_dxf.name,'_V','_2018','16','_.QUIT','_N'])
    source=ezdxf.readfile(source_dxf)
    doc=ezdxf.readfile(BASE/'dwg_roundtrip.dxf');model=doc.modelspace()
    baseline={e.dxf.handle:tags(e) for e in model}
    old_text={e.dxf.handle:e.dxf.text if e.dxftype()=='TEXT' else e.text for e in model.query('TEXT MTEXT')}
    assert len(old_text)==1091 and not list(model.query('INSERT'))
    blocks=[e for e in source.modelspace().query('INSERT') if e.dxf.name.startswith('SYM_')]
    centers=[e for e in source.modelspace() if e.dxf.layer.startswith('GREEN_CENTERLINE_')]
    assert len(blocks)==60 and len({e.dxf.name for e in blocks})==60
    assert Counter(e.dxf.layer for e in centers)=={'GREEN_CENTERLINE_CONFIRMED':12,'GREEN_CENTERLINE_INFERRED':1}
    shift,matched,matching=match_geometry(doc,blocks)
    doc.layers.new(HIDDEN,dxfattribs=dict(color=8));doc.layers.get(HIDDEN).off()
    doc.layers.new('SYMBOL_OBJECTS',dxfattribs=dict(color=7))
    for r in matched:model.doc.entitydb[r['handle']].dxf.layer=HIDDEN
    old_handles={e.dxf.handle for e in model}
    importer=Importer(source,doc)
    # Import only selected inserts and centerline entities. No older text/table objects.
    importer.import_entities(blocks+centers,target_layout=model);importer.finalize()
    inserted=[e for e in model if e.dxf.handle not in old_handles]
    new_blocks=[e for e in inserted if e.dxftype()=='INSERT']
    new_centers=[e for e in inserted if e.dxf.layer.startswith('GREEN_CENTERLINE_')]
    assert len(inserted)==73 and len(new_blocks)==60 and len(new_centers)==13
    for entity in new_blocks:
        entity.translate(float(shift[0]),float(shift[1]),0)
        entity.dxf.layer='SYMBOL_OBJECTS'
    for layer,color in [('GREEN_CENTERLINE_CONFIRMED',4),('GREEN_CENTERLINE_INFERRED',30)]:
        entry=doc.layers.get(layer);entry.on();entry.thaw();entry.dxf.color=color
        entry.dxf.discard('true_color')
    expected_blocks={e.dxf.name:dict(insert=list(e.dxf.insert),xscale=e.dxf.xscale,
        yscale=e.dxf.yscale,rotation=e.dxf.rotation) for e in new_blocks}
    expected_centers=[dict(layer=e.dxf.layer,points=[list(p) for p in e.get_points('xyb')]) for e in new_centers]
    assert all(e.dxftype()=='LWPOLYLINE' for e in new_centers)
    expected=len(baseline)+73
    assert len(model)==expected and len(model.query('TEXT MTEXT'))==1091
    doc.header['$INSUNITS']=0;doc.header['$LWDISPLAY']=False
    dxf=OUT/(STEM+'.dxf');doc.saveas(dxf)
    assert not ezdxf.readfile(dxf).audit().has_errors
    manifest=dict(inference_llm=False,new_recognition=False,answer_files_loaded=False,
        base_dwg=str((BASE/'drawing_with_tables_yolo_ocr.dwg').resolve()),
        base_dwg_sha256=base_manifest['dwg_sha256'],symbol_centerline_source=str(SOURCE.resolve()),
        symbol_centerline_source_sha256=sha(SOURCE),base_entities=len(baseline),
        existing_text_preserved=1091,table_text_preserved=880,symbol_blocks=60,
        imported_symbol_geometry_entities=11357,confirmed_centerlines=12,inferred_bridge=1,
        original_symbol_geometry_hidden=len(matched),symbol_translation_xy=shift.tolist(),
        matching=matching,block_references=expected_blocks,centerlines=expected_centers,
        original_symbol_entities=matched,units='PDFIMPORT coordinate units; no mm calibration.',
        limitations=['Existing OCR errors and spacing differences retained.',
            'One centerline bridge is inferred, distinguished by its layer and orange color.',
            'Matched original flat symbol geometry preserved on an off layer.',
            'Unmatched or partially overlapping base geometry remains visible to avoid deleting pipe/curve details. Some overlap may remain at symbol boundaries.'])
    (OUT/'integration_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('INTEGRATED_DXF_READY',expected,flush=True)
    cad.run_cad(dxf,'save_integrated',['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
    assert dwg.exists() and dwg.read_bytes()[:6]==b'AC1032'
    log=cad.run_cad(dwg,'verify_integrated',[
        '(setq ss (ssget "_X" (list (cons 410 "Model"))))',
        '(princ (strcat "\\nMODEL_ENTITIES=" (itoa (sslength ss))))',
        '(setq tx (ssget "_X" (list (cons 410 "Model") (cons 0 "TEXT,MTEXT"))))',
        '(princ (strcat "\\nEDITABLE_TEXT=" (itoa (sslength tx))))',
        '_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
    content=log.read_bytes().decode('utf-16-le',errors='replace')
    assert int(re.findall(r'MODEL_ENTITIES=(\d+)',content)[-1])==expected
    assert int(re.findall(r'EDITABLE_TEXT=(\d+)',content)[-1])==1091
    actual=ezdxf.readfile(roundtrip);space=actual.modelspace()
    assert not actual.audit().has_errors and len(space)==expected
    hidden={r['handle'] for r in matched}
    for handle,original in baseline.items():
        e=actual.entitydb[handle]
        if handle in hidden:
            assert e.dxf.layer==HIDDEN
            assert tags(e,True)==[(k,v) for k,v in original if k!=8]
        else:assert tags(e)==original,handle
    for handle,text in old_text.items():
        e=actual.entitydb[handle]
        assert (e.dxf.text if e.dxftype()=='TEXT' else e.text)==text
    saved_blocks={e.dxf.name:e for e in space.query('INSERT')}
    assert set(saved_blocks)==set(expected_blocks)
    for name,expected_block in expected_blocks.items():
        e=saved_blocks[name]
        assert np.allclose(e.dxf.insert,expected_block['insert'],rtol=0,atol=1e-9)
        assert all(abs(getattr(e.dxf,k)-expected_block[k])<1e-9 for k in ['xscale','yscale','rotation'])
        assert not actual.layers.get(e.dxf.layer).is_off()
    saved_centers=[e for e in space if e.dxf.layer.startswith('GREEN_CENTERLINE_')]
    for expected_center,e in zip(expected_centers,saved_centers):
        assert e.dxf.layer==expected_center['layer']
        assert np.allclose(e.get_points('xyb'),expected_center['points'],rtol=0,atol=1e-9)
    assert len(saved_centers)==13 and actual.layers.get(HIDDEN).is_off()
    assert not list(space.query('IMAGE'))
    nested=list(recursive_decompose(list(saved_blocks.values())))
    assert len(nested)==11357 and all(e.dxftype() in KINDS for e in nested)
    assert sha(BASE/'drawing_with_tables_yolo_ocr.dwg')==base_manifest['dwg_sha256']
    manifest.update(dwg_reopened=True,dxf_audit_errors=0,all_base_entities_preserved=True,
        all_existing_text_unchanged=True,all_60_blocks_reopened=True,all_13_centerlines_reopened=True,
        dwg_sha256=sha(dwg),output_dwg=str(dwg.resolve()))
    (OUT/'integration_report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('INTEGRATED_DWG_VERIFIED',dwg.name,flush=True)


if __name__=='__main__':main()
