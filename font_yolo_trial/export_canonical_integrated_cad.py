"""Use one CAD geometry source for symbols, centerlines and saved local OCR.

The earlier symbol DWG supplies ownership templates and centerlines only. Symbol
curves/vertices are copied exclusively from the latest OCR drawing. No recognition,
LLM calls or answer files are involved. Original geometry remains on an off layer.
"""
from collections import Counter
import json
from pathlib import Path
import re

import ezdxf
from ezdxf.disassemble import recursive_decompose
import numpy as np
from scipy.spatial import cKDTree

import export_fusion_review_cad as cad
from export_integrated_review_cad import sha,points,sample,tags,KINDS

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_integrated_local_fixed'
BASE=ROOT/'output/cad_with_tables_ocr'
TEMPLATES=ROOT/'output/cad_integrated_local/symbol_centerline_source.dxf'
STEM='integrated_symbols_centerlines_yolo_ocr'
HIDDEN='SYMBOL_SOURCE_GEOMETRY'


def same_geometry(first,second):
    assert first.dxftype()==second.dxftype()
    a,b=points(first),points(second)
    if a.shape==b.shape and np.allclose(a,b,rtol=0,atol=1e-7):return
    assert cKDTree(a).query(b)[0].max()<1e-7
    assert cKDTree(b).query(a)[0].max()<1e-7


def make_owner_index(base,templates):
    bp=np.asarray([p[:2] for e in base.modelspace().query('LWPOLYLINE') for p in e.get_points('xy')])
    source_parts=list(recursive_decompose(templates))
    sp=np.asarray([p[:2] for e in source_parts if e.dxftype()=='LWPOLYLINE' for p in e.get_points('xy')])[::20]
    _,near=cKDTree(bp).query(sp);delta=bp[near]-sp
    mask=(np.abs(delta[:,0])<1e-5)&(np.abs(delta[:,1])<.005)
    assert mask.mean()>.8
    shift=np.median(delta[mask],axis=0);shift[np.abs(shift)<1e-6]=0
    arrays=[];labels=[]
    for i,insert in enumerate(templates):
        rows=[]
        for e in recursive_decompose([insert]):
            assert e.dxftype() in KINDS
            e.translate(float(shift[0]),float(shift[1]),0)
            rows.append(sample(points(e),.002))
        v=np.concatenate(rows);arrays.append(v);labels.append(np.full(len(v),i,np.int16))
    data=np.concatenate(arrays);ownership=np.concatenate(labels);tree=cKDTree(data)
    def owner(entity):
        v=points(entity)
        if len(v)<2:return None
        d,j=tree.query(v,distance_upper_bound=.03)
        if not np.all(np.isfinite(d)):return None
        d,j=tree.query(sample(v,.01),distance_upper_bound=.03)
        if not np.all(np.isfinite(d)):return None
        result=ownership[j]
        if not np.all(result==result[0]):return None
        return int(result[0])
    return owner,tree,shift


def main():
    OUT.mkdir(parents=True,exist_ok=True);cad.OUT=OUT
    dwg=OUT/(STEM+'.dwg');roundtrip=OUT/'dwg_roundtrip.dxf'
    if dwg.exists() or roundtrip.exists():raise FileExistsError('Preserve previous output')
    base_report=json.loads((BASE/'conversion_report.json').read_text(encoding='utf-8'))
    assert sha(BASE/'drawing_with_tables_yolo_ocr.dwg')==base_report['dwg_sha256']
    doc=ezdxf.readfile(BASE/'dwg_roundtrip.dxf');model=doc.modelspace()
    original={e.dxf.handle:tags(e) for e in model}
    original_text={e.dxf.handle:e.dxf.text if e.dxftype()=='TEXT' else e.text for e in model.query('TEXT MTEXT')}
    assert len(original_text)==1091
    source=ezdxf.readfile(TEMPLATES)
    templates=[e for e in source.modelspace().query('INSERT') if e.dxf.name.startswith('SYM_')]
    centers=[e for e in source.modelspace() if e.dxf.layer.startswith('GREEN_CENTERLINE_')]
    assert len(templates)==60 and len(centers)==13
    owner,tree,shift=make_owner_index(doc,templates)
    print('CANONICAL_OWNERSHIP_READY',shift.tolist(),flush=True)
    doc.layers.new(HIDDEN,dxfattribs=dict(color=8));doc.layers.get(HIDDEN).off()
    doc.layers.new('SYMBOL_OBJECTS',dxfattribs=dict(color=7))
    origins=[];definitions=[];inserts=[];counts=Counter()
    for template in templates:
        origin=np.asarray(template.dxf.insert,dtype=float);origin[:2]+=shift
        name=template.dxf.name
        assert name not in doc.blocks
        definitions.append(doc.blocks.new(name));origins.append(origin)
        inserts.append(model.add_blockref(name,origin,dxfattribs=dict(layer='SYMBOL_OBJECTS')))
    def attach(entity,which):
        if which is None:model.add_entity(entity)
        else:
            entity.translate(*(-origins[which]));definitions[which].add_entity(entity);counts[which]+=1
        return dict(handle=entity.dxf.handle,owner=which)
    moved=[]
    for e in list(model):
        if e.dxf.handle not in original or e.dxftype() not in KINDS:continue
        if doc.layers.get(e.dxf.layer).is_off():continue
        v=points(e)
        if len(v)<2 or v[:,0].min()>70:continue
        which=owner(e)
        if which is not None:
            layer=e.dxf.layer;created=attach(e.copy(),which)
            moved.append(dict(handle=e.dxf.handle,mode='whole',layer=layer,created=[created]))
            e.dxf.layer=HIDDEN
        elif e.dxftype()=='LWPOLYLINE':
            proximity,_=tree.query(v,distance_upper_bound=.03)
            if not np.any(np.isfinite(proximity)):continue
            pieces=list(e.virtual_entities());owners=[owner(p) for p in pieces]
            if not any(x is not None for x in owners):continue
            # Only zero-width polylines are decomposed. LINE/ARC primitives preserve
            # every original segment and bulge exactly; curves are not re-fitted.
            assert not e.dxf.get('const_width',0)
            assert all(abs(p[2])+abs(p[3])<1e-12 for p in e.get_points('xyseb'))
            layer=e.dxf.layer;created=[attach(p,o) for p,o in zip(pieces,owners)]
            moved.append(dict(handle=e.dxf.handle,mode='split',layer=layer,created=created))
            e.dxf.layer=HIDDEN
    assert len(counts)==60 and min(counts.values())>0
    print('CANONICAL_SYMBOLS',sum(counts.values()),'objects in 60 blocks',
        Counter(r['mode'] for r in moved),flush=True)
    center_records=[]
    for layer,color in [('GREEN_CENTERLINE_CONFIRMED',4),('GREEN_CENTERLINE_INFERRED',30)]:
        assert layer not in doc.layers
        doc.layers.new(layer,dxfattribs=dict(color=color))
    for center in centers:
        e=center.copy();model.add_foreign_entity(e,copy=False)
        center_records.append(dict(handle=e.dxf.handle,layer=e.dxf.layer,points=[list(p) for p in e.get_points('xyb')]))
    doc.header['$INSUNITS']=0;doc.header['$LWDISPLAY']=False
    expected_count=len(model)
    dxf=OUT/(STEM+'.dxf');doc.saveas(dxf)
    assert not ezdxf.readfile(dxf).audit().has_errors
    report=dict(inference_llm=False,new_recognition=False,answer_files_loaded=False,
        canonical_geometry_source=str((BASE/'dwg_roundtrip.dxf').resolve()),
        base_dwg_sha256=base_report['dwg_sha256'],base_entities=len(original),
        source_template_dxf_sha256=sha(TEMPLATES),symbol_template_alignment_xy=shift.tolist(),
        imported_old_symbol_geometry=0,symbols=60,confirmed_centerlines=12,inferred_centerlines=1,
        existing_editable_text=1091,table_text=880,block_entity_counts={templates[i].dxf.name:n for i,n in counts.items()},
        original_symbol_objects_hidden=len(moved),split_originals=sum(r['mode']=='split' for r in moved),
        mapping=moved,centerlines=center_records,expected_model_entities=expected_count,
        method='One canonical CAD geometry. Old symbol blocks guide ownership only. Copy original entities or exact LINE/ARC primitives into blocks; each original has exactly one visible replacement.',
        limits=['Existing OCR mistakes/spacing retained. One centerline connection is inferred.',
            'Block membership transferred geometrically from previous symbol grouping, not re-detected.',
            'PDFIMPORT units retained; no mm calibration.'])
    (OUT/'integration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cad.run_cad(dxf,'save_integrated',['_.ZOOM','_E','_.SAVEAS','_2018',dwg.name,'_.QUIT','_N'])
    assert dwg.exists() and dwg.read_bytes()[:6]==b'AC1032'
    log=cad.run_cad(dwg,'verify_integrated',[
        '(setq ss (ssget "_X" (list (cons 410 "Model"))))',
        '(princ (strcat "\\nMODEL_ENTITIES=" (itoa (sslength ss))))',
        '(setq tx (ssget "_X" (list (cons 410 "Model") (cons 0 "TEXT,MTEXT"))))',
        '(princ (strcat "\\nEDITABLE_TEXT=" (itoa (sslength tx))))',
        '_.AUDIT','_N','_.DXFOUT',roundtrip.name,'_V','_2018','16','_.QUIT','_N'])
    content=log.read_bytes().decode('utf-16-le',errors='replace')
    assert int(re.findall(r'MODEL_ENTITIES=(\d+)',content)[-1])==expected_count
    assert int(re.findall(r'EDITABLE_TEXT=(\d+)',content)[-1])==1091
    actual=ezdxf.readfile(roundtrip);space=actual.modelspace()
    assert not actual.audit().has_errors and len(space)==expected_count
    by_original={r['handle']:r for r in moved}
    for handle,initial in original.items():
        e=actual.entitydb[handle]
        if handle in by_original:
            assert e.dxf.layer==HIDDEN
            assert tags(e,True)==[(k,v) for k,v in initial if k!=8]
        else:assert tags(e)==initial,handle
    for handle,text in original_text.items():
        e=actual.entitydb[handle]
        assert (e.dxf.text if e.dxftype()=='TEXT' else e.text)==text
    saved_blocks={e.dxf.name:e for e in space.query('INSERT')}
    assert len(saved_blocks)==60 and set(saved_blocks)=={t.dxf.name for t in templates}
    checked=0;replacement_handles=set()
    for record in moved:
        e=actual.entitydb[record['handle']]
        expected=[e] if record['mode']=='whole' else list(e.virtual_entities())
        assert len(expected)==len(record['created'])
        for original_piece,created in zip(expected,record['created']):
            assert created['handle'] not in replacement_handles
            replacement_handles.add(created['handle'])
            saved=actual.entitydb[created['handle']].copy()
            assert saved.dxf.layer==record['layer'] and not actual.layers.get(saved.dxf.layer).is_off()
            if created['owner'] is not None:
                ins=saved_blocks[templates[created['owner']].dxf.name]
                assert ins.dxf.xscale==ins.dxf.yscale==ins.dxf.zscale==1 and ins.dxf.rotation==0
                saved.transform(ins.matrix44())
            same_geometry(original_piece,saved);checked+=1
    for center in center_records:
        e=actual.entitydb[center['handle']]
        assert e.dxf.layer==center['layer']
        assert np.allclose(e.get_points('xyb'),center['points'],rtol=0,atol=1e-9)
    assert actual.layers.get(HIDDEN).is_off() and not list(space.query('IMAGE'))
    assert len(space.query('TEXT MTEXT'))==1091
    assert len(replacement_handles)==sum(len(r['created']) for r in moved)
    assert sha(BASE/'drawing_with_tables_yolo_ocr.dwg')==base_report['dwg_sha256']
    report.update(dwg_reopened=True,dxf_audit_errors=0,all_base_entities_preserved=True,
        all_text_unchanged=True,exact_geometry_replacements_verified=checked,
        duplicate_original_symbol_geometry_visible=False,all_60_blocks_reopened=True,
        all_13_centerlines_reopened=True,dwg_sha256=sha(dwg),output_dwg=str(dwg.resolve()))
    (OUT/'integration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CANONICAL_DWG_VERIFIED',checked,'exact replacements',dwg.name,flush=True)


if __name__=='__main__':main()
