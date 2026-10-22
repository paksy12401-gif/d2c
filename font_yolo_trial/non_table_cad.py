"""Keep PDFIMPORT vectors outside tables and attach uncorrected local OCR text."""
import csv
import json
from collections import Counter
from pathlib import Path

import ezdxf
from ezdxf import bbox
from ezdxf.enums import TextEntityAlignment

OUT = Path(__file__).resolve().parent / 'cad_without_tables'
HEIGHT = 2383.93994140625
TABLES = [[5045,170,7140,1250],[7990,70,11170,2230]]


def pdf_bounds(entity, cache):
    bounds=bbox.extents([entity],cache=cache)
    if not bounds.has_data:return None
    a,b=bounds.extmin,bounds.extmax
    return [a.x*72,HEIGHT-b.y*72,b.x*72,HEIGHT-a.y*72]


def in_table(box):
    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    return any(t[0]<=cx<=t[2] and t[1]<=cy<=t[3] and
               box[2]-box[0]<=t[2]-t[0]+2 and box[3]-box[1]<=t[3]-t[1]+2 for t in TABLES)


def prediction_layer(item, proposal):
    if item['kind']=='yolo':
        return 'YOLO_TEXT'
    return 'OCR_TEXT' if item['score']>=.98 and proposal['paddle_score']>=.7 else 'OCR_REVIEW'


def main():
    doc=ezdxf.readfile(OUT/'raw_import.dxf'); model=doc.modelspace();cache=bbox.Cache()
    doc.header['$INSUNITS']=0
    source_count=len(model); removed=[]
    for entity in list(model):
        box=pdf_bounds(entity,cache)
        if box and in_table(box):
            removed.append({'handle':entity.dxf.handle,'type':entity.dxftype(),'box_pdf':box})
            model.delete_entity(entity)
    assert len(model)>1000 and len(removed)>1000
    geometry_count=len(model)
    doc.saveas(OUT/'non_table_geometry.dxf')
    print('GEOMETRY',geometry_count,'EXCLUDED',len(removed),flush=True)
    ocr_path=OUT/'ocr_results.json'
    if not ocr_path.exists():return
    ocr=json.loads(ocr_path.read_text(encoding='utf-8'))
    assert ocr['llm_used'] is False and ocr['word_correction'] is False
    for name,color in [('YOLO_TEXT',3),('OCR_TEXT',6),('OCR_REVIEW',1),('OCR_ORIGINAL_OUTLINES',8)]:
        doc.layers.new(name,dxfattribs={'color':color})
    doc.layers.get('YOLO_TEXT').dxf.true_color=0x008230
    doc.styles.new('OCR_LATIN',dxfattribs={'font':'txt.shx'})
    doc.styles.new('OCR_KOREAN',dxfattribs={'font':'malgun.ttf'})
    original=[(e,pdf_bounds(e,cache)) for e in model]
    native_text=[b for e,b in original if b and e.dxftype() in ('TEXT','MTEXT')]
    records=[];hidden=set();counts=Counter()
    for item in ocr['detections']:
        box=item['box_pdf'];x0,y0,x1,y1=box
        if in_table(box):raise AssertionError('OCR inside excluded table')
        cx,cy=(x0+x1)/2,(y0+y1)/2
        if any(b[0]-1<=cx<=b[2]+1 and b[1]-1<=cy<=b[3]+1 for b in native_text):
            counts['already_native_text']+=1;continue
        proposal=ocr['proposals'][item['proposal']]
        layer=prediction_layer(item,proposal)
        entity=model.add_text(item['text'],dxfattribs={'layer':layer,'height':max(.005,(y1-y0)/72),
                            'style':'OCR_KOREAN' if item['kind']=='korean' else 'OCR_LATIN','color':256})
        entity.set_placement((x0/72,(HEIGHT-y1)/72,0),(x1/72,(HEIGHT-y1)/72,0),align=TextEntityAlignment.FIT)
        assigned=[]
        # YOLO predictions stay visible without erasing connector or wire geometry.
        if layer=='OCR_TEXT':
            for old,b in original:
                if not b or old.dxftype() not in ('LWPOLYLINE','ARC','ELLIPSE','CIRCLE'):continue
                if b[0]>=x0-.35 and b[1]>=y0-.35 and b[2]<=x1+.35 and b[3]<=y1+.35:
                    old.dxf.layer='OCR_ORIGINAL_OUTLINES';hidden.add(old.dxf.handle);assigned.append(old.dxf.handle)
        counts[layer]+=1
        records.append({**item,'layer':layer,'text_handle':entity.dxf.handle,'original_handles':assigned})
    doc.layers.get('OCR_REVIEW').off();doc.layers.get('OCR_ORIGINAL_OUTLINES').off()
    doc.header['$INSUNITS']=0
    doc.saveas(OUT/'non_table_yolo_ocr.dxf')
    reread=ezdxf.readfile(OUT/'non_table_yolo_ocr.dxf')
    assert not reread.audit().has_errors
    assert not reread.layers.get('YOLO_TEXT').is_off() and not reread.layers.get('YOLO_TEXT').is_frozen()
    assert sum(r['kind']=='yolo' for r in records)==len(reread.modelspace().query('TEXT[layer=="YOLO_TEXT"]'))
    assert all(abs(e.dxf.insert.z)<1e-9 for e in reread.modelspace().query('TEXT MTEXT'))
    assert not any(in_table(b) for e in reread.modelspace() if (b:=pdf_bounds(e,bbox.Cache())))
    summary={'llm_used':False,'dictionary_correction':False,'source_entities':source_count,'excluded_table_entities':len(removed),
             'preserved_non_table_entities':geometry_count,'hidden_original_outlines':len(hidden),'ocr_counts':dict(counts),
             'table_exclusions_pdf':TABLES,'pdf_to_cad_scale':1/72,'units':'PDFIMPORT units; no physical millimetre calibration',
             'auto_acceptance':'All saved YOLO detections at conf >=0.1 are visible; no Paddle agreement gate. Existing native text is not duplicated. Korean >=0.98 with proposal >=0.7.',
             'limitations':'YOLO_TEXT includes raw misdetections and overlays the preserved source geometry. Korean candidates below acceptance remain off on OCR_REVIEW.',
             'records':records,'removed_entities':removed}
    (OUT/'conversion_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'recognized_text.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['text','score','kind','layer','box_pdf','text_handle']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    print(json.dumps({k:summary[k] for k in ('source_entities','excluded_table_entities','preserved_non_table_entities','ocr_counts')}),flush=True)


if __name__=='__main__':
    assert in_table([5100,500,5110,510]) and not in_table([6100,1300,6200,1350])
    main()
