"""Evaluate frozen part diagnostics separately from answer-free inference."""
import json
from collections import defaultdict
from pathlib import Path

from evaluate_real_validation import alignment
from iterate_fusion import owned_by_field,lines

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output/cad_stroke_templates'


def main():
    raw=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    diagnostics={(r['method'],r['index']):r for r in raw['records']}
    evaluation=json.loads((ROOT.parent/'output/cad_fusion_refined/evaluation.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    new=json.loads((ROOT.parent/'output/cad_fusion_refined/combined_results.json').read_text(encoding='utf-8'))
    fields=[r for r in evaluation['rows'] if r['method']=='refined']
    rows=[]
    for field in fields:
        for method,data in [('original',old),('refined',new)]:
            selected=[(i,g) for i,g in enumerate(data['detections']) if owned_by_field(g,field)]
            for policy,key in [('unchanged',None),('distance_only','threshold_review'),('conflict','conflict_review')]:
                reviewed=[(i,g) for i,g in selected if key and diagnostics.get((method,i),{}).get(key)]
                kept=[g for i,g in selected if (i,g) not in reviewed]
                text='\n'.join(lines(kept))
                rows.append(dict(group=field['group'],id=field['id'],kind=field['kind'],image=field['image'],
                    method=method,policy=policy,reference=field['reference'],prediction=text,boxes=len(kept),
                    flagged=len(reviewed),flagged_text=''.join(g['text'] for _,g in reviewed),
                    score=alignment(field['reference'],text,field['ambiguities'])))
    grouped=defaultdict(list)
    for r in rows:grouped[r['group'],r['kind'],r['method'],r['policy']].append(r)
    summaries=[]
    for (group,kind,method,policy),items in grouped.items():
        summaries.append(dict(group=group,kind=kind,method=method,policy=policy,
            exact=sum(r['score']['exact'] for r in items),regions=len(items),
            characters=sum(r['score']['reference_characters'] for r in items),
            errors=sum(r['score']['edit_distance'] for r in items),
            flagged=sum(r['flagged'] for r in items),remaining_boxes=sum(r['boxes'] for r in items)))
    result=dict(scope='Same-PDF exploratory postfilter ablation. Removing review candidates is simulated, not deployed; errors count dropped correct characters.',
                groups_overlap=True,answers_used_only_for_evaluation=True,summaries=summaries,rows=rows)
    (OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in summaries:
        if r['group']=='fixed_development' and r['kind'] in ('outlined_latin','drawing_negative'):
            print(json.dumps(r,ensure_ascii=True),flush=True)
    print('CROP_INVALID',sum(not r.get('crop_valid',False) for r in raw['records']),flush=True)


if __name__=='__main__':main()
