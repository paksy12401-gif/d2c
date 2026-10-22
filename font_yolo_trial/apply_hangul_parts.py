"""Resolve a narrow vowel ambiguity only when CTC candidates and source parts agree."""
import copy
import csv
import hashlib
import json
from pathlib import Path

from hangul_parts_probe import add_vertical_vowel
from iterate_fusion import area_coverage,fuse_words,lines

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output/cad_fusion_parts_applied'


def proposed_changes(record):
    changes=[];text=record['text']
    for index,char in enumerate(text):
        alternate=add_vertical_vowel(char)
        if not alternate:continue
        shapes=[g for g in record['geometry'] if g['index']==index]
        if len({g['height'] for g in shapes if g['present']})<3:continue
        evidence=[]
        for candidate in record['char_candidates']:
            decoded,info=candidate['raw']['rec_text']
            if len(decoded)!=len(text) or any(c!=text[i] for i,c in enumerate(decoded) if i!=index):continue
            if len(info)<6 or index>=len(info[5]):continue
            support=next((a['score'] for a in info[5][index] if a['text']==alternate),0.)
            if support>=.03:evidence.append(dict(height=candidate['height'],ctc_score=support))
        if len({e['height'] for e in evidence})>=2:
            changes.append(dict(index=index,before=char,after=alternate,evidence=evidence,
                                rule='horizontal_vowel_plus_source_right_stem_and_CTC_candidate_consensus'))
    return changes


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent/'output/cad_fusion_refined/combined_results.json'
    source_data=json.loads(source.read_text(encoding='utf-8'));data=copy.deepcopy(source_data)
    diagnostics_file=ROOT.parent/'output/cad_hangul_parts/results.json'
    diagnostics=json.loads(diagnostics_file.read_text(encoding='utf-8'))
    updates=[]
    for record in diagnostics['records']:
        changes=proposed_changes(record)
        if not changes:continue
        word=data['words'][record['field_id']][record['word_index']]
        if word['accepted']!=record['text']:raise ValueError('Stale Korean-part diagnostics')
        chars=list(word['accepted'])
        for change in changes:chars[change['index']]=change['after']
        before=word['accepted'];word['accepted']=''.join(chars);word['text']=word['accepted']
        word['part_verified_changes']=changes
        updates.append(dict(field_id=record['field_id'],word_index=record['word_index'],
                            before=before,after=word['accepted'],changes=changes))
    changed_fields={u['field_id'] for u in updates}
    for record in data['records']:
        if record['id'] not in changed_fields:continue
        words=data['words'][record['id']]
        # Preserve independently recognized geometric punctuation and Latin glyphs exactly.
        fused=[]
        for glyph in record['fused_glyphs']:
            replacement=next((w for w in words if 'part_verified_changes' in w and
                              area_coverage(glyph['box_pdf'],w['box_pdf'])>.8 and
                              area_coverage(w['box_pdf'],glyph['box_pdf'])>.8),None)
            fused.append({**glyph,'text':replacement['accepted'],'engine':'korean_ocr_with_vowel_parts'} if replacement else glyph)
        record['fused_glyphs']=fused;record['fused_lines']=lines(fused)
    detections=[]
    by_field={r['id']:r for r in data['records']}
    for g in source_data['detections']:
        if g.get('field_id') in changed_fields:continue
        detections.append(g)
    for identity in changed_fields:detections.extend({**g,'field_id':identity} for g in by_field[identity]['fused_glyphs'])
    data.update(detections=detections,part_updates=updates,
                baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                part_diagnostics_sha256=hashlib.sha256(diagnostics_file.read_bytes()).hexdigest(),
                inference_llm=False,answers_loaded=False,dictionary_correction=False,
                word_correction='Only recorded source-stroke + CTC-supported vowel disambiguation; no dictionary or expected strings.')
    (OUT/'combined_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'recognized_lines.txt').write_text('\n'.join(f"{r['id']}: {' / '.join(r['fused_lines'])}" for r in data['records']),encoding='utf-8')
    with (OUT/'recognized_text.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['field_id','text','score','engine','x0','y0','x1','y1'])
        for g in detections:writer.writerow([g.get('field_id',''),g['text'],g['score'],g['engine'],*g['box_pdf']])
    print(json.dumps(updates,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
