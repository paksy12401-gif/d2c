"""Use server OCR context only when the actual font shape supports the change."""
import json

from refine_table_ocr import BASE,ROOT,compact,index_map
from table_confusion_guard import repair_rows

OUT=ROOT/'output/cad_table_font_training'


def contextual_character(old,a,b,shape,height_ratio,server_score):
    if a!=b or server_score<.90:return old
    if old in 'Il1|' and a in 'Il' and shape and shape['shape_class']=='Il' and shape['shift_agreement'] and shape['min_shift_confidence']>=.99:
        return a
    if old!=a and old.lower()==a.lower() and a.lower() in 'aemnorsuvwxz':
        if a.isupper() and height_ratio>=.90:return a
        if a.islower() and height_ratio<=.85:return a
    return old


def main():
    font=json.loads((OUT/'font_predictions.json').read_text(encoding='utf-8'))
    server=json.loads((OUT/'server_predictions.json').read_text(encoding='utf-8'))
    assert not server['llm_inference'] and not server['answers_loaded'] and len(server['results'])==1760
    predictions={(r['id'],r['variant']):r for r in server['results']}
    glyphs={r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    results=[]
    for row in font['results']:
        result=dict(row);result['font_only_text']=row['text'];result['server_context_updates']=[]
        if row['table']!='left':results.append(result);continue
        text=compact(row['text']);source=glyphs[row['id']]
        if len(text)!=len(source['glyphs']):results.append(result);continue
        first,second=predictions[row['id'],'normal'],predictions[row['id'],'more_ink']
        a,b=compact(first['text']),compact(second['text'])
        left,right=index_map(text,a),index_map(text,b)
        shapes={r['index']:r for r in row['verifier_glyphs']}
        output=list(text)
        for i,old in enumerate(text):
            if i not in left or i not in right:continue
            box=source['glyphs'][i]['box_pdf'];ratio=(box[3]-box[1])/source['cap']
            new=contextual_character(old,a[left[i]],b[right[i]],shapes.get(i),ratio,min(first['score'],second['score']))
            if new!=old:
                output[i]=new;result['server_context_updates'].append(dict(index=i,before=old,after=new,height_ratio=ratio,shape=shapes.get(i),server_score=min(first['score'],second['score'])))
        iterator=iter(output);result['text']=''.join(c if c.isspace() else next(iterator) for c in row['text'])
        results.append(result)
    cells={r['id']:r for r in json.loads((BASE/'cells.json').read_text())['cells']}
    english={(r['id'],r['variant']):r for r in json.loads((ROOT/'output/cad_table_refinement/english_predictions.json').read_text(encoding='utf-8'))['results']}
    results,shape_changes=repair_rows(results,cells,glyphs,english,None)
    (OUT/'combined_predictions.json').write_text(json.dumps(dict(llm_inference=False,heldout_labels_loaded=False,
        inference_uses_training_transcripts=False,method='Trained glyph verifier, guarded server context, and YOLO/OCR/source-open-shape C guard; right table preserved.',shape_confusion_changes=shape_changes,results=results),ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMBINED',len(results),'context updates',sum(len(r['server_context_updates']) for r in results),flush=True)


if __name__=='__main__':main()
