"""Local English OCR and evidence-based table fusion. No answer sheets are loaded."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
import time

import cv2
import numpy as np

from table_cell_experiment import ROOT, OUT as BASE

OUT = ROOT / 'output/cad_table_refinement'
sys.path.insert(0, str(ROOT))


def compact(text): return ''.join(text.split())


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1): current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ca!=cb)))
        previous = current
    return previous[-1]


def alphabet(cell):
    table, row, col = cell['table'], cell['row'], cell['column']
    if table == 'left' and row < 31 and col == 0: return string.digits
    if table == 'right' and row > 0:
        if col in [0,1,6]: return string.digits
        if col == 2: return string.digits + '.-'
    # PIN ID2 deliberately retains S/J-1 and S/J-2.
    return None


def image_for(cell, variant):
    image = cv2.imdecode(np.fromfile(BASE/'clean'/(cell['id']+'.png'), np.uint8), 0)
    yy, xx = np.where(image < 200)
    if len(xx): image = image[yy.min():yy.max()+1, xx.min():xx.max()+1]
    if variant == 'more_ink' and cell['table'] == 'left': image = cv2.erode(image, np.ones((2,2), np.uint8))
    target = 48 if variant == 'more_ink' else 32
    scale = target / image.shape[0]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image,8,8,10,10,cv2.BORDER_CONSTANT,value=255)


def ctc_details(probabilities, characters):
    ids=probabilities.argmax(-1)
    starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]
    ends=np.r_[starts[1:],len(ids)]
    result=[]
    for a,b in zip(starts,ends):
        index=int(ids[a])
        if index==0: continue
        peak=int(a+np.argmax(probabilities[a:b,index]))
        alternatives=np.argsort(probabilities[peak])[-4:][::-1]
        result.append(dict(text=characters[index],score=float(probabilities[peak,index]),
            column=int(a),end_column=int(b),alternatives=[dict(text=characters[int(i)],score=float(probabilities[peak,int(i)])) for i in alternatives]))
    return dict(text=''.join(r['text'] for r in result),columns=len(ids),chars=result)


def english():
    from models_local import _offline
    import torch
    torch.set_num_threads(4)
    cells = json.loads((BASE/'cells.json').read_text())['cells']
    jobs = {}
    for cell in cells:
        for variant in ['normal','more_ink']:
            image = image_for(cell, variant)
            allowed = alphabet(cell)
            digest = hashlib.sha256(str((image.shape,allowed)).encode()+image.tobytes()).hexdigest()
            jobs.setdefault(digest,dict(image=image,allowed=allowed,addresses=[]))['addresses'].append((cell,variant))
    results = []
    started = time.monotonic()
    with _offline():
        import easyocr
        from easyocr.recognition import get_text
        from easyocr.config import imgH
        reader = easyocr.Reader(['en'],gpu=True,detector=False,download_enabled=False,
            model_storage_directory=str(ROOT/'models/easyocr'),user_network_directory=str(ROOT/'models/easyocr/user_network'),verbose=False)
        captured=[]
        def capture(module,inputs,output): captured.append(output.detach().float().softmax(-1).cpu().numpy())
        hook=reader.recognizer.register_forward_hook(capture)
        print('ENGLISH',len(jobs),'unique inputs',flush=True)
        grouped = {}
        for digest, job in jobs.items(): grouped.setdefault(job['allowed'],[]).append(digest)
        for allowed, digests in grouped.items():
            digests.sort(key=lambda d:jobs[d]['image'].shape[1]/jobs[d]['image'].shape[0])
            ignore = ''.join(set(reader.character)-set(allowed)) if allowed else ''
            for start in range(0,len(digests),16):
                batch = digests[start:start+16]
                images = []
                for digest in batch:
                    image = jobs[digest]['image']
                    images.append(([[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]],image))
                width = max(math.ceil(im.shape[1]/im.shape[0])*imgH for _,im in images)
                captured.clear()
                values = get_text(reader.character,imgH,width,reader.recognizer,reader.converter,images,
                    ignore_char=ignore,decoder='greedy',batch_size=16,workers=0,device=reader.device,contrast_ths=0)
                assert len(values)==len(batch)
                probabilities=np.concatenate(captured)
                if allowed:
                    invalid=[i for i,c in enumerate(reader.converter.character) if i and c not in allowed]
                    probabilities[:,:,invalid]=0
                    probabilities/=probabilities.sum(-1,keepdims=True)
                for digest, (_,text,score),probability in zip(batch,values,probabilities):
                    details=ctc_details(probability,reader.converter.character)
                    assert details['text']==text
                    for cell, variant in jobs[digest]['addresses']:
                        results.append(dict(**cell,variant=variant,text=text,score=float(score),input_digest=digest,allowed=allowed,ctc=details))
                (OUT/'english_predictions.json').write_text(json.dumps(dict(llm_inference=False,answers_loaded=False,
                    model='EasyOCR english_g2',elapsed_seconds=time.monotonic()-started,results=results),ensure_ascii=False,indent=2),encoding='utf-8')
                print('ENGLISH',len(results),'/ 1760',round(time.monotonic()-started),'seconds',flush=True)
    assert len(results)==1760


def paddle_retry():
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    cells=json.loads((BASE/'cells.json').read_text())['cells']
    selected=[c for c in cells if c['table']=='left' or (c['table']=='right' and c['column']==3 and c['row']>0)]
    jobs={}
    for cell in selected:
        for variant in ['normal','more_ink']:
            image=cv2.cvtColor(image_for(cell,variant),cv2.COLOR_GRAY2BGR)
            digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            jobs.setdefault(digest,dict(image=image,addresses=[]))['addresses'].append((cell,variant))
    results=[];partial_jobs={};started=time.monotonic()
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        original=model.paddlex_predictor.post_op;captured=[]
        def postprocess(pred,**kwargs):
            probabilities=np.asarray(pred[0])
            for i,p in enumerate(probabilities):
                detail=ctc_details(p,original.character)
                if kwargs.get('return_word_box'):
                    detail['columns']*=kwargs['wh_ratio_list'][i]/kwargs['max_wh_ratio']
                captured.append(detail)
            return original(pred,**kwargs)
        model.paddlex_predictor.post_op=postprocess
        ordered=sorted(jobs,key=lambda d:jobs[d]['image'].shape[1])
        for start in range(0,len(ordered),16):
            batch=ordered[start:start+16];captured.clear()
            predictions=list(model.predict(input=[jobs[d]['image'] for d in batch],batch_size=16,return_word_box=True))
            assert len(predictions)==len(batch)==len(captured)
            for digest,prediction,detail in zip(batch,predictions,captured):
                raw=prediction.json.get('res',prediction.json);text=raw['rec_text'][0]
                assert detail['text']==text
                image=jobs[digest]['image']
                for cell,variant in jobs[digest]['addresses']:
                    results.append(dict(id=cell['id'],variant=variant,text=text,score=float(raw['rec_score']),ctc=detail))
                    if cell['table']!='left':continue
                    for match in re.finditer(r'[A-Za-z0-9|]+',text):
                        token=match.group()
                        if not(3<=len(token)<=12 and re.search(r'[O0Il1|]',token)):continue
                        first,last=detail['chars'][match.start()],detail['chars'][match.end()-1]
                        x0=max(0,round((first['column']-.5)/detail['columns']*image.shape[1]-image.shape[0]*.2))
                        x1=min(image.shape[1],round((last['end_column']+.5)/detail['columns']*image.shape[1]+image.shape[0]*.3))
                        if x1<=x0:continue
                        crop=image[:,x0:x1].copy()
                        key=hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
                        partial_jobs.setdefault(key,dict(image=crop,addresses=[]))['addresses'].append(dict(id=cell['id'],variant=variant,start=match.start(),end=match.end(),token=token))
            print('PADDLE',len(results),'full cells',round(time.monotonic()-started),'seconds',flush=True)
        partials=[];ordered=sorted(partial_jobs,key=lambda d:partial_jobs[d]['image'].shape[1])
        for start in range(0,len(ordered),16):
            batch=ordered[start:start+16];captured.clear()
            predictions=list(model.predict(input=[partial_jobs[d]['image'] for d in batch],batch_size=16,return_word_box=True))
            assert len(predictions)==len(batch)==len(captured)
            for digest,prediction,detail in zip(batch,predictions,captured):
                raw=prediction.json.get('res',prediction.json)
                for address in partial_jobs[digest]['addresses']:
                    partials.append(dict(**address,text=raw['rec_text'][0],score=float(raw['rec_score']),ctc=detail))
            print('PADDLE',len(partials),'partial words',flush=True)
        (OUT/'paddle_retry.json').write_text(json.dumps(dict(llm_inference=False,answers_loaded=False,results=results,partials=partials),ensure_ascii=False,indent=2),encoding='utf-8')


def fold_code(value): return compact(value).translate(str.maketrans({'O':'0','I':'1','l':'1','|':'1'}))


def connector_match(text, candidates):
    value = compact(text)
    if not re.fullmatch(r'\d{8}_[A-Za-z0-9_]+',value): return None
    matches = {c for c in candidates if c[:8]==value[:8] and fold_code(c)==fold_code(value)}
    return next(iter(matches)) if len(matches)==1 else None


def color_tokens(baseline):
    by_id = {(r['id'],r['variant']):r for r in baseline}
    tokens = Counter()
    evidence = []
    for row in baseline:
        if row['variant']!='clean_cell' or row['table']!='right' or row['column']!=3 or row['row']==0: continue
        other = by_id[row['id'],'raw_cell']
        value = compact(row['text'])
        if value==compact(other['text']) and min(row['score'],other['score'])>=.95 and re.fullmatch(r'[A-Z]{2}(?:/[A-Z]{2})?',value):
            tokens.update(value.split('/')); evidence.append(dict(id=row['id'],text=value))
    return dict(tokens), evidence


def equivalent(a,b):
    return a==b or any(a in group and b in group for group in ['O0','I1l|']) or a.lower()==b.lower()


def color_reread(text,views,tokens):
    valid=lambda s:bool(re.fullmatch(r'[A-Z]{2}(?:/[A-Z]{2})?',s)) and all(t in tokens for t in s.split('/'))
    if valid(compact(text)):return None
    candidates={compact(v['text']) for v in views if v['score']>=.90 and valid(compact(v['text'])) and distance(compact(text),compact(v['text']))<=1}
    if len(candidates)!=1:return None
    candidate=next(iter(candidates))
    for view in views:
        chars=[c for c in view['ctc']['chars'] if not c['text'].isspace()]
        if len(chars)!=len(candidate):return None
        for target,char in zip(candidate,chars):
            if target==char['text']:continue
            alternatives={a['text']:a['score'] for a in char['alternatives']}
            probability=alternatives.get(target,0)
            if probability<.10 or probability/max(char['score'],1e-8)<.25:return None
    return candidate


def index_map(a,b):
    dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):dp[i][0]=i
    for j in range(len(b)+1):dp[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(not equivalent(a[i-1],b[j-1])))
    i,j=len(a),len(b);mapping={}
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(not equivalent(a[i-1],b[j-1])):
            mapping[i-1]=j-1;i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:i-=1
        else:j-=1
    return mapping


def character_vote(text, views, partials):
    base=compact(text)
    protected_digits={i for m in re.finditer(r'\d{2,}',base) for i in range(m.start(),m.end())}
    votes=[{'baseline':dict(text=c,score=1.0)} for c in base]
    mappings={}
    for name,view in views.items():
        chars=[c for c in view['ctc']['chars'] if not c['text'].isspace()]
        value=''.join(c['text'] for c in chars)
        mapping=index_map(base,value);mappings[name]=mapping
        for i,j in mapping.items():
            if chars[j]['score']>=.80 and equivalent(base[i],chars[j]['text']):votes[i][name]=chars[j]
    for part in partials:
        name='paddle_'+part['variant']
        if name not in mappings:continue
        full=views[name]['ctc']['chars']
        token=part['token'];value=re.sub(r'^[^A-Za-z0-9|]+|[^A-Za-z0-9|]+$','',part['text'])
        if len(token)!=len(value) or not all(equivalent(a,b) for a,b in zip(token,value)):continue
        offset=sum(not c['text'].isspace() for c in full[:part['start']])
        chars=[c for c in part['ctc']['chars'] if c['text'].isalnum() or c['text']=='|']
        if len(chars)!=len(value):continue
        for i,j in mappings[name].items():
            if offset<=j<offset+len(value):
                c=chars[j-offset]
                if c['score']>=.80 and equivalent(base[i],c['text']):votes[i][f'partial_{part["variant"]}_{offset}']=c
    output=list(base);updates=[]
    for i,by_view in enumerate(votes):
        counts=Counter(v['text'] for v in by_view.values())
        best,count=counts.most_common(1)[0]
        if i in protected_digits and not best.isdigit():continue
        if best!=base[i] and count>=3 and count>counts[base[i]]:
            output[i]=best;updates.append(dict(index=i,before=base[i],after=best,votes={k:dict(text=v['text'],score=v['score']) for k,v in by_view.items()}))
    # Preserve the baseline's word separators.
    iterator=iter(output)
    return ''.join(c if c.isspace() else next(iterator) for c in text),updates


def fuse():
    baseline_data = json.loads((BASE/'predictions.json').read_text(encoding='utf-8'))
    baseline = {r['id']:r for r in baseline_data['results'] if r['variant']=='clean_cell'}
    english_data = json.loads((OUT/'english_predictions.json').read_text(encoding='utf-8'))
    assert not english_data['answers_loaded'] and len(english_data['results'])==1760
    english = {(r['id'],r['variant']):r for r in english_data['results']}
    source_file = ROOT/'output/cad_fusion_parts_applied/combined_results.json'
    body = json.loads(source_file.read_text(encoding='utf-8'))
    assert not body['inference_llm'] and not body['answers_loaded']
    connectors = {token for record in body['records'] for line in record['fused_lines']
                  for token in re.findall(r'\b\d{8}_[A-Za-z0-9_]+\b',line)}
    tokens, token_evidence = color_tokens(baseline_data['results'])
    retry=json.loads((OUT/'paddle_retry.json').read_text(encoding='utf-8')) if (OUT/'paddle_retry.json').exists() else dict(results=[],partials=[])
    paddle={(r['id'],r['variant']):r for r in retry['results']}
    partials={}
    for p in retry['partials']:partials.setdefault(p['id'],[]).append(p)
    records = []
    for identity, base in baseline.items():
        first, second = english[identity,'normal'],english[identity,'more_ink']
        value = base['text']; reason = 'baseline'; flags = [];updates=[]
        a,b = compact(first['text']),compact(second['text'])
        stable = a==b and min(first['score'],second['score'])>=.80
        if base['table']=='left' and (identity,'normal') in paddle:
            views={'english_normal':first,'english_more_ink':second,
                   'paddle_normal':paddle[identity,'normal'],'paddle_more_ink':paddle[identity,'more_ink']}
            value,updates=character_vote(value,views,partials.get(identity,[]))
            if updates:reason='character_and_partial_votes'
        if base['table']=='right' and base['row']>0:
            if base['column']==3:
                valid = lambda s: bool(re.fullmatch(r'[A-Z]{2}(?:/[A-Z]{2})?',s)) and all(t in tokens for t in s.split('/'))
                if stable and valid(a) and distance(compact(value),a)<=1: value=first['text'];reason='color_english_agreement'
                if (identity,'normal') in paddle:
                    x,y=paddle[identity,'normal'],paddle[identity,'more_ink']
                    v=compact(x['text'])
                    if v==compact(y['text']) and min(x['score'],y['score'])>=.90 and valid(v) and distance(compact(value),v)<=1:
                        value=x['text'];reason='color_paddle_two_inputs'
                    elif base['score']<.95:
                        candidate=color_reread(value,[x,y],tokens)
                        if candidate:value=candidate;reason='color_ctc_alternative_crosscheck'
                if compact(value)!='-' and not valid(compact(value)): flags.append('color_outside_observed_vocabulary')
            if base['column'] in [5,7]:
                candidate=connector_match(value,connectors)
                if candidate and candidate!=compact(value): value=candidate;reason='body_yolo_connector_crosscheck'
        allowed = alphabet(base)
        if allowed and any(c not in allowed for c in compact(value)):
            if stable and a and all(c in allowed for c in a): value=first['text'];reason='numeric_column_reread'
            else: flags.append('numeric_format_unresolved')
        if a!=b and compact(value) not in [a,b]: flags.append('model_disagreement')
        records.append(dict(id=identity,table=base['table'],row=base['row'],column=base['column'],
            bbox_pdf=base['bbox_pdf'],baseline=base['text'],text=value,reason=reason,review_flags=flags,character_updates=updates,
            english_normal=dict(text=first['text'],score=first['score']),english_more_ink=dict(text=second['text'],score=second['score'])))
    result=dict(llm_inference=False,answers_loaded=False,source_sha256=baseline_data['source_sha256'],
        body_predictions_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),body_connector_candidates=sorted(connectors),
        color_token_counts=tokens,color_token_evidence=token_evidence,
        note='Color vocabulary is observed OCR agreement, not an authoritative code list. Unknown tokens are flagged. Same-PDF crosschecks are not independent ground truth.',
        results=records)
    assert len(records)==880
    (OUT/'refined_predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FUSED',Counter(r['reason'] for r in records),'CHANGED',sum(compact(r['baseline'])!=compact(r['text']) for r in records),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['english','paddle','fuse']);args=parser.parse_args()
    {'english':english,'paddle':paddle_retry,'fuse':fuse}[args.stage]()
