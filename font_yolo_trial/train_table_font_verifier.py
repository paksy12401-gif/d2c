"""Train a five-class verifier on selected CAD-font glyphs, not held-out words."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from refine_table_ocr import BASE,ROOT,compact

OUT=ROOT/'output/cad_table_font_training'
CLASSES=['O','0','Il','1']


def target_class(char):return 'Il' if char in 'Il' else char


def verified_character(char,shape,confidence,agreement,context):
    if not agreement or confidence<.99:return char
    if char in 'O0':return shape if shape in ['O','0'] else char
    if char not in 'Il1|':return char
    if shape=='1':return '1'
    if shape=='Il' and len(context)==2 and len(set(context))==1 and context[0] in ['I','l']:return context[0]
    return char


def patch(image,cell,glyph,cap):
    box=glyph['box_pdf'];origin=cell['bbox_pdf'];scale=48/cap
    center=[((box[i]+box[i+2])/2-origin[i])*4 for i in [0,1]]
    ratio=scale/4
    matrix=np.array([[ratio,0,32-center[0]*ratio],[0,ratio,32-center[1]*ratio]],np.float32)
    result=cv2.warpAffine(image,matrix,(64,64),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    lo=[max(0,int(np.floor(32-(box[i+2]-box[i])*scale/2-1.5))) for i in [0,1]]
    hi=[min(64,int(np.ceil(32+(box[i+2]-box[i])*scale/2+1.5))) for i in [0,1]]
    mask=np.zeros((64,64),bool);mask[lo[1]:hi[1],lo[0]:hi[0]]=True
    result[~mask]=255
    return result


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(24*8*8,64),nn.ReLU(),nn.Linear(64,len(CLASSES)))
    def forward(self,x):return self.layers(x)


def prepare():
    training=json.loads((OUT/'train.json').read_text(encoding='utf-8'))['cells']
    split=json.loads((OUT/'holdout.json').read_text(encoding='utf-8'))
    assert not({r['id'] for r in training}&{r['id'] for r in split['answers']})
    assert not({compact(r['reference']) for r in training}&{compact(r['reference']) for r in split['answers']})
    cells={c['id']:c for c in json.loads((BASE/'cells.json').read_text())['cells']}
    source={r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    images=[];labels=[];records=[];seen={}
    for row in training:
        identity=row['id'];field=source[identity];text=compact(row['reference'])
        assert len(text)==len(field['glyphs'])
        image=cv2.imdecode(np.fromfile(BASE/'clean'/(identity+'.png'),np.uint8),0)
        for index,(char,glyph) in enumerate(zip(text,field['glyphs'])):
            if char not in 'O0Il1':continue
            target_label=target_class(char)
            value=patch(image,cells[identity],glyph,field['cap'])
            digest=hashlib.sha256(value.tobytes()).hexdigest()
            if digest in seen:
                assert seen[digest]==target_label,'Identical pixels have conflicting labels; visual distinction is not learnable.'
                continue
            seen[digest]=target_label;images.append(value);labels.append(CLASSES.index(target_label))
            name=f'{len(images)-1:03d}_{ord(char)}.png'
            target=OUT/'training_glyphs'/name;target.parent.mkdir(exist_ok=True)
            target.write_bytes(cv2.imencode('.png',value)[1].tobytes())
            records.append(dict(cell=identity,index=index,label=char,target_label=target_label,image=name,pixel_sha256=digest))
    assert set(labels)==set(range(len(CLASSES)))
    np.savez_compressed(OUT/'training_glyphs.npz',images=np.array(images),labels=np.array(labels))
    manifest=dict(classes=CLASSES,train_file_sha256=hashlib.sha256((OUT/'train.json').read_bytes()).hexdigest(),
        unique_glyphs=len(images),class_counts=dict(Counter(r['target_label'] for r in records)),records=records,
        ambiguous_shape_group='Uppercase I and lowercase l are identical straight stems in this table font; learned as one group and not distinguished by image-only training.',
        limitation='Glyphs from five source cells of one font. Augmentation is synthetic; same-font held-out cells do not establish unseen-font accuracy.')
    (OUT/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(8*90,len(CLASSES)*100),'white');draw=ImageDraw.Draw(sheet)
    for row,char in enumerate(CLASSES):
        entries=[(im,rec) for im,rec in zip(images,records) if rec['target_label']==char][:8]
        for col,(im,rec) in enumerate(entries):
            x,y=col*90,row*100
            draw.text((x+5,y+2),f'{rec["label"]} #{CLASSES.index(char)}',fill='black')
            sheet.paste(Image.fromarray(im).convert('RGB'),(x+8,y+20))
    sheet.save(OUT/'training_glyphs.png')
    print(json.dumps(manifest['class_counts']),flush=True)


def train():
    torch.manual_seed(814);np.random.seed(814);random.seed(814);torch.set_num_threads(4)
    source=np.load(OUT/'training_glyphs.npz')
    images=torch.tensor((255-source['images'])/255,dtype=torch.float32,device='cuda')[:,None]
    labels=torch.tensor(source['labels'],device='cuda',dtype=torch.long)
    pools=[torch.where(labels==c)[0] for c in range(len(CLASSES))]
    model=Verifier().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]
    # Fixed training budget; no held-out transcripts or scores select epochs.
    for step in range(600):
        target=torch.randint(len(CLASSES),(128,),device='cuda')
        indices=torch.stack([pools[int(c)][torch.randint(len(pools[int(c)]),(1,),device='cuda')][0] for c in target])
        batch=images[indices]
        angles=(torch.rand(128,device='cuda')-.5)*.035
        scales=1+(torch.rand(128,device='cuda')-.5)*.04
        affine=torch.zeros((128,2,3),device='cuda')
        affine[:,0,0]=angles.cos()*scales;affine[:,0,1]=-angles.sin()*scales
        affine[:,1,0]=angles.sin()*scales;affine[:,1,1]=angles.cos()*scales
        affine[:,:,2]=(torch.rand(128,2,device='cuda')-.5)*.08
        grid=F.affine_grid(affine,batch.shape,align_corners=False)
        batch=F.grid_sample(batch,grid,align_corners=False)
        if step%3==0:batch=F.max_pool2d(batch,3,stride=1,padding=1)
        batch=(batch+torch.randn_like(batch)*.015).clamp(0,1)
        logits=model(batch);loss=F.cross_entropy(logits,target)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0 or step==599:
            row=dict(step=step+1,loss=float(loss.detach()),batch_accuracy=float((logits.argmax(1)==target).float().mean()))
            history.append(row);print(json.dumps(row),flush=True)
    model.eval()
    with torch.no_grad(): accuracy=float((model(images).argmax(1)==labels).float().mean())
    torch.save(model.cpu().state_dict(),OUT/'font_verifier.pt')
    manifest=json.loads((OUT/'training_manifest.json').read_text())
    (OUT/'training_report.json').write_text(json.dumps(dict(classes=CLASSES,steps=600,seed=814,
        clean_training_accuracy=accuracy,training_unique_glyphs=manifest['unique_glyphs'],
        heldout_labels_loaded=False,history=history,weights_sha256=hashlib.sha256((OUT/'font_verifier.pt').read_bytes()).hexdigest()),indent=2))
    print('TRAINING COMPLETE',accuracy,flush=True)


def infer():
    torch.set_num_threads(4)
    model=Verifier();model.load_state_dict(torch.load(OUT/'font_verifier.pt',map_location='cpu',weights_only=True));model.eval()
    cells={c['id']:c for c in json.loads((BASE/'cells.json').read_text())['cells']}
    source={r['id']:r for r in json.loads((BASE/'yolo_predictions.json').read_text(encoding='utf-8'))['results']}
    previous=json.loads((ROOT/'output/cad_table_refinement/refined_predictions.json').read_text(encoding='utf-8'))
    english_data=json.loads((ROOT/'output/cad_table_refinement/english_predictions.json').read_text(encoding='utf-8'))
    english={(r['id'],r['variant']):r for r in english_data['results']}
    from refine_table_ocr import index_map
    results=[]
    for record in previous['results']:
        item=dict(record);item['previous_text']=record['text'];item['verifier_updates']=[];item['verifier_glyphs']=[]
        if record['table']!='left':results.append(item);continue
        field=source[record['id']];text=compact(record['text'])
        if len(text)!=len(field['glyphs']):item['verifier_status']='glyph_count_mismatch';results.append(item);continue
        image=cv2.imdecode(np.fromfile(BASE/'clean'/(record['id']+'.png'),np.uint8),0)
        output=list(text)
        context=[]
        for variant in ['normal','more_ink']:
            view=english[record['id'],variant]
            chars=[c for c in view['ctc']['chars'] if not c['text'].isspace()]
            context.append((chars,index_map(text,''.join(c['text'] for c in chars))))
        for index,(char,glyph) in enumerate(zip(text,field['glyphs'])):
            if char not in 'O0Il1|':continue
            value=patch(image,cells[record['id']],glyph,field['cap'])
            ink=torch.tensor((255-value)/255,dtype=torch.float32)[None,None]
            inputs=torch.cat([ink,torch.roll(ink,1,2),torch.roll(ink,-1,2),torch.roll(ink,1,3),torch.roll(ink,-1,3)])
            with torch.no_grad(): probabilities=model(inputs).softmax(-1)
            labels=probabilities.argmax(1);predicted=CLASSES[int(labels[0])]
            confidence=float(probabilities[:,int(labels[0])].min())
            item['verifier_glyphs'].append(dict(index=index,shape_class=predicted,min_shift_confidence=confidence,shift_agreement=bool((labels==labels[0]).all())))
            candidates=[]
            if predicted=='Il':
                for chars,mapping in context:
                    other=mapping.get(index)
                    if other is not None and chars[other]['text'] in 'Il' and chars[other]['score']>=.95:candidates.append(chars[other]['text'])
            predicted=verified_character(char,predicted,confidence,bool((labels==labels[0]).all()),candidates)
            if predicted!=char:
                output[index]=predicted;item['verifier_updates'].append(dict(index=index,before=char,after=predicted,min_shift_confidence=confidence,box_pdf=glyph['box_pdf']))
        iterator=iter(output);item['text']=''.join(c if c.isspace() else next(iterator) for c in record['text'])
        item['verifier_status']='changed' if item['verifier_updates'] else 'preserved'
        results.append(item)
    (OUT/'font_predictions.json').write_text(json.dumps(dict(llm_inference=False,heldout_labels_loaded=False,
        training_labels_embedded=False,classes=CLASSES,results=results),ensure_ascii=False,indent=2),encoding='utf-8')
    print('FONT INFERENCE',Counter(r.get('verifier_status','right_table_preserved') for r in results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','infer']);args=parser.parse_args()
    {'prepare':prepare,'train':train,'infer':infer}[args.stage]()
