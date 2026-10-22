"""Bounded synthetic fine-tune: supplied templates only; no evaluation answers."""
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from prepare import extract_glyphs, resize_glyph, paste_glyph, write_image, yolo_labels
from train import CharacterTrainer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fusion_iterations/spacing_training'
NAMES = string.ascii_uppercase + string.ascii_lowercase + string.digits + ":-/.'_"


def connector(image, rng):
    x, y = rng.integers(15, 255, 2)
    w, h = int(rng.integers(50, 110)), int(rng.integers(35, 85))
    color = (0, 0, 0)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
    for j in range(2, w-5, 18):
        for k in range(8, h-5, 18):
            cv2.circle(image, (x+j+5, y+k), 5, color, 1)
    cv2.line(image, (x-15, y+h//2), (x, y+h//2), color, 1)
    cv2.line(image, (x+w, y+h//2), (min(383,x+w+35), y+h//2), color, 1)


def scene(glyphs, seed, blank=False):
    rng = np.random.default_rng(seed)
    image = np.full((384,384,3), 255, np.uint8)
    if blank:
        for _ in range(int(rng.integers(1, 4))): connector(image, rng)
        return image, []
    objects = []
    rows = int(rng.integers(1, 4))
    for row in range(rows):
        cap = int(rng.integers(32, 83))
        count = int(rng.integers(1, 6))
        ids = rng.integers(0, len(glyphs), count).tolist()
        if count > 1 and rng.random() < .55: ids[int(rng.integers(count))] = int(rng.integers(62,68))
        pieces = [resize_glyph(glyphs[i], cap, 1) for i in ids]
        gaps = rng.integers(-3, 3, max(0,count-1))
        width = sum(p[0].shape[1] for p in pieces) + int(gaps.sum())
        while width > 360:
            cap -= 2
            pieces = [resize_glyph(glyphs[i], cap, 1) for i in ids]
            width = sum(p[0].shape[1] for p in pieces) + int(gaps.sum())
        top = min(p[1] for p in pieces)
        bottom = max(p[1]+p[0].shape[0] for p in pieces)
        band = 360//rows
        x = int(rng.integers(8, max(9,376-width)))
        baseline = 12 + row*band - top + max(0,(band-(bottom-top))//2)
        for j,(cid,(alpha,offset)) in enumerate(zip(ids,pieces)):
            objects.append(paste_glyph(image,alpha,x,baseline+offset,np.zeros(3),cid,row,NAMES[cid]))
            x += alpha.shape[1] + (int(gaps[j]) if j < count-1 else 0)
    if rng.random() < .15:
        image = cv2.bitwise_not(image)
    return image, objects


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    glyphs=[]
    for name,chars in [('alphabet',string.ascii_uppercase),('lowercase',string.ascii_lowercase),('digits',string.digits),('punctuation',":-/.'")]:
        found,_ = extract_glyphs(ROOT/f'punctuation_fixed/source_{name}.png',OUT,chars,len(glyphs),reference_char='/' if name=='punctuation' else None)
        glyphs.extend(found)
    # Explicit geometry for the missing class, not a reference string from the drawing.
    alpha = np.zeros((5,55),np.float32); alpha[2,1:-1]=1
    glyphs.append(dict(alpha=alpha,char='_',reference_height=50,top_from_baseline=1))
    splits={}
    for split,count,base in [('train',520,61000),('val',104,72000),('test',130,83000)]:
        for i in range(count):
            im,objects=scene(glyphs,base+i,blank=(i%5==0))
            write_image(OUT/f'images/{split}/{i:04d}.png',im)
            dest=OUT/f'labels/{split}/{i:04d}.txt';dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(yolo_labels(objects,384,384,len(NAMES)),encoding='utf-8')
        splits[split]=count
    (OUT/'manifest.json').write_text(json.dumps(dict(classes=list(NAMES),splits=splits,scope='Same supplied templates, synthetic scenes. No real validation/audit image or answer used in training.'),indent=2),encoding='utf-8')
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT.resolve()),train='images/train',val='images/val',test='images/test',names=dict(enumerate(NAMES))),allow_unicode=True,sort_keys=False),encoding='utf-8')


def main():
    torch.set_num_threads(4)
    prepare()
    if (OUT/'training/weights/best.pt').exists():
        print('EXISTING TRAINING: not overwriting',flush=True);return
    model=YOLO(str(ROOT/'punctuation_fixed/training/weights/best.pt'))
    model.train(trainer=CharacterTrainer,data=str(OUT/'data.yaml'),epochs=12,imgsz=384,batch=16,
        device=0,workers=0,project=str(OUT),name='training',seed=20261017,cls_remap=False,
        optimizer='AdamW',lr0=.00035,lrf=.2,warmup_epochs=1,patience=12,amp=False,cache=False,plots=False,
        mosaic=0,mixup=0,cutmix=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
        fliplr=0,flipud=0,hsv_h=0,hsv_s=0,hsv_v=0)
    print('SPACING TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
