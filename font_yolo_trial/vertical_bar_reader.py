"""Opt-in | supplement: unchanged letter/punctuation models, new class only.

Ambiguous plain-stem I/l/| cannot be distinguished by identical pixels alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from prepare import read_image, write_image
from preserved_reader import PreservedReader, symbol_nms
from train import read_rows
from train_vertical_bar import OUT, NAMES


def strong_ink(image):
    background = np.median(image.reshape(-1, 3), axis=0)
    contrast = np.max(np.abs(image.astype(float)-background), axis=2)
    return contrast > max(20, contrast.max()*.8)


def bar_shape(ink, box):
    h, w = ink.shape
    x0, y0, x1, y1 = box
    margin = max(3, round((y1-y0)*.2))
    left, top = max(0, int(np.floor(x0))-2), max(0, int(np.floor(y0))-margin)
    right, bottom = min(w, int(np.ceil(x1))+2), min(h, int(np.ceil(y1))+margin)
    if left >= right or top >= bottom:
        return False
    ys, xs = np.where(ink[top:bottom, left:right])
    if not len(xs):
        return False
    width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    # Reject a fragment of an I/l/1, or a long drawing line continuing past the box.
    return bool(height >= 12 and width/height <= .2 and len(xs)/(width*height) >= .85
                and ys.min()+top >= y0-3 and ys.max()+top <= y1+3)


def word_neighbors(bar,existing,pool):
    height=bar[3]-bar[1];neighbors=[]
    for other in existing:
        if not 0<=int(other[5])<62:continue
        vertical=max(0,min(other[3],bar[3])-max(other[1],bar[1]))
        if vertical<.55*min(height,other[3]-other[1]):continue
        center=(other[0]+other[2])/2
        if bar[0]-2<=center<=bar[2]+2:continue
        # A neighboring stem also supported as | belongs to a symbol run, not word evidence.
        if any(p[0]-2<=center<=p[2]+2 for p in pool):continue
        gap=max(bar[0]-other[2],other[0]-bar[2],0)
        if gap<=height*.4:neighbors.append(other)
    return neighbors


def ink_height(ink,box):
    h,w=ink.shape
    x0,y0=max(0,int(np.floor(box[0]))),max(0,int(np.floor(box[1])))
    x1,y1=min(w,int(np.ceil(box[2]))),min(h,int(np.ceil(box[3])))
    if x1<=x0 or y1<=y0:return 0
    ys,_=np.where(ink[y0:y1,x0:x1])
    return int(ys.max()-ys.min()+1) if len(ys) else 0


def relative_bar_height(ink,bar,existing,pool,min_ratio):
    """Use nearby same-row cap-height letters, never the candidate's own I/l box."""
    bh=ink_height(ink,bar)
    reference_chars=set('ABCDEFGHJKLMNOPQRSTUVWXYZ023456789bdfhk')
    references=[];alternatives=[]
    for other in existing:
        cid=int(other[5])
        if cid<0 or cid>=62 or NAMES[cid] in 'Iil1' or other[4]<.5:continue
        cx=(other[0]+other[2])/2
        if any(p[0]-2<=cx<=p[2]+2 for p in pool):continue
        height=other[3]-other[1]
        vertical=max(0,min(other[3],bar[3])-max(other[1],bar[1]))
        if vertical<height*.6:continue
        if abs((other[1]+other[3]-bar[1]-bar[3])/2)>height*.65:continue
        gap=max(other[0]-bar[2],bar[0]-other[2],0)
        if gap>height*6:continue
        measured=ink_height(ink,other)
        if measured>=8:
            item=dict(box=other,height=measured,gap=gap)
            (references if NAMES[cid] in reference_chars else alternatives).append(item)
    if references:
        anchor=float(np.percentile([r['height'] for r in references],75))
        # Case can be confused (X/x, V/v); accept equal-height support without treating x-height as cap height.
        references.extend(r for r in alternatives if .9<=r['height']/anchor<=1.1)
    references=sorted(references,key=lambda r:r['gap'])[:8]
    if len(references)<2:
        return dict(accepted=False,reason='no_reference',bar_ink_height=bh,references=references)
    cap=float(np.percentile([r['height'] for r in references],75))
    ratio=bh/cap
    return dict(accepted=ratio>=min_ratio,reason='longer' if ratio>=min_ratio else 'not_longer',
                bar_ink_height=bh,reference_height=cap,height_ratio=ratio,min_ratio=min_ratio,references=references)


def merge_bars(image, existing, candidates, protect_words=False, min_height_ratio=None):
    if min_height_ratio is not None and (not np.isfinite(min_height_ratio) or min_height_ratio<=1):
        raise ValueError('min_height_ratio must be finite and greater than 1')
    ink = strong_ink(image)
    boxes = [list(map(float, b)) for b in existing]
    accepted, rejected, context_rejected, height_checks = [], [], [], []
    pool=symbol_nms([b for b in candidates if int(b[5]) == 67 and b[4] >= .6])
    for bar in pool:
        if min_height_ratio is not None:
            length=relative_bar_height(ink,bar,existing,pool,min_height_ratio)
            height_checks.append(dict(bar=bar,**length))
            if not length['accepted']:
                rejected.append(bar)
                continue
        neighbors=word_neighbors(bar,existing,pool) if protect_words else []
        if neighbors:
            rejected.append(bar)
            context_rejected.append(dict(bar=bar,neighbors=neighbors))
            continue
        touching = []
        for i, old in enumerate(boxes):
            intersection = max(0, min(bar[2],old[2])-max(bar[0],old[0])) * max(0,min(bar[3],old[3])-max(bar[1],old[1]))
            area = max(1, (bar[2]-bar[0])*(bar[3]-bar[1]))
            if intersection/area >= .2:
                touching.append(i)
        region = list(bar[:4])
        for i in touching:
            region = [min(region[0],boxes[i][0]),min(region[1],boxes[i][1]),max(region[2],boxes[i][2]),max(region[3],boxes[i][3])]
        permitted = all(int(boxes[i][5]) in (8, 37, 53) for i in touching)
        if not permitted or not bar_shape(ink, region):
            rejected.append(bar)
            continue
        removed = [boxes[i] for i in touching]
        boxes = [b for i,b in enumerate(boxes) if i not in touching]
        boxes.append(bar)
        accepted.append(dict(bar=bar, removed=removed))
    return np.asarray(boxes, dtype=np.float32).reshape(-1,6), dict(accepted=accepted, rejected=rejected,context_rejected=context_rejected,height_checks=height_checks)


class VerticalBarReader(PreservedReader):
    def __init__(self, bar_weights=None, adapt_large=False, protect_words=False, min_height_ratio=None):
        super().__init__()
        self.adapt_large=adapt_large
        self.protect_words=protect_words
        self.min_height_ratio=min_height_ratio
        self.paths['vertical_bar'] = Path(bar_weights) if bar_weights is not None else OUT/'training/weights/best.pt'
        self.vertical_bar = YOLO(str(self.paths['vertical_bar']))
        assert self.vertical_bar.names[67] == '|'
        self.names = dict(self.names)
        self.names[67] = '|'

    def add_bars(self, image, result, candidates):
        merged, details = merge_bars(image, result.boxes.cpu().numpy().data.tolist(), candidates,protect_words=self.protect_words,min_height_ratio=self.min_height_ratio)
        return Results(orig_img=image, path=result.path, names=self.names, boxes=merged), details

    def predict(self, image, path='input.png', mode='tiles', imgsz=384):
        original, baseline, details = super().predict(image, path, mode, imgsz)
        # Native 384px tiles avoid enlarging a small glyph crop beyond training sizes.
        candidates = self._run(self.vertical_bar, image, path, 'tiles', 384)
        proposed=candidates.boxes.cpu().numpy().data.tolist()
        heights=[b[3]-b[1] for b in baseline.boxes.cpu().numpy().data.tolist() if b[3]-b[1]>=12]
        if self.adapt_large and heights and np.median(heights)>72:
            ratio=56/float(np.median(heights))
            scaled=cv2.resize(image,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
            sx,sy=scaled.shape[1]/image.shape[1],scaled.shape[0]/image.shape[0]
            resized=self._run(self.vertical_bar,scaled,path,'tiles',384)
            proposed.extend([[b[0]/sx,b[1]/sy,b[2]/sx,b[3]/sy,*b[4:]] for b in resized.boxes.cpu().numpy().data.tolist()])
        result, bars = self.add_bars(image, original, proposed)
        details['vertical_bar'] = bars
        return result, baseline, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bar-weights',type=Path,default=Path(__file__).resolve().parent/'vertical_bar_branch/training/weights/best.pt')
    parser.add_argument('--min-height-ratio',type=float,default=1.15)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reader = VerticalBarReader(args.bar_weights,adapt_large=True,min_height_ratio=args.min_height_ratio)
    result, _, details = reader.predict(read_image(args.input), args.input)
    args.output.mkdir(parents=True,exist_ok=True)
    write_image(args.output/'detected.png', result.plot(line_width=1,font_size=10))
    report = dict(rows=read_rows(result), vertical_bar=details['vertical_bar'], detections=[dict(label=reader.names[int(b[5])], confidence=b[4], xyxy=b[:4]) for b in result.boxes.cpu().numpy().data.tolist()])
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
