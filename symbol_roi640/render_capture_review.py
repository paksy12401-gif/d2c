"""Render proposal sheets and native-size annotated PNGs for local inspection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent/'capture_box_review_20261008'
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
BIG=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)

def display_image(c):
    im=Image.open(ROOT/'images'/c['filename']).convert('RGB')
    if c['background']=='dark':
        a=np.asarray(im).astype(np.int16)
        bg=np.median(a.reshape(-1,3),axis=0)
        # Inspection only: training/original images remain byte-identical.
        intensity=np.clip(np.max(abs(a-bg),axis=2)*3,0,255).astype(np.uint8)
        im=Image.fromarray(255-intensity).convert('RGB')
    return im

def render():
    data=json.loads((ROOT/'proposals.json').read_text(encoding='utf-8'))
    (ROOT/'annotated').mkdir(exist_ok=True)
    (ROOT/'sheets').mkdir(exist_ok=True)
    for c in data['captures']:
        im=display_image(c);d=ImageDraw.Draw(im)
        for obj in c['objects']:
            box=obj['bbox_xyxy_px'];label=f"{obj['box_id']} : {obj['class_name']}?"
            color='#f97316' if obj['kind']=='reference' else '#d000a0'
            d.rectangle(box,outline=color,width=2)
            x,y=box[:2];y=max(0,y-19)
            d.rectangle((x,y,min(im.width,x+len(label)*9),y+19),fill='white')
            d.text((x,y),label,font=FONT,fill=color)
        im.save(ROOT/'annotated'/c['filename'])
    for start in range(0,len(data['captures']),12):
        canvas=Image.new('RGB',(4*400,3*365),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for k,c in enumerate(data['captures'][start:start+12]):
            im=Image.open(ROOT/'annotated'/c['filename']);im.thumbnail((390,320))
            x=k%4*400;y=k//4*365
            canvas.paste(im,(x+(400-im.width)//2,y+35))
            draw.text((x+8,y+5),f"{c['capture_id']}  {c['original_name'].split()[-1]}  ({len(c['objects'])} boxes)",font=BIG,fill='black')
        canvas.save(ROOT/'sheets'/f'review_{start//12+1:02d}.jpg',quality=93)

if __name__=='__main__':render()
