"""Freeze real PDF regions and visually reviewed references before inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from preserved_reader import foreground

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'real_font_validation'


def cap_band(image, center_y, hint):
    counts=(foreground(image)>0).sum(axis=1)
    active=counts>=max(2, int(image.shape[1]*.008))
    bands=np.flatnonzero(np.diff(np.r_[False,active,False].astype(np.int8))).reshape(-1,2)
    bands=[(int(a),int(b)) for a,b in bands if b-a>=max(4,hint*.2)]
    return min(bands,key=lambda ab:abs((ab[0]+ab[1])/2-center_y)) if bands else None


def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'cad_without_tables/source.pdf'
    doc=fitz.open(source);page=doc[0]
    old=json.loads((ROOT/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    refs=[]
    def add(name,kind,box,text,proposal=None,ambiguities=None,origin='assistant visual review authorized for validation only'):
        refs.append(dict(id=name,kind=kind,box_pdf=list(map(float,box)),reference=text,old_proposal=proposal,
                         ambiguities=ambiguities or [],reference_origin=origin))
    specs=[(29,'FRONT_MAO_X1',['MAO']),(77,'FRONT_MAO_X2',['MAO']),
           (101,'FRONT_MAO_X1',['MAO']),(134,'FRONT_X1',[]),(150,'FRONT_X1',[]),
           (122,'X1_X1',[]),(173,'E603_LT_X1_X1',['E603']),(105,'FLEECE TAPE',[]),
           (87,'SPLICE POINT',[]),(95,'M651LEFT',[]),
           (97,'REAR COVER NO: OPU88567',['OPU'])]
    for i,(index,text,ambiguous_words) in enumerate(specs,1):
        ambiguities=[]
        for word in ambiguous_words:
            offset=''.join(text.split()).index(word)
            for j,char in enumerate(word):
                if char in 'O0':ambiguities.append({'index':offset+j,'allowed':['O','0']})
        box=list(old['proposals'][index]['box'])
        if index==97:box[0]=3104
        add(f'L{i:02d}','outlined_latin',box,text,index,ambiguities)
    add('L12','outlined_latin',[2055,978,2131,992],'GROMMET')
    add('L13','outlined_latin',[4217,1333,4265,1347],'bll15914')
    add('L14','outlined_latin',[1288,2134,1351,2147],'13592509',ambiguities=[{'index':6,'allowed':['0','O']}])
    add('L15','outlined_latin',[430,2146,494,2159],'13352244')
    add('L16','outlined_latin',old['proposals'][133]['box'],'A470LEFT',133,[{'index':3,'allowed':['0','O']}])
    add('K17','korean',old['proposals'][200]['box'],'작업 주기',200)
    add('K18','korean',old['proposals'][204]['box'],'3. 부자재 LIST 참조하여 적용할것.',204)
    fitz.TOOLS.set_small_glyph_heights(True)
    words=page.get_text('words')
    for i,value in enumerate(['95','140','170','200'],19):
        word=next(w for w in words if w[4]==value)
        add(f'N{i}','native_font_control',word[:4],value,origin='embedded PDF text; LucidaHandwriting-Italic; already editable, not the supplied CAD font')
    add('D23','drawing_negative',[1945,495,1980,585],'')
    add('D24','drawing_negative',[1690,729,1730,821],'')
    assert len(refs)==24 and len({r['id'] for r in refs})==24
    regions=[];answers=[];panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    for item in refs:
        x0,y0,x1,y1=item['box_pdf'];hint=y1-y0
        search=fitz.Rect(x0,y0-hint*.5,x1,y1+hint*.5)
        pix=page.get_pixmap(matrix=fitz.Matrix(4,4),clip=search,alpha=False)
        bgr=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)[:,:,::-1].copy()
        band=cap_band(bgr,((y0+y1)/2-pix.y/4)*4,hint*4) if item['kind']!='drawing_negative' else None
        if band:
            y0,y1=pix.y/4+band[0]/4,pix.y/4+band[1]/4
        cap=y1-y0 if band else (13 if item['kind']=='drawing_negative' else hint)
        target=[x0,y0,x1,y1] if band else item['box_pdf']
        region={'id':item['id'],'kind':item['kind'],'box_pdf':target,'reference_cap_height_pdf':cap,
                'source_box_before_height_measurement':item['box_pdf']}
        regions.append(region)
        answers.append({k:v for k,v in item.items() if k not in ('box_pdf','kind')})
        clip=fitz.Rect(target)+(-2,-2,2,2)
        pix=page.get_pixmap(matrix=fitz.Matrix(8,8),clip=clip,alpha=False)
        pix.save(OUT/f"{item['id']}_source.png")
        image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);image.thumbnail((1100,110))
        panel=Image.new('RGB',(1200,165),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,4),f"{item['id']}  {item['reference']}",font=font,fill='black');panel.paste(image,(45,40));panels.append(panel)
    (OUT/'regions.json').write_text(json.dumps({'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'llm_inference':False,'regions':regions},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'answers.json').write_text(json.dumps({'usage':'Evaluation only. Never loaded by inference or training. Assistant visual references are not externally certified.',
        'whitespace_policy':'Ignore whitespace for character scoring; raw outputs retained.',
        'answers':answers},ensure_ascii=False,indent=2),encoding='utf-8')
    for start in range(0,len(panels),8):
        sheet=Image.new('RGB',(1200,165*len(panels[start:start+8])),'#ddd')
        for j,panel in enumerate(panels[start:start+8]):sheet.paste(panel,(0,j*165))
        sheet.save(OUT/f'verified_references_{start//8+1}.png')
    print(json.dumps({'regions':len(regions),'cap_heights':[(r['id'],r['reference_cap_height_pdf']) for r in regions]}),flush=True)


if __name__=='__main__':main()
