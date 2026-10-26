"""Keep directly observed glyphs separate from the installed-font candidates."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image
import pymupdf

OUT=Path(__file__).parent/'observed_additions'
SRC=next(p for p in Path('C:/Users/Win11Pro/Desktop').glob('2.*/*.pdf') if p.name=='전도면 예시.pdf')
HASH=hashlib.sha256(SRC.read_bytes()).hexdigest()
doc=pymupdf.open(SRC);page=doc[0];draw=page.get_cdrawings(); display=page.get_displaylist()

def edges(path_ids):
    result=[]
    for pid in path_ids:
        for item in draw[pid]['items']:
            assert item[0]=='l', (pid,item[0])
            result.append([*item[1],*item[2]])
    return np.array(result,dtype=float)

def emit(font,ch,e,pids,cap,baseline,mode,word):
    if ch in [g['char'] for g in font['glyphs']]:return
    target=OUT/font['font_id'];target.mkdir(parents=True,exist_ok=True)
    cp=f'U{ord(ch):04X}';p=e.reshape(-1,2);lo=p.min(0);hi=p.max(0)
    vec=target/(cp+'.npy');np.save(vec,e)
    # Canonical raster comes from unaltered original PDF, with common word cap
    # and baseline. No per-glyph independent width/height normalization.
    clip=pymupdf.Rect(*lo,*hi)+(-.02,-.02,.02,.02)
    pix=display.get_pixmap(matrix=pymupdf.Matrix(64,64),clip=clip,colorspace=pymupdf.csGRAY,alpha=False)
    raw=Image.frombytes('L',(pix.width,pix.height),pix.samples)
    scale=320/cap; size=(max(1,round(clip.width*scale)),max(1,round(clip.height*scale)))
    raw.save(target/(cp+'_original_context.png'))
    # Render only the selected source subpaths: italic glyph bounding boxes can
    # overlap, so a rectangular source crop alone is not a clean glyph master.
    canvasdoc=pymupdf.open();canpage=canvasdoc.new_page(width=640,height=640);shape=canpage.new_shape()
    points=e.reshape(-1,2).copy();points[:,0]=320+(points[:,0]-(lo[0]+hi[0])/2)*scale;points[:,1]=480+(points[:,1]-baseline)*scale
    contours=[];current=[]
    for a,b in points.reshape(-1,2,2):
        if not current or not np.allclose(current[-1],a,atol=1e-5):
            if current:contours.append(current)
            current=[a.tolist()]
        current.append(b.tolist())
    if current:contours.append(current)
    for contour in contours:
        if mode=='filled':assert np.allclose(contour[0],contour[-1],atol=1e-5),(word,ch,'open_contour')
        shape.draw_polyline(contour)
    width=0 if mode=='filled' else .213*scale
    shape.finish(color=None if mode=='filled' else (0,0,0),fill=(0,0,0) if mode=='filled' else None,width=width,even_odd=True,closePath=False)
    shape.commit();cpix=canpage.get_pixmap(matrix=pymupdf.Matrix(2,2),colorspace=pymupdf.csGRAY,alpha=False)
    canvas=Image.frombytes('L',(cpix.width,cpix.height),cpix.samples).resize((640,640),Image.Resampling.LANCZOS);canvasdoc.close()
    png=target/(cp+'.png');canvas.save(png)
    yy,xx=np.where(np.array(canvas)<254)
    font['glyphs'].append(dict(id=font['font_id']+'_'+cp,char=ch,image=str(png.relative_to(OUT)).replace('\\','/'),
        vector=str(vec.relative_to(OUT)).replace('\\','/'),bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
        source_path_ids=pids,source_bbox=[*lo.tolist(),*hi.tolist()],source_cap_height=cap,source_baseline_y=baseline,
        baseline_px=480,cap_height_px=320,source_coordinate_system='pdf_y_down',render_mode=mode,
        source_word=word,source_pdf_sha256=HASH,source_font_identity_verified=False,sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
        source_segment_selection='Explicit glyph path IDs for KET; for filled words exact x-disjoint source path subsegments, not OCR geometry',
        original_path_shared_with_other_letters=len(pids)==1,source_stroke_width=0 if mode=='filled' else .213,
        stroke_ratio=0 if mode=='filled' else .213/cap))

fonts=[]
for fid,pid,text,refchar in [('pdf_upright_serif',69473,'LEVER','E'),
                             ('pdf_upright_gulim_like',128395,'DETAILC','E'),
                             ('pdf_note_latin',173147,'FRNT_MAIN','T')]:
    e=edges([pid]);r=pymupdf.Rect(draw[pid]['rect'])
    if pid==173147:e=e[np.max(e[:,[0,2]],axis=1)<458.8];r=pymupdf.Rect(*e.reshape(-1,2).min(0),*e.reshape(-1,2).max(0))
    pix=display.get_pixmap(matrix=pymupdf.Matrix(64,64),clip=r+(-.02,-.02,.02,.02),colorspace=pymupdf.csGRAY,alpha=False)
    mask=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width)<128
    active=mask.any(0); padded=np.r_[False,active,False].astype(int); starts=np.where(np.diff(padded)==1)[0];ends=np.where(np.diff(padded)==-1)[0]
    if text=='LEVER':
        # Source V has a tiny gap between its two thin legs. Its two observed
        # runs are grouped by the reviewed word, not assigned to two letters.
        bands=[]
        for a,b in zip(starts,ends):
            if bands and a-bands[-1][1]<.13*64:bands[-1][1]=b
            else:bands.append([a,b])
        starts,ends=np.array(bands).T
    assert len(starts)==len(text),(text,len(starts))
    selections=[]
    for ch,a,b in zip(text,starts,ends):
        left=(pix.x+a)/64-.02;right=(pix.x+b)/64+.02
        selected=e[(e[:,[0,2]].mean(1)>=left)&(e[:,[0,2]].mean(1)<=right)]
        assert len(selected)>0
        assert np.min(selected[:,[0,2]])>=left-.03 and np.max(selected[:,[0,2]])<=right+.03
        selections.append((ch,selected))
    reference=next(e for ch,e in selections if ch==refchar);yp=reference[:,[1,3]]; baseline=float(yp.max());cap=float(np.ptp(yp))
    font=next((f for f in fonts if f['font_id']==fid),None)
    if font is None:font=dict(font_id=fid,provenance_status='observed_original_pdf_partial',glyphs=[]);fonts.append(font)
    for ch,e in selections:
        if ch!='_':emit(font,ch,e,[pid],cap,baseline,'filled',text)

font=dict(font_id='pdf_ket_outline_logo',provenance_status='observed_original_pdf_partial_logo_not_font_family',glyphs=[]);fonts.append(font)
groups={'K':list(range(43291,43303)),'E':list(range(43275,43287)),'T':list(range(43267,43275))}
allpoints=np.concatenate([edges(ids) for ids in groups.values()]).reshape(-1,2);base=float(allpoints[:,1].max());cap=float(np.ptp(allpoints[:,1]))
for ch,pids in groups.items():emit(font,ch,edges(pids),pids,cap,base,'outline','KET')
OUT.mkdir(exist_ok=True)
(OUT/'masters_manifest.json').write_text(json.dumps(dict(source=str(SRC),source_sha256=HASH,fonts=fonts,
    note='Partial directly observed source glyphs, not complete fonts; original source vectors and common cap/baseline retained. KET logo identity not generalized to other glyphs.'),ensure_ascii=False,indent=2),encoding='utf-8')
print([(f['font_id'],len(f['glyphs'])) for f in fonts])
