from pathlib import Path
import json,hashlib
import pymupdf
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
coverage=json.loads((OUT/'coverage.json').read_text(encoding='utf-8'))
rows=[
 ('U01','LEVER',[818,11,846,23],'upright_serif','Upright serif strokes, strong thick/thin contrast; different from the slanted wire-label sample.'),
 ('U02','COVER',[619,596,649,609],'upright_serif','Second independent location of the upright serif label style.'),
 ('U03','VIEW "H"',[913,28,977,45],'upright_sans','Smooth upright sans-serif view title, visibly heavier than the thin CAD stroke labels.'),
 ('U04','DETAIL C',[486,574,528,587],'upright_sans','Upright sans-serif detail caption; may be the same family as VIEW at another size.'),
 ('U05','FRNT_MAIN / CS PAD',[414,205,487,237],'upright_sans_mixed_korean','Latin text in mixed Korean instructions; compare with view titles rather than classify only the Korean text.'),
 ('U06','KET 1',[107,196,139,210],'oblique_outline_sans','Oblique outlined sans-serif lettering; no serif terminals like the wire-label ItalicT candidate.'),
 ('U07','111 / 112',[61,603,100,614],'micro_pin_digits_candidate','Very small pin indices; closer inspection needed to distinguish actual font from lineweight/raster-size effects.'),
 ('U08','63 / 64 / 65',[164,195,220,203],'micro_pin_digits_candidate','Small pin indices alongside narrow large part names; do not infer a new font merely from size.'),
 ('U09','8062 / SECTION D-D',[741,412,781,431],'flattened_cad_stroke_candidate','Wide flat stroke lettering may be an existing CAD shape with anisotropic width/height scaling.'),
 ('U10','RP211',[56,105,79,117],'narrow_upright_cad','Narrow upright component-name style; likely width variant or related stroke font, not proven a new font.'),
 ('U11','245',[572,263,594,279],'thin_tall_dimension_digits','Tall thin numeric dimension style; compare after matching cap height and width.'),
 ('U12','(W)',[428,181,444,191],'small_parenthesized_sans','Compact upright parenthesized annotation, unlike narrow component labels; font identity unverified.'),
 ('U13','FKC-XHL206',[2011,216,2055,227],'wide_upright_cad','Wider upright stroke code in the same large connector; possible width variation, no exact font claim.'),
 ('U14','VIEW "B"',[972,431,1007,445],'wide_thin_cad_view','A thin geometric VIEW caption differs visually from the heavy VIEW H caption; compare its stroke topology before merging styles.'),
 ('U15','1.0~1.2kgf/m',[5,672,57,687],'upright_sans_mixed_case','Torque note contains lower-case Latin kgf/m and digits, useful missing true source exemplars.'),
]
with pymupdf.open(coverage['source']) as doc:
 dl=doc[0].get_displaylist();records=[]
 for sid,text,box,group,note in rows:
  path=OUT/(sid+'.png');pix=dl.get_pixmap(matrix=pymupdf.Matrix(18,18),clip=pymupdf.Rect(box),colorspace=pymupdf.csRGB,alpha=False);pix.save(path)
  records.append(dict(id=sid,visible_text_candidate=text,bbox_pdf=box,image=path.name,shape_group_candidate=group,interpretation=note,exact_font_identity_verified=False,source_modified=False,source_sha256=coverage['source_sha256'],image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
sheet=Image.new('RGB',(1800,180*((len(records)+1)//2)),'#edf0f4');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(records):
 x=i%2*900;y=i//2*180;im=Image.open(OUT/r['image']);im.thumbnail((870,140))
 draw.text((x+8,y+5),r['id']+'  '+r['shape_group_candidate'],font=font,fill='black');sheet.paste(im,(x+8,y+32));draw.rectangle((x,y,x+898,y+178),outline='#bcc2cc')
sheet.save(OUT/'representatives.png')
with (OUT/'representatives.json').open('w',encoding='utf-8') as f:json.dump(dict(source=coverage['source'],source_sha256=coverage['source_sha256'],representatives=records),f,ensure_ascii=False,indent=2)
print(len(records))
