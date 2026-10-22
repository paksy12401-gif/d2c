"""Geometry-only punctuation supplements, with explicit Korean word ownership."""
import numpy as np

from infer_real_validation import inside


def extra_symbol(group,cap,baseline):
    a,b,c,d=group['box'];width=c-a;height=d-b
    if height<cap*.025 and cap*.25<width<cap*1.4 and baseline-cap*.7<b<baseline-cap*.25:return '-'
    if not(cap*.12<width<cap*.45 and cap*.75<height<cap*1.3):return None
    points=np.asarray([(s[0],s[1]) for s in group['segments']]+[(s[2],s[3]) for s in group['segments']])
    top=points[points[:,1]<=b+height*.15,0];bottom=points[points[:,1]>=d-height*.15,0]
    middle=points[np.abs(points[:,1]-(b+d)/2)<=height*.30,0]
    if not(len(top) and len(bottom) and len(middle)):return None
    if abs(top.mean()-bottom.mean())>width*.3:return None
    bend=(top.mean()+bottom.mean())/2-middle.mean()
    if bend>width*.35:return '('
    if bend<-width*.35:return ')'
    return None


def supplement(glyphs,path,korean_words):
    result=list(glyphs)
    for group in path['groups']:
        text=extra_symbol(group,path['cap'],path['baseline'])
        if not text:continue
        box=group['box']
        if any(inside(box,w['box_pdf']) for w in korean_words):continue
        if any(g['text']==text and inside(g['box_pdf'],[box[0]-.1*path['cap'],box[1]-.1*path['cap'],box[2]+.1*path['cap'],box[3]+.1*path['cap']]) for g in result):continue
        result=[g for g in result if not inside(g['box_pdf'],[box[0]-.03*path['cap'],box[1]-.03*path['cap'],box[2]+.03*path['cap'],box[3]+.03*path['cap']])]
        result.append(dict(text=text,score=1.,box_pdf=box,engine='vector_extra_geometry_symbol'))
    return result
