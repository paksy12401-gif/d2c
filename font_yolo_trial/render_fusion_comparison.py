"""Visual review sheet: original pixels and unchanged before/after predictions."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent.parent/'output/cad_fusion_iterations'


def main():
    data=json.loads((ROOT/'evaluation.json').read_text(encoding='utf-8'))
    selected=[('development','L01'),('development','L13'),('development','L09'),('development','K18'),('audit','H12'),('audit','H11')]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
    image=Image.new('RGB',(1500,230*len(selected)+70),'white');draw=ImageDraw.Draw(image)
    draw.text((15,10),'실제 도면 비교: 원본 / 이전 출력 / 자동 검색 후 출력 (오류가 남은 예 포함)',font=font,fill='black')
    for i,(split,identity) in enumerate(selected):
        y=65+i*230
        old=next(r for r in data['rows'] if r['split']==split and r['id']==identity and r['config']=='previous_pipeline')
        new=next(r for r in data['rows'] if r['split']==split and r['id']==identity and r['config']=='automatic_selected')
        draw.text((15,y),identity,font=font,fill='black')
        source=Image.open(ROOT/f'assets/{split}_{identity}_source.png').convert('RGB');source.thumbnail((1390,85))
        image.paste(source,(90,y))
        draw.text((20,y+95),'이전: '+old['prediction'].replace('\n',' / '),font=small,fill='#9a3030')
        draw.text((20,y+130),'현재: '+new['prediction'].replace('\n',' / '),font=small,fill='#185b35')
        draw.text((20,y+170),f"문자 오류 {old['score']['edit_distance']} → {new['score']['edit_distance']} / 동일 PDF의 검사 영역에 한정",font=small,fill='#444444')
        draw.line((10,y+215,1490,y+215),fill='#cccccc')
    image.save(ROOT/'comparison.png')


if __name__=='__main__':main()
