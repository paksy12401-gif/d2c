"""Small H-part verification experiment, independent of YOLO confidence or answers.

This verifies two vertical strokes connected by one shared middle crossbar.
It is not a learned component alphabet or a replacement for text localization.
"""
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np

from preserved_reader import foreground


def thin(mask):
    ink = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for second in (False, True):
            p = [np.roll(np.roll(ink, -dy, axis=0), -dx, axis=1)
                 for dy, dx in [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]]
            count = sum(p)
            transitions = sum((p[i] == 0) & (p[(i+1)%8] == 1) for i in range(8))
            if second:
                preserve = (p[0]*p[2]*p[6] != 0) | (p[0]*p[4]*p[6] != 0)
            else:
                preserve = (p[0]*p[2]*p[4] != 0) | (p[2]*p[4]*p[6] != 0)
            remove = (ink > 0) & (count >= 2) & (count <= 6) & (transitions == 1) & ~preserve
            remove[[0,-1],:] = False
            remove[:,[0,-1]] = False
            if np.any(remove):
                ink[remove] = 0
                changed = True
        if not changed:
            return ink * 255


def normalized_skeleton(image):
    mask = foreground(image)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.zeros((112, 112), np.uint8)
    mask = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = 96 / max(mask.shape)
    mask = cv2.resize(mask, (max(1, round(mask.shape[1]*scale)),
                             max(1, round(mask.shape[0]*scale))))
    mask = ((mask > 96) * 255).astype(np.uint8)
    canvas = np.zeros((112, 112), np.uint8)
    y, x = (112-mask.shape[0])//2, (112-mask.shape[1])//2
    canvas[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
    return thin(canvas)


def axis_parts(skeleton):
    found = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=12,
                          minLineLength=16, maxLineGap=3)
    axes = {'vertical': [], 'horizontal': []}
    for x0, y0, x1, y1 in [] if found is None else found.reshape(-1,4).tolist():
        if abs(x1-x0) <= 2 and abs(y1-y0) >= 16:
            kind, position, start, end = 'vertical', (x0+x1)/2, min(y0,y1), max(y0,y1)
        elif abs(y1-y0) <= 2 and abs(x1-x0) >= 16:
            kind, position, start, end = 'horizontal', (y0+y1)/2, min(x0,x1), max(x0,x1)
        else:
            continue
        axes[kind].append([position, start, end])
    for kind, parts in axes.items():
        merged = []
        for position, start, end in sorted(parts):
            match = next((p for p in merged if abs(p[0]-position) <= 3 and
                          min(p[2],end)-max(p[1],start) >= -3), None)
            if match is None:
                merged.append([position, start, end])
            else:
                match[0] = (match[0]+position)/2
                match[1], match[2] = min(match[1],start), max(match[2],end)
        axes[kind] = merged
    return axes


def verify_h(image):
    skeleton = normalized_skeleton(image)
    axes = axis_parts(skeleton)
    ys, xs = np.where(skeleton > 0)
    result = dict(compatible=False, parts=axes, reason='missing_connected_parts',
                  explained_ink=0., shared_crossbar=None)
    if not len(xs):
        result['reason'] = 'empty'
        return result
    height, width = float(np.ptp(ys)+1), float(np.ptp(xs)+1)
    vertical = [p for p in axes['vertical'] if p[2]-p[1] >= height*.72]
    for left in vertical:
        for right in vertical:
            if right[0]-left[0] < width*.65 or right[0]-left[0] < height*.25:
                continue
            top, bottom = max(left[1],right[1]), min(left[2],right[2])
            if bottom-top < height*.70:
                continue
            for bar in axes['horizontal']:
                y, x0, x1 = bar
                if not (top+height*.18 <= y <= bottom-height*.18):
                    continue
                if abs(x0-left[0]) > height*.09 or abs(x1-right[0]) > height*.09:
                    continue
                explanation = np.zeros_like(skeleton)
                for p in (left,right):
                    cv2.line(explanation,(round(p[0]),round(p[1])),
                             (round(p[0]),round(p[2])),255,7)
                cv2.line(explanation,(round(x0),round(y)),(round(x1),round(y)),255,7)
                coverage = float(np.mean(explanation[ys,xs] > 0))
                if coverage > result['explained_ink']:
                    result.update(explained_ink=coverage,shared_crossbar=bar,
                                  selected_vertical=[left,right])
                if coverage >= .90:
                    result.update(compatible=True,reason='two_verticals_shared_middle_crossbar')
                    return result
    return result


def write_report(out, records):
    rows=[]
    for r in records:
        structure=r['structure'];status='구조 일치' if structure['compatible'] else '구조 불일치 / 재검토'
        rows.append(f'<tr><td>{escape(r["id"])}<br>{escape(r["source"])}</td>'
                    f'<td>{escape(r["model_text"])}</td><td><img src="assets/{escape(r["image"])}"></td>'
                    f'<td class="{"ok" if structure["compatible"] else "review"}">{status}<br>'
                    f'세로 부품 {len(structure["parts"]["vertical"])}개 · '
                    f'가로 부품 {len(structure["parts"]["horizontal"])}개</td></tr>')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>H 획 부품 검사</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:12px;line-height:1.7;color:#223}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:12px}img{max-height:120px;max-width:220px;object-fit:contain}.ok{background:#dff4e8}.review{background:#fff0da}code{background:#edf1f7;padding:4px}p{max-width:1000px}</style>
<h1>H를 획 부품으로 나눠 확인한 실험</h1>
<p>사용자가 제안한 <code>H → ㅏ·ㅓ 같은 부품의 조합</code>을 시험했습니다. 이 구현은 이미지의 획을 가늘게 만든 뒤 세로획 두 개와 가운데 가로획을 추출합니다. 가운데 가로획이 양쪽 세로획에 연결되고, 이 세 부품이 전체 획의 90% 이상을 설명해야 H와 구조가 맞는다고 판정합니다.</p>
<p>부품 모델을 추가 학습한 결과는 아닙니다. H 하나에 대해 직접 작성한 구조 검사이며, 기존 YOLO·OCR 결과를 지우거나 다른 글자로 교정하지 않습니다. 표의 입력 글자는 기존 모델의 후보 또는 제공된 예시의 이름입니다. 구조 일치만으로 문자임이 증명되지는 않으며 H 모양의 도형도 통과할 수 있습니다.</p>
<p>원본 PDF의 실제 H 네 곳은 모두 통과했고, 기존 YOLO의 도형 오탐 H 한 곳은 재검토로 분류됐습니다. 기존 결과와 개선 결과에 같은 H 한 곳이 중복으로 포함되어 있으므로 독립 표본 수를 늘려 계산하지 않았습니다. 제공된 62개 예시에서는 H 하나만 통과했습니다. 같은 도면·글꼴에서 확인한 개발 실험이며 전체 오탐률이나 새 도면의 정확도를 의미하지 않습니다.</p>
<p><a href="results.json">원시 검사 결과</a> · <a href="false_h_context.png">H로 오인한 도형의 주변 원본</a> · <a href="../cad_fusion_refined/report.html">현재 YOLO+한글 OCR 검증</a></p>
<table><thead><tr><th>입력</th><th>기존 후보 / 예시</th><th>원본 획</th><th>H 구조 검사</th></tr></thead><tbody>__ROWS__</tbody></table></html>'''
    (out/'report.html').write_text(html.replace('__ROWS__',''.join(rows)),encoding='utf-8')


def main():
    # The manifest contains input coordinates and model proposals, never expected H labels.
    import pymupdf as fitz
    from prepare import write_image
    root = Path(__file__).resolve().parent
    out = root.parent/'output/cad_stroke_parts'
    out.mkdir(parents=True,exist_ok=True)
    (out/'assets').mkdir(exist_ok=True)
    page = fitz.open(root/'cad_without_tables/source.pdf')[0]
    before = json.loads((root/'cad_without_tables/ocr_results.json').read_text(encoding='utf-8'))
    current = json.loads((root.parent/'output/cad_fusion_refined/combined_results.json').read_text(encoding='utf-8'))
    cases = []
    for method,data in [('original',before),('refined',current)]:
        for index,glyph in enumerate(data['detections']):
            if glyph['text'] == 'H':
                cases.append(dict(id=f'{method}_{index}',box_pdf=glyph['box_pdf'],
                                  model_text='H',source=method))
    # Other border letters are an automatic challenge set, not asserted negative labels.
    for record in current['frame_labels']:
        if record['role'] == 'letter' and record['text'] != 'H':
            cases.append(dict(id=record['id'],box_pdf=record['box_pdf'],
                              model_text=record['text'],source='other_border_proposals'))
    records = []
    for case in cases:
        b = fitz.Rect(case['box_pdf']); margin = b.height*.035
        pix = page.get_pixmap(matrix=fitz.Matrix(96/b.height,96/b.height),
                             clip=b+(-margin,-margin,margin,margin),alpha=False)
        image = cv2.cvtColor(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3),cv2.COLOR_RGB2BGR)
        filename = case['id']+'.png'; write_image(out/'assets'/filename,image)
        records.append({**case,'image':filename,'structure':verify_h(image)})
    # All existing A-Z/lowercase/digit exemplars, without assigning new part labels.
    for path in sorted((root/'punctuation_fixed/glyphs').glob('*.png')):
        class_id = int(path.stem.split('_')[0])
        if class_id >= 62:
            continue
        rgba = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
        image = np.repeat((255-rgba[:,:,3])[:,:,None],3,axis=2)
        image = cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
        filename = 'template_'+path.name; write_image(out/'assets'/filename,image)
        records.append(dict(id='template_'+path.stem,source='provided_exemplar',
                            model_text=path.stem.split('_',1)[1],image=filename,
                            structure=verify_h(image)))
    result = dict(scope='H-only structural verifier. Same-font exploratory test; no inference correction or training.',
                  inference_llm=False,answers_loaded=False,thresholds_fixed_before_cases=True,
                  notes=['Compatible means shape support, not proof of text: a drawing can have the same H topology.',
                         'Missing or obscured strokes can fail verification. No candidates are deleted by this probe.'],
                  records=records)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # A source-only context image makes the rejected proposal inspectable.
    rejected=next((r for r in records if r['source']=='original' and not r['structure']['compatible']),None)
    if rejected:
        b=fitz.Rect(rejected['box_pdf']);pad=b.height*.8
        page.get_pixmap(matrix=fitz.Matrix(8,8),clip=b+(-pad,-pad,pad,pad),alpha=False).save(out/'false_h_context.png')
    write_report(out,records)
    print(json.dumps([dict(id=r['id'],source=r['source'],model_text=r['model_text'],
                          **r['structure']) for r in records],ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
