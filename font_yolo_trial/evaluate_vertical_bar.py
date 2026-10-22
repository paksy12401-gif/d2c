"""Raw old/new YOLO comparison. No answer-based edits or production model replacement."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from evaluate_preserved_reader import match
from prepare import read_image, write_image
from train import read_labels, read_rows
from train_vertical_bar import OLD, OUT, NAMES


def score(results, paths, label_dir):
    counts = np.zeros(3, dtype=int)
    selected = {name: np.zeros(3, dtype=int) for name in ['|', 'I', 'l', '1']}
    matched = []
    for result, path in zip(results, paths):
        h, w = result.orig_shape
        truth = read_labels(label_dir/(path.stem+'.txt'), w, h)
        boxes = result.boxes.data.cpu().numpy()
        correct, fp, fn = match(boxes, truth)
        counts += [len(correct), fp, fn]
        matched.append(correct)
        for char, total in selected.items():
            cid = NAMES.index(char)
            subset = [t for t in truth if t[0] == cid]
            hit, extra, missing = match([b for b in boxes if int(b[5]) == cid], subset)
            total += [len(hit), extra, missing]
    def values(total):
        tp, fp, fn = map(int, total)
        return dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn))
    return dict(overall=values(counts), selected_classes={k:values(v) for k,v in selected.items()}), matched


def main():
    torch.set_num_threads(4)
    dest = OUT/'evaluation'
    dest.mkdir(exist_ok=True)
    paths = dict(before=OLD/'training/weights/best.pt', after=OUT/'training/weights/best.pt')
    models = {k: YOLO(str(p)) for k,p in paths.items()}
    assert models['after'].names == dict(enumerate(NAMES))
    report = dict(confidence=.1, iou=.5, model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        scope='Synthetic splits reuse the training glyph templates. The supplied screenshot is training material. Neither is independent real-drawing accuracy.', datasets={}, source={})
    for name, directory, prefix in [('old_alphanumeric', OLD.parent/'alphanumeric', ''), ('old_punctuation', OLD, ''), ('new_bar_scenes', OUT, 'bar_')]:
        images = sorted((directory/'images/test').glob(prefix+'*.png'))
        summary, matches = {}, {}
        for model_name, model in models.items():
            results = list(model.predict([str(p) for p in images], imgsz=384, batch=8, device=0, conf=.1, iou=.5, end2end=False, agnostic_nms=False, stream=True, verbose=False))
            assert len(results) == len(images)
            summary[model_name], matches[model_name] = score(results, images, directory/'labels/test')
        summary['previously_correct_glyphs_lost'] = sum(len(a-b) for a,b in zip(matches['before'], matches['after']))
        report['datasets'][name] = dict(images=len(images), **summary)
        print(json.dumps(dict(dataset=name, **summary)), flush=True)
    source = read_image(OUT/'source_vertical_bar.png')
    truth = [(67, np.asarray(box, dtype=float)) for box in json.loads((OUT/'manifest.json').read_text())['source_boxes']]
    for model_name, model in models.items():
        report['source'][model_name] = {}
        for view in ('native_padded', 'full_384'):
            image = source
            if view == 'native_padded':
                image = np.full((384,384,3), np.median(source.reshape(-1,3), axis=0).astype(np.uint8), np.uint8)
                image[:source.shape[0], :source.shape[1]] = source
            result = model.predict(image, imgsz=384, device=0, conf=.1, iou=.5, end2end=False, agnostic_nms=False, verbose=False)[0]
            boxes = result.boxes.data.cpu().numpy()
            hit, fp, fn = match(boxes, truth)
            write_image(dest/f'{model_name}_{view}.png', result.plot(line_width=1, font_size=10))
            report['source'][model_name][view] = dict(bar_tp=len(hit), fp=fp, fn=fn, rows=read_rows(result), detections=[dict(label=model.names[int(b[5])], confidence=float(b[4]), xyxy=b[:4].tolist()) for b in boxes])
    (dest/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = []
    for name, item in report['datasets'].items():
        for version in ['before','after']:
            overall=item[version]['overall'];bar=item[version]['selected_classes']['|']
            rows.append(f'<tr><td>{name}</td><td>{version}</td><td>{overall["precision"]:.3%}</td><td>{overall["recall"]:.3%}</td><td>{bar["tp"]}/{bar["tp"]+bar["fn"]}</td><td>{bar["fp"]}</td></tr>')
    page = '<!doctype html><meta charset="utf-8"><title>Vertical bar training</title><style>body{font:16px sans-serif;max-width:1000px;margin:32px auto}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #aaa}img{max-width:48%}</style>'
    page += '<h1>| 추가 학습 비교</h1><p>기존 I · l · 1 클래스는 유지하고 | 를 67번으로 추가했습니다. 기존 배포 모델은 교체하지 않았습니다.</p><p>아래 수치는 같은 글꼴 템플릿으로 만든 합성 검사입니다. 첨부 스크린샷도 학습에 사용했으므로 독립 평가가 아닙니다.</p>'
    page += '<table><tr><th>진단 세트</th><th>모델</th><th>전체 정밀도</th><th>전체 재현율</th><th>| 정답 검출</th><th>| 오탐</th></tr>'+''.join(rows)+'</table><h2>첨부 원본: native padded</h2>'
    page += '<img src="before_native_padded.png"><img src="after_native_padded.png"><pre>'+html.escape(json.dumps(report['source'], indent=2))+'</pre>'
    (dest/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(report['source']), flush=True)


if __name__ == '__main__':
    main()
