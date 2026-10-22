"""Compare the actual opt-in reader with its unchanged 62+67-class baseline."""
import json

import numpy as np
import torch

from evaluate_preserved_reader import match
from prepare import read_image
from preserved_reader import merge_predictions
from train import read_labels
from train_vertical_bar import OLD, OUT
from vertical_bar_reader import VerticalBarReader


def write_report(report):
    rows = []
    for name,item in report['datasets'].items():
        bar = item['bar']
        rows.append(f'<tr><td>{name}</td><td>{item["images"]}</td><td>{bar["tp"]}/{bar["tp"]+bar["fn"]}</td><td>{bar["fp"]}</td><td>{len(item["previously_correct_old_glyphs_lost"])}</td></tr>')
    page = '''<!doctype html><meta charset="utf-8"><title>| 추가 학습 결합 검사</title>
<style>body{font:16px sans-serif;max-width:960px;margin:32px auto;line-height:1.7}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}</style>
<h1>세로줄 기호 | 추가 학습</h1>
<p>I · l · 1과 다른 새 클래스 67번으로 12회 추가 학습했습니다. 기존 클래스 번호는 유지했습니다.</p>
<p>새 모델 전체 교체에서는 기존 문자 일부 손실이 관측되어, 기존 62/67클래스 모델을 유지한 결합 실행기를 별도로 만들었습니다. 새 모델의 | 후보 중 점수 0.6 이상이며 원본 획이 단순한 세로 막대로 확인되는 경우만 반영합니다. 기존 DWG와 기본 실행기는 교체하지 않았습니다.</p>
<h2>실제 결합 실행기 검사</h2><p>같은 글꼴 템플릿으로 만든 합성 검사입니다. 다른 실제 도면의 정확도를 의미하지 않습니다. 기존 260장에서는 전체 결합 전후 결과가 동일했습니다.</p>
<table><tr><th>검사 세트</th><th>이미지</th><th>| 검출/정답 수</th><th>| 오탐</th><th>기존 정답 손실 이미지</th></tr>'''+''.join(rows)+'''</table>
<h2>첨부 원본</h2><p>3개 모두 |로 검출했습니다. 점수 약 0.924~0.940. 이 이미지는 학습에 사용했으므로 독립 검증이 아닙니다.</p><img src="combined_source/detected.png">
<p>원본 작은 이미지를 바로 384px로 확대하면 이번 원본 검사는 실패했습니다. 결합 실행기는 원본 크기의 384px 타일로 |를 검사합니다. 해당 확대 실패도 <a href="report.html">모델 단독 전후 비교</a>에 기록했습니다.</p>
<p>새 실행 파일: font_yolo_trial/vertical_bar_reader.py<br>새 가중치: font_yolo_trial/vertical_bar/training/weights/best.pt<br><a href="combined_results.json">결합 검사 수치</a></p>'''
    (OUT/'evaluation/combined_report.html').write_text(page,encoding='utf-8')


def main():
    torch.set_num_threads(4)
    reader = VerticalBarReader()
    report = dict(scope='Same-template synthetic diagnostic; no independent drawing accuracy claim.', datasets={})
    for name, directory, prefix in [('old_alphanumeric',OLD.parent/'alphanumeric',''), ('old_punctuation',OLD,''), ('new_bar_scenes',OUT,'bar_')]:
        totals = {key:np.zeros(3,dtype=int) for key in ['before','after','bar']}
        images = sorted((directory/'images/test').glob(prefix+'*.png'))
        lost, changes, negative_fp = [], 0, 0
        for path in images:
            image = read_image(path)
            result, baseline, details = reader.predict(image,path,'full',384)
            old, _ = merge_predictions(baseline.boxes.cpu().numpy().data.tolist(),details['symbols'],details['cleaned'])
            new = result.boxes.cpu().numpy().data.tolist()
            truth = read_labels(directory/'labels/test'/(path.stem+'.txt'),384,384)
            before, fp, fn = match(old,truth)
            totals['before'] += [len(before),fp,fn]
            after, fp, fn = match(new,truth)
            totals['after'] += [len(after),fp,fn]
            old_lost = [j for j in before-after if truth[j][0] < 67]
            if old_lost:
                lost.append(dict(image=path.name,truth_indices=old_lost))
            bars = [b for b in new if int(b[5])==67]
            correct, fp, fn = match(bars,[t for t in truth if t[0]==67])
            totals['bar'] += [len(correct),fp,fn]
            changes += len(details['vertical_bar']['accepted'])
            if not truth:
                negative_fp += len(bars)
        def values(total):
            tp,fp,fn=map(int,total)
            return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))
        report['datasets'][name] = dict(images=len(images), **{k:values(v) for k,v in totals.items()}, previously_correct_old_glyphs_lost=lost,accepted_bars=changes,bar_false_positives_on_empty_drawing_scenes=negative_fp)
        print(json.dumps(dict(dataset=name,**report['datasets'][name])),flush=True)
    (OUT/'evaluation/combined_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_report(report)


if __name__ == '__main__':
    main()
