"""Merge existing isolated examples without inventing independent validation.

The four validation classes recur within the one drawing. Tight isolated crops
have no certified physical-instance provenance; keep all their descendants out
of the benchmark. This conservative rule is not a claim that every such crop
actually overlaps validation. Other isolated classes have no labelled target
in the held-out scene regions. No additional augmentation or training occurs.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

from PIL import Image

HERE = Path(__file__).resolve().parent
ISOLATED = HERE / 'dataset_yolo26_boxreviewed_20261008'
SCENE = HERE / 'capture_train_val_20261008'
OUT = HERE / 'dataset_yolo26_train_val_20261008'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label(class_id, box):
    x1, y1, x2, y2 = box
    return f'{class_id} {(x1+x2)/1280:.9f} {(y1+y2)/1280:.9f} {(x2-x1)/640:.9f} {(y2-y1)/640:.9f}\n'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = load(ISOLATED / 'source_boxes.json')
    old_records = load(ISOLATED / 'index.json')
    scene = load(SCENE / 'split_manifest.json')
    classes = load(ISOLATED / 'classes.json')
    assert classes == scene['classes']
    val_classes = {a['class_name'] for t in scene['tiles'] if t['split'] == 'val' for a in t['annotations']}
    source_splits = {sid: ('quarantine' if s['class_name'] in val_classes else 'train')
                     for sid, s in sources.items()}
    # Lineage is required, not inferred from augmented filenames.
    families = defaultdict(list)
    for r in old_records:
        assert r['source_id'] in sources
        families[r['source_id']].append(r)
    assert set(families) == set(sources)
    assert all(Counter(r['kind'] for r in family) == {'original': 1, 'augmented': 16}
               for family in families.values())
    for split in ['train', 'val', 'quarantine']:
        (OUT / 'images' / split).mkdir(parents=True)
        (OUT / 'labels' / split).mkdir(parents=True)
    records = []
    for t in scene['tiles']:
        split = t['split']
        image_path = f"images/{split}/scene_{Path(t['image']).name}"
        label_path = f"labels/{split}/scene_{Path(t['label']).name}"
        shutil.copy2(SCENE / t['image'], OUT / image_path)
        shutil.copy2(SCENE / t['label'], OUT / label_path)
        records.append({'dataset': 'scene', 'source_id': t['parent_capture_id'],
                        'source_region_group': t['source_region_group'], 'kind': 'scene_tile',
                        'split': split, 'image': image_path, 'label': label_path,
                        'source_image': str(SCENE / t['image']), 'source_label': str(SCENE / t['label']),
                        'resize_applied': False})
    resized = 0
    for r in old_records:
        sid = r['source_id']
        split = source_splits[sid]
        # Source-relative identity makes names unique even for repeated stems.
        stem = f"isolated_{sid}_{r['kind']}_{Path(r['image']).stem}"
        image_path = f'images/{split}/{stem}.png'
        label_path = f'labels/{split}/{stem}.txt'
        original_image = ISOLATED / r['image']
        original_label = ISOLATED / r['label']
        with Image.open(original_image) as im:
            w, h = im.size
            transform = None
            if (w, h) == (640, 640):
                shutil.copy2(original_image, OUT / image_path)
                shutil.copy2(original_label, OUT / label_path)
            else:
                assert r['kind'] == 'original'
                ratio = min(640/w, 640/h)
                nw, nh = max(1, round(w*ratio)), max(1, round(h*ratio))
                px, py = (640-nw)//2, (640-nh)//2
                canvas = Image.new('RGB', (640, 640), 'white')
                canvas.paste(im.convert('RGB').resize((nw, nh), Image.Resampling.LANCZOS), (px, py))
                canvas.save(OUT / image_path)
                fields = original_label.read_text().split()
                assert len(fields) == 5 and int(fields[0]) == sources[sid]['class_id']
                cx, cy, bw, bh = map(float, fields[1:])
                old_box = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
                new_box = [old_box[0]*nw/w+px, old_box[1]*nh/h+py,
                           old_box[2]*nw/w+px, old_box[3]*nh/h+py]
                (OUT / label_path).write_text(label(int(fields[0]), new_box), encoding='utf-8')
                transform = {'input_size': [w, h], 'scale_xy': [nw/w, nh/h],
                             'padding_xy': [px, py], 'original_bbox': old_box, 'output_bbox': new_box}
                resized += 1
        records.append({'dataset': 'isolated', 'source_id': sid, 'kind': r['kind'],
                        'class_name': sources[sid]['class_name'], 'split': split,
                        'image': image_path, 'label': label_path,
                        'source_image': str(original_image), 'source_label': str(original_label),
                        'resize_applied': transform is not None, 'resize_transform': transform})
    hashes = defaultdict(set)
    lineage = defaultdict(set)
    instance_counts = {s: Counter() for s in ['train', 'val', 'quarantine']}
    for r in records:
        image_path, label_path = OUT/r['image'], OUT/r['label']
        with Image.open(image_path) as im:
            assert im.size == (640, 640)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[r['split']].add(digest)
        r['sha256_image'] = digest
        lineage[(r['dataset'], r['source_id'])].add(r['split'])
        if not r['resize_applied']:
            assert image_path.read_bytes() == Path(r['source_image']).read_bytes()
            assert label_path.read_bytes() == Path(r['source_label']).read_bytes()
        for line in label_path.read_text().splitlines():
            fields = line.split()
            assert len(fields) == 5 and fields[0] in classes
            cx, cy, w, h = map(float, fields[1:])
            assert w > 0 and h > 0
            assert cx-w/2 >= -1e-7 and cy-h/2 >= -1e-7
            assert cx+w/2 <= 1+1e-7 and cy+h/2 <= 1+1e-7
            instance_counts[r['split']][classes[fields[0]]] += 1
        if r.get('resize_transform'):
            t = r['resize_transform']; sx, sy = t['scale_xy']; px, py = t['padding_xy']
            b = t['output_bbox']
            inverse = [(b[0]-px)/sx, (b[1]-py)/sy, (b[2]-px)/sx, (b[3]-py)/sy]
            assert max(abs(a-b) for a, b in zip(inverse, t['original_bbox'])) < 1e-6
    assert all(len(v) == 1 for v in lineage.values())
    assert not hashes['train'] & hashes['val']
    assert set(instance_counts['val']) <= set(instance_counts['train'])
    assert len(records) == 2015
    assert len(list((OUT/'images').rglob('*.png'))) == len(records)
    assert len(list((OUT/'labels').rglob('*.txt'))) == len(records)
    summary = {'scene_images': 179, 'isolated_originals': 108, 'isolated_existing_augmented': 1728,
               'isolated_source_counts': dict(Counter(source_splits.values())),
               'isolated_image_counts': dict(Counter(r['split'] for r in records if r['dataset']=='isolated')),
               'combined_image_counts': dict(Counter(r['split'] for r in records)),
               'combined_total_images': len(records), 'originals_letterboxed_to_640': resized,
               'validation_classes': sorted(val_classes), 'same_original_family_cross_split': 0,
               'identical_image_hashes_train_val': 0, 'new_augmentation_applied': False,
               'training_started': False, 'all_images_640': True,
               'isolated_quarantine_reason': 'Unresolved physical-instance provenance for classes recurring in validation; possible overlap, not confirmed overlap.',
               'benchmark_scope': 'four-class region holdout of one drawing; not a full 38-class independent-drawing benchmark'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'classes.json').write_text(json.dumps(classes,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(SCENE/'split_manifest.json',OUT/'scene_split_manifest.json')
    shutil.copy2(ISOLATED/'source_boxes.json',OUT/'isolated_source_boxes.json')
    shutil.copy2(HERE/'merge_isolated_split.py',OUT/'merge_isolated_split.py')
    with (OUT/'isolated_source_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source_id','class_name','split','original_count','existing_augmented_count','reason'])
        writer.writeheader()
        for sid,s in sources.items():
            writer.writerow({'source_id':sid,'class_name':s['class_name'],'split':source_splits[sid],
                             'original_count':1,'existing_augmented_count':16,
                             'reason': 'physical_instance_unresolved_for_validation_class' if source_splits[sid]=='quarantine'
                             else 'target_class_absent_from_heldout_scene_regions; train_only'})
    with (OUT/'class_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class_id','class_name','train_instances','val_instances','quarantine_instances'])
        for cid,name in classes.items():writer.writerow([cid,name,*(instance_counts[s][name] for s in ['train','val','quarantine'])])
    for split in ['train','val']:
        (OUT/f'{split}.txt').write_text(''.join(f"./{r['image']}\n" for r in records if r['split']==split),encoding='utf-8')
    (OUT/'README.txt').write_text('''기존 개별 심볼 + 실제 도면 캡처 분리 결과

학습 1598장 / 검증 21장 / 보류 396장, 전부 640×640입니다.
기존 개별 원본 87장과 그 증강본 1392장은 학습용입니다.
기존 개별 원본 21장과 그 증강본 336장은 출처 불명확으로 보류입니다.
보류 데이터는 학습/검증에 사용하지 마세요. 삭제한 데이터는 없습니다.

7/11/12/14 개별 캡처는 주변 좌표/치수가 잘려 동일 물리 부품 여부를
확정할 수 없어 해당 클래스의 기존 개별 캡처 전체를 보수적으로 보류했습니다.
모두 실제 중복이라고 판정한 것은 아닙니다. 원본 도면 위치가 확인되면
독립 부품의 원본과 증강본을 함께 학습으로 복귀시킬 수 있습니다.
이번 단계에서 새로 검증으로 옮긴 개별 캡처는 없습니다.
검증은 기존의 21개 실제 도면 타일을 그대로 유지합니다.
나머지 클래스는 검증 영역에 라벨 대상이 없어 학습에만 사용합니다.

동일 원본의 증강본 16개는 원본과 동일 분할입니다.
이미 640인 이미지/라벨은 바이트 그대로 유지했습니다.
크기가 다른 원본 69장은 종횡비 유지 후 흰 여백으로 640을 맞추고
동일 변환을 박스 좌표에 적용했습니다. 원본 폴더는 수정하지 않았습니다.
새 증강이나 학습은 실행하지 않았습니다.

검증 클래스는 7/11/12/14입니다. 한 도면의 영역 분리 평가이므로
전체 38클래스 성능 또는 새 도면 성능을 나타내지 않습니다.
이전 개별 캡처 보류는 위치 추정을 억지로 확정하지 않기 위한 조치입니다.
도면 영역 분리 자체의 추정 좌표/수동 대응 한계는 scene_split_manifest.json에 기록했습니다.
원본/증강 관계와 분할 이유: isolated_source_assignments.csv
좌표 변환 및 파일 출처: index.json
클래스별 라벨 수량: class_distribution.csv
이미지 목록: train.txt / val.txt
''',encoding='utf-8-sig')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
