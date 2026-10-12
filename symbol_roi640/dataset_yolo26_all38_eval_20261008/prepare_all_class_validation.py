"""Create all-class capture holdout, keeping the region benchmark separate.

Different captures of one physical CAD symbol are NOT independent instances.
This dataset measures recognition of held-out captures of known drawing parts.
The four-class region holdout retains its original, more conservative scope.
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
SOURCE = HERE / 'dataset_yolo26_train_val_20261008'
OUT = HERE / 'dataset_yolo26_all38_eval_20261008'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    records = read(SOURCE/'index.json')
    sources = read(SOURCE/'isolated_source_boxes.json')
    classes = read(SOURCE/'classes.json')
    region_classes = set(read(SOURCE/'summary.json')['validation_classes'])
    bad = {sid for sid, s in sources.items()
           if s.get('capture_review_status') == 'recapture_recommended'}
    candidates = defaultdict(list)
    for sid, s in sources.items():
        if not sid.startswith('cad_') and sid not in bad and s['class_name'] not in region_classes:
            candidates[s['class_name']].append(sid)
    # Retain one non-CAD original family per available class for training.
    # Class 17 has no complete PDF isolated capture; use its scene training tiles.
    train_sources = {sorted(ids)[0] for ids in candidates.values()}
    partitions = ['train', 'val_all_classes', 'val_region4', 'quarantine']
    for split in partitions:
        (OUT/'images'/split).mkdir(parents=True)
        (OUT/'labels'/split).mkdir(parents=True)
    output = []
    source_partitions = {}
    for r in records:
        sid = r['source_id']
        if r['dataset'] == 'scene':
            split = {'train':'train', 'val':'val_region4', 'quarantine':'quarantine'}[r['split']]
            reason = 'existing scene-region partition'
            family_partition = split
        elif sid in bad:
            split = family_partition = 'quarantine'
            reason = 'existing incomplete-capture warning'
        elif sid in train_sources:
            split = family_partition = 'train'
            reason = 'training original with all its existing augmentations'
        else:
            family_partition = 'val_all_classes'
            split = 'val_all_classes' if r['kind'] == 'original' else 'quarantine'
            reason = ('held-out original capture; same physical part may occur in training'
                      if r['kind'] == 'original' else 'augmentation of held-out capture; excluded from training and evaluation')
        image = f"images/{split}/{Path(r['image']).name}"
        label = f"labels/{split}/{Path(r['label']).name}"
        shutil.copy2(SOURCE/r['image'], OUT/image)
        shutil.copy2(SOURCE/r['label'], OUT/label)
        assert (SOURCE/r['image']).read_bytes() == (OUT/image).read_bytes()
        assert (SOURCE/r['label']).read_bytes() == (OUT/label).read_bytes()
        output.append({**r, 'prior_split':r['split'], 'split':split, 'image':image, 'label':label,
                       'family_partition':family_partition, 'assignment_reason':reason})
        key = (r['dataset'],sid)
        assert key not in source_partitions or source_partitions[key] == family_partition
        source_partitions[key] = family_partition
    counts = Counter(r['split'] for r in output)
    labels = {s: Counter() for s in partitions}
    hashes = {s: set() for s in partitions}
    families = {s: set() for s in partitions}
    for r in output:
        s = r['split']
        with Image.open(OUT/r['image']) as im:
            assert im.size == (640,640)
        digest = hashlib.sha256((OUT/r['image']).read_bytes()).hexdigest()
        assert digest == r['sha256_image']
        hashes[s].add(digest)
        families[s].add((r['dataset'],r['source_id']))
        if s.startswith('val'):
            assert r['kind'] != 'augmented'
        for line in (OUT/r['label']).read_text().splitlines():
            fields = line.split()
            assert len(fields) == 5 and fields[0] in classes
            cx,cy,w,h = map(float,fields[1:])
            assert w>0 and h>0 and min(cx-w/2,cy-h/2)>=-1e-7 and max(cx+w/2,cy+h/2)<=1+1e-7
            labels[s][classes[fields[0]]] += 1
    for s in ['val_all_classes','val_region4']:
        assert not families['train'] & families[s]
        assert not hashes['train'] & hashes[s]
    assert set(labels['train']) == set(labels['val_all_classes']) == set(classes.values())
    assert set(labels['val_region4']) == region_classes
    # Preserve the prior region-holdout training policy for its four classes.
    assert not any(r['dataset']=='isolated' and r.get('class_name') in region_classes
                   and r['split']=='train' for r in output)
    assert len(output) == sum(counts.values()) == 2015
    assert len(list((OUT/'images').rglob('*.png'))) == len(output)
    assert len(list((OUT/'labels').rglob('*.txt'))) == len(output)
    summary = {'image_counts':dict(counts),'total_images':len(output),
               'training_class_count':len(labels['train']), 'all_class_validation_class_count':len(labels['val_all_classes']),
               'all_class_validation_min_instances_per_class':min(labels['val_all_classes'].values()),
               'all_class_validation_scope':'held-out captures of known drawing parts; physical instance independence is NOT established',
               'region_validation_scope':'held-out regions of the same drawing, four classes, with the earlier estimated-position limitations',
               'incomplete_isolated_sources_excluded':sorted(bad),
               'heldout_original_augmentations_excluded':sum(r['kind']=='augmented' and r['family_partition']=='val_all_classes' for r in output),
               'train_validation_original_family_overlap':0,'train_validation_identical_image_overlap':0,
               'all_images_640':True,'existing_images_labels_preserved':True,'new_augmentation_applied':False,'training_started':False}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(SOURCE/'classes.json',OUT/'classes.json')
    shutil.copy2(SOURCE/'scene_split_manifest.json',OUT/'scene_split_manifest.json')
    shutil.copy2(SOURCE/'isolated_source_boxes.json',OUT/'isolated_source_boxes.json')
    shutil.copy2(HERE/'prepare_all_class_validation.py',OUT/'prepare_all_class_validation.py')
    for s in ['train','val_all_classes','val_region4']:
        (OUT/f'{s}.txt').write_text(''.join(f"./{r['image']}\n" for r in output if r['split']==s),encoding='utf-8')
    for name, val in [('data_all38.yaml','val_all_classes'),('data_region4.yaml','val_region4')]:
        yaml = '# all38 is a known-part capture evaluation, not unseen-instance accuracy\n'
        yaml += 'path: '+json.dumps(OUT.as_posix(),ensure_ascii=False)+'\ntrain: images/train\nval: images/'+val+'\nnames:\n'
        yaml += ''.join(f'  {cid}: {json.dumps(c,ensure_ascii=False)}\n' for cid,c in classes.items())
        (OUT/name).write_text(yaml,encoding='utf-8')
    with (OUT/'class_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class_id','class_name',*partitions])
        for cid,c in classes.items():writer.writerow([cid,c,*(labels[s][c] for s in partitions)])
    with (OUT/'source_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','source_id','family_partition','originals_or_tiles','existing_augmented','reason'])
        for (dataset,sid),fp in source_partitions.items():
            group=[r for r in output if r['dataset']==dataset and r['source_id']==sid]
            writer.writerow([dataset,sid,fp,sum(r['kind']!='augmented' for r in group),sum(r['kind']=='augmented' for r in group),group[0]['assignment_reason']])
    (OUT/'README.txt').write_text('''38클래스 캡처 평가 + 4클래스 영역 평가

학습 680장: 38클래스 모두 포함.
전체 클래스 평가 73장: 38클래스 모두 포함, 원본 캡처만 사용.
기존 영역 평가 21장: 7/11/12/14의 검증 영역을 유지.
보류 1241장: 검증 원본의 증강본 1168장, 잘림 경고 원본 가족 34장,
             기존 도면 겹침 보류 타일 39장. 삭제하지 않았습니다.

data_all38.yaml: 38클래스 모두 지표를 계산하는 설정입니다.
이 평가는 같은 도면 부품을 다른 배경/크기로 캡처했을 때의 인식 확인입니다.
학습과 검증에 동일 물리 부품의 다른 캡처가 포함될 수 있습니다.
새 부품/새 도면의 일반화 성능을 뜻하지 않으며, 일부 클래스는 1장뿐입니다.
한 개의 총 mAP만으로 실제 도면 성능을 판단하지 마세요.

data_region4.yaml: 기존 4클래스 영역 검증용 설정입니다.
이 결과는 전체 클래스 캡처 평가와 따로 기록해야 합니다.
같은 도면 안의 영역 분리이며 추정 PDF 좌표/수동 대응의 한계는
scene_split_manifest.json에 남겨두었습니다.

원본과 그 증강본이 학습/검증에 나뉘지 않도록 처리했습니다.
검증 가족의 증강본은 보류 폴더에 두며 학습/검증 목록에서 제외했습니다.
잘림 경고가 있는 src_009, pdf_extra_17_001도 전체 가족을 보류했습니다.
7/11/12/14 개별 캡처는 학습에서 제외하여 기존 영역 검증 정책을 유지합니다.
완전한 개별 PDF 캡처가 없는 17 클래스는 실제 도면 타일로 학습합니다.
그 밖의 클래스는 개별 PDF 원본 가족 하나를 학습에 남기고
CAD 원본과 나머지 PDF 원본 캡처를 전체 클래스 평가에 사용합니다.

모든 이미지 640×640, 기존 이미지와 라벨은 변경하지 않았습니다.
새 증강 및 모델 학습은 실행하지 않았습니다.
class_distribution.csv: 클래스별 실제 라벨 수량
source_assignments.csv: 원본 가족 배치
index.json: 파일별 출처와 배치, 검증에서 제외한 증강본의 가족 연결
summary.json: 검사 결과 및 평가 범위
YAML path는 현재 프로젝트의 절대 경로입니다. 폴더 이동 시 수정하세요.
''',encoding='utf-8-sig')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
