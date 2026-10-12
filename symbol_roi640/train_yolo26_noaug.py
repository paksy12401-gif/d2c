"""Local YOLO26s trial, with a verified deterministic image transform pipeline.

Run from the notebook or .venv-ml. This never calls OpenAI or uploads drawings.
Only the official public pretrained weights may need an initial download.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE/'dataset_yolo26_all38_eval_20261008'
KNOWN = HERE/'known_drawing_training_20261008'
RUN_ROOT = HERE/'training_yolo26_noaug_20261008'
os.environ['YOLO_CONFIG_DIR'] = str(RUN_ROOT/'runtime_settings')
os.environ['YOLO_VERBOSE'] = 'false'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'DISABLED'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionTrainer
from PIL import Image, ImageDraw, ImageFont

# Keep this project's library settings inside the workspace. Do not reuse API
# credentials or experiment logger settings from the user's global settings.
settings.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb'] if k in settings})
settings.update({'api_key':'','openai_api_key':''})


def write_bytes_durable(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def write_json(path, value):
    write_bytes_durable(path,json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item()).encode('utf-8'))


def flush_checkpoint(path):
    with path.open('r+b') as f:
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, f'Checkpoint integrity failed: {path}'


class NoAugDetectionTrainer(DetectionTrainer):
    def save_model(self):
        saved = super().save_model()
        if saved:
            for path in (self.last,self.best):
                if path.exists():
                    flush_checkpoint(path)
            # Keep the last complete checkpoint if final optimizer stripping is interrupted.
            write_bytes_durable(self.wdir/'last_verified.pt',self.last.read_bytes())
        return saved

    def final_eval(self):
        super().final_eval()
        for path in (self.last,self.best):
            if path.exists():
                flush_checkpoint(path)

    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path,mode,batch)
        if mode == 'train':
            # augment=False passed to model.train controls prediction TTA, not
            # dataset augmentation. Use the library's actual non-aug pipeline.
            dataset.augment = False
            dataset.transforms = dataset.build_transforms(hyp=self.args)
            transforms = [type(t).__name__ for t in dataset.transforms.transforms]
            assert transforms == ['LetterBox','Format'],transforms
            first,second = dataset[0],dataset[0]
            assert torch.equal(first['img'],second['img'])
            assert torch.equal(first['bboxes'],second['bboxes'])
            write_json(Path(self.save_dir)/'no_augmentation_audit.json',{
                'dataset_augment':dataset.augment,'transforms':transforms,
                'repeated_sample_images_identical':True,'repeated_sample_boxes_identical':True,
                'existing_offline_augmented_images_used':True,'new_online_augmentation':False})
        return dataset


def disable_external_callbacks(instance):
    # Includes Platform callbacks that may otherwise use an environment API key.
    for event, callbacks in instance.callbacks.items():
        instance.callbacks[event] = [f for f in callbacks
                                     if not f.__module__.startswith('ultralytics.utils.callbacks.')
                                     and f.__module__ != 'ultralytics.utils.events']


def epoch_progress(trainer):
    losses = trainer.label_loss_items(trainer.tloss,prefix='train')
    row = {'epoch':min(trainer.epoch+1,trainer.epochs),'epochs':trainer.epochs,
           'elapsed_seconds':round(time.time()-trainer.train_time_start,1),
           'losses':{k:float(v) for k,v in losses.items()}}
    write_json(Path(trainer.save_dir)/'progress.json',row)
    print('PROGRESS '+json.dumps(row),flush=True)


def make_preview(model,folder,split,files=None,label_paths=None):
    files = sorted((DATA/'images'/split).glob('*.png')) if files is None else files
    previews = folder/'prediction_previews';previews.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    rows = []
    for result in model.predict(source=[str(p) for p in files],imgsz=640,device=0,batch=8,
                                conf=.25,iou=.7,rect=False,augment=False,verbose=False,stream=True):
        p = Path(result.path)
        im = Image.open(p).convert('RGB')
        truth = im.copy();draw = ImageDraw.Draw(truth)
        gt = []
        label_path = Path(label_paths[str(p.resolve())]) if label_paths else DATA/'labels'/split/p.with_suffix('.txt').name
        for line in label_path.read_text().splitlines():
            cid,cx,cy,w,h=map(float,line.split());cid=int(cid)
            box = [(cx-w/2)*640,(cy-h/2)*640,(cx+w/2)*640,(cy+h/2)*640]
            gt.append({'class_id':cid,'bbox':box})
            draw.rectangle(box,outline='#00aa88',width=3)
            draw.text((box[0],max(0,box[1]-18)),str(model.names[cid]),font=font,fill='#007766')
        # Counts use a fixed, visible operating threshold; AP sweeps confidence.
        pred = [{'class_id':int(c),'confidence':float(conf),'bbox':b.tolist()}
                for b,c,conf in zip(result.boxes.xyxy.cpu(),result.boxes.cls.cpu(),result.boxes.conf.cpu())]
        unused=set(range(len(gt)));tp=0
        for a in sorted(pred,key=lambda x:x['confidence'],reverse=True):
            best=None;best_iou=0
            for j in unused:
                b=gt[j]
                if a['class_id']!=b['class_id']:continue
                x1,y1,x2,y2=a['bbox'];u1,v1,u2,v2=b['bbox']
                intersection=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
                union=(x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-intersection
                iou=intersection/union if union>0 else 0
                if iou>best_iou:best_iou=iou;best=j
            if best is not None and best_iou>=.5:unused.remove(best);tp+=1
        canvas=Image.new('RGB',(1280,675),'white');canvas.paste(truth,(0,35))
        plotted=Image.fromarray(result.plot()[...,::-1]);canvas.paste(plotted,(640,35))
        d=ImageDraw.Draw(canvas);d.text((10,8),'GROUND TRUTH',font=font,fill='black')
        d.text((650,8),f'PREDICTION conf>=0.25 | TP {tp} FP {len(pred)-tp} FN {len(unused)}',font=font,fill='black')
        preview=previews/(p.stem+'.jpg');canvas.save(preview,quality=91)
        rows.append({'image':p.name,'gt_count':len(gt),'prediction_count':len(pred),'tp':tp,
                     'fp':len(pred)-tp,'fn':len(unused),'preview':str(preview.relative_to(folder)),
                     'ground_truth':gt,'predictions':pred})
    write_json(folder/'predictions_conf025.json',rows)
    return {'confidence':.25,'match_iou':.5,'tp':sum(r['tp'] for r in rows),
            'fp':sum(r['fp'] for r in rows),'fn':sum(r['fn'] for r in rows)}


def evaluate(weights,run,known_drawing=False):
    reports=run/'reports';reports.mkdir(exist_ok=True)
    model=YOLO(str(weights));model.add_callback('on_val_start',disable_external_callbacks)
    model.add_callback('on_predict_start',disable_external_callbacks)
    summaries={}
    checks=[('all38','val_all_classes',DATA/'data_all38.yaml'),
            ('region4','val_region4',DATA/'data_region4.yaml')]
    if known_drawing:
        checks.append(('all_scene',None,KNOWN/'data_known_drawing.yaml'))
    for title,split,config in checks:
        if split is None:
            files=[Path(s) for s in (KNOWN/'check_scene.txt').read_text(encoding='utf-8').splitlines()]
            label_paths={str((DATA/r['image']).resolve()):str(DATA/r['label']) for r in read_json(DATA/'index.json')}
        else:
            files=sorted((DATA/'images'/split).glob('*.png'));label_paths=None
        metrics=model.val(data=str(config),imgsz=640,batch=8,device=0,workers=0,
                          conf=.001,iou=.7,rect=False,augment=False,plots=True,save_json=False,
                          project=str(reports),name=title,exist_ok=True,verbose=False)
        folder=reports/title
        table=[{k:(v.item() if hasattr(v,'item') else v) for k,v in row.items()} for row in metrics.summary()]
        write_json(folder/'per_class.json',table)
        with (folder/'per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
            f.flush();os.fsync(f.fileno())
        summary={'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
                 'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),
                 'evaluated_classes':len(table),'images':len(files),'input_size':[640,640],'rect':False,
                 'fixed_confidence_counts':make_preview(model,folder,split,files,label_paths)}
        write_json(folder/'metrics.json',summary);summaries[title]=summary
        print('EVALUATION '+title+' '+json.dumps(summary),flush=True)
    summary={'model':'YOLO26s Detect','weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
             'evaluations':summaries,'new_online_augmentation':False,
             'all38_scope':'known-part capture evaluation; not independent physical instances or new drawings',
             'region4_scope':'same-drawing region holdout; earlier estimated-position limitations apply',
             'training_epochs':len(list(csv.DictReader((run/'results.csv').open(encoding='utf-8')))),
             'known_drawing_fit':known_drawing}
    if known_drawing:
        summary['all38_scope']=summary['region4_scope']='Rechecks on training-included images of the specified drawing; not independent validation.'
    write_json(run/'result_summary.json',summary)
    lines=['# YOLO26s 학습 결과','',f"학습 횟수: {summary['training_epochs']} epochs. 추가 온라인 증강 없음.",'',
           '| 평가 | 이미지 | 클래스 | Precision | Recall | mAP50 | mAP50–95 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for title,s in summaries.items():
        lines.append(f"| {title} | {s['images']} | {s['evaluated_classes']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['mAP50']:.3f} | {s['mAP50_95']:.3f} |")
    lines += ['', 'AP는 정답률과 다른 검출 평가 지표입니다. mAP50은 정답박스와 IoU 0.5 이상을 기준으로 평가합니다.',
              '38클래스 평가는 같은 물리 부품의 다른 캡처가 포함될 수 있고, 일부 클래스의 검증 자료는 1장뿐입니다.',
              '4클래스 영역 검증도 한 도면 안에서 평가하므로 새 도면 일반화 성능을 뜻하지 않습니다.',
              '학습은 정해진 횟수만 실행하고 마지막 epoch의 last.pt를 평가했습니다. 검증 점수를 보고 반복 선택하지 않았습니다.',
              '', 'reports/*/per_class.csv: 클래스별 결과', 'reports/*/prediction_previews/: 정답과 예측을 나란히 표시',
              'reports/*/predictions_conf025.json: confidence 0.25, 매칭 IoU 0.5에서 TP/FP/FN',
              'no_augmentation_audit.json: 실제 입력 변환 검사', 'weights/last.pt: 이번 결과의 학습 가중치']
    if known_drawing:
        lines += ['', '이번 실행은 해당 도면 맞춤 학습입니다. 위 모든 평가 이미지를 학습에도 사용했습니다.',
                  '이전 실행과 수치를 비교할 수 있지만, 이번 수치는 독립 검증 성능이 아니라 해당 도면의 재검출 결과입니다.',
                  '학습에 복귀한 확대 심볼과 주변 도면 타일을 함께 사용해 크기/배경의 범위를 보완했습니다.']
    write_bytes_durable(run/'RESULT.md',('\n'.join(lines)+'\n').encode('utf-8'))
    return summary


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_training(epochs=50,batch=8,name='baseline_noaug',evaluate_only=False,known_drawing=False):
    run=RUN_ROOT/name
    if evaluate_only:
        return evaluate(run/'weights/last.pt',run,known_drawing)
    if (run/'result_summary.json').exists():
        print('이미 완료된 실행 결과를 반환합니다. 새 학습은 다른 name을 사용하세요.',flush=True)
        return read_json(run/'result_summary.json')
    if run.exists():
        raise FileExistsError(f'중단된 결과를 덮어쓰지 않습니다: {run}')
    assert torch.cuda.is_available(),'GPU를 사용할 수 없습니다. VS Code 커널은 .venv-ml을 선택하세요.'
    torch.set_num_threads(4)
    starting_weights=RUN_ROOT/'baseline_noaug/weights/last.pt' if known_drawing else PROJECT/'models/yolo26s.pt'
    model=YOLO(str(starting_weights))
    model.add_callback('on_pretrain_routine_start',disable_external_callbacks)
    model.add_callback('on_fit_epoch_end',epoch_progress)
    config=KNOWN/'data_known_drawing.yaml' if known_drawing else DATA/'data_all38.yaml'
    args=dict(data=str(config),epochs=epochs,imgsz=640,batch=batch,device=0,
              workers=0,amp='bf16',cache=False,optimizer='AdamW',lr0=.0005 if known_drawing else .001,lrf=.01,
              seed=42,deterministic=True,patience=0,val=False,plots=True,save=True,
              project=str(RUN_ROOT),name=name,exist_ok=False,verbose=False,
              mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,
              shear=0.,perspective=0.,flipud=0.,fliplr=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
              bgr=0.,multi_scale=0.,erasing=0.,auto_augment=None,augment=False,close_mosaic=0)
    environment={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    model.train(trainer=NoAugDetectionTrainer,**args)
    write_json(run/'environment.json',environment)
    write_json(run/'requested_training_config.json',args)
    assert read_json(run/'no_augmentation_audit.json')['new_online_augmentation'] is False
    return evaluate(run/'weights/last.pt',run,known_drawing)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--name',default='baseline_noaug')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--known-drawing',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run_training(args.epochs,args.batch,args.name,args.evaluate_only,args.known_drawing),ensure_ascii=False,indent=2),flush=True)
