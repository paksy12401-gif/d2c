"""Validate and package original-coordinate proposals for owner review."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from PIL import Image
from render_capture_review import display_image,render

HERE=Path(__file__).resolve().parent
ROOT=HERE/'capture_box_review_20261008'

def main():
    data=json.loads((ROOT/'proposals.json').read_text(encoding='utf-8'))
    render()
    (ROOT/'display').mkdir(exist_ok=True)
    (ROOT/'labels_draft_NOT_APPROVED').mkdir(exist_ok=True)
    rows=[]
    for c in data['captures']:
        assert c['status']=='unreviewed'
        original=Path(r'C:\Users\Win11Pro\Desktop\capture')/c['original_name']
        assert (ROOT/'images'/c['filename']).read_bytes()==original.read_bytes()
        c['source_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
        display_image(c).save(ROOT/'display'/c['filename'])
        lines=[]
        for obj in c['objects']:
            x1,y1,x2,y2=obj['bbox_xyxy_px'];w,h=c['width'],c['height']
            assert 0<=x1<x2<=w and 0<=y1<y2<=h,(c['capture_id'],obj)
            assert data['classes'][str(obj['class_id'])]==obj['class_name']
            assert obj['status']=='unreviewed'
            lines.append(f"{obj['class_id']} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            rows.append(dict(capture_id=c['capture_id'],original_name=c['original_name'],box_id=obj['box_id'],
                class_name=obj['class_name'],class_id=obj['class_id'],x1=x1,y1=y1,x2=x2,y2=y2,
                status='unreviewed',touches_edge=obj['touches_edge']))
        (ROOT/'labels_draft_NOT_APPROVED'/Path(c['filename']).with_suffix('.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'proposals.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'boxes_to_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(HERE/'template_class_reference.png',ROOT/'class_reference.png')
    shutil.copy2(HERE/'dataset_yolo26_boxreviewed_20261008'/'classes.json',ROOT/'classes.json')
    template=(HERE/'capture_review_page.html').read_text(encoding='utf-8')
    embedded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    (ROOT/'review.html').write_text(template.replace('__REVIEW_DATA__',embedded),encoding='utf-8')
    (ROOT/'README.txt').write_text('''정답박스 검수 패키지 — 2026-10-08

110장 / 221개 박스 후보 / 기존 38개 클래스 ID 유지
원본 이미지의 크기와 파일 내용을 유지했습니다.

1. 압축을 풀고 review.html을 Edge 또는 Chrome으로 여세요.
2. 클래스, 박스 범위, 빠진 심볼을 확인하세요.
3. 박스를 클릭하여 클래스 변경, 이동/모서리 크기 조정, 삭제가 가능합니다.
4. 빠진 심볼은 [박스 그리기]로 드래그하고 클래스를 지정하세요.
5. 한 장의 모든 대상이 맞으면 [이 이미지 검수 완료]를 누르세요.
6. 끝나면 [검수 JSON 다운로드]로 저장하고 해당 JSON을 전달해주세요.
   브라우저 자동 저장은 보조 기능입니다. 반드시 JSON을 내려받으세요.

간단히 채팅으로 수정 요청해도 됩니다.
예: 025-01은 왼쪽으로 더 넓혀 / 046-02 클래스는 23이 맞음.

images/: 원본 파일을 그대로 복사한 이미지 (파일명만 capture_001 등으로 통일)
annotated/: 박스 ID와 클래스 후보가 표시된 전체 해상도 이미지
sheets/: 12장씩 묶은 목록 (축소는 검수 표시용이며 학습 원본에 영향 없음)
display/: 어두운 CAD 캡처 대비 보정 보기, 검수 전용
classes.json / class_reference.png: 기존 클래스 번호와 형상
proposals.json / boxes_to_review.csv: 원본 픽셀 좌표 및 검수 상태
labels_draft_NOT_APPROVED/: YOLO 형식의 초안. 검수 완료 전 학습에 쓰지 마세요.

생성 방식: 로컬 색상/연결 성분 추출과 기존 템플릿 비교 후 시각적으로
클래스 후보 및 합쳐진 부품 박스를 정리했습니다. 학습된 YOLO 추론이 아닙니다.
템플릿 거리 값은 확률이나 신뢰도 점수가 아닙니다.
누락 및 경계 잘림의 완전한 판정은 아직 끝나지 않았습니다.

블록 하나당 박스 하나. 20+6, 21+5, 22+3은 한 박스로 묶었습니다.
심볼 본체/자체 연결부를 포함하고 외부 배선, 치수, 동그라미 번호는 제외합니다.
검은 커넥터 참고 그림의 부품번호 클래스도 포함합니다.
067번은 14번 두 심볼의 상단이 원본에서 잘려 재캡처가 권장됩니다.
기타 이미지의 경계에 닿은 박스도 잘림 여부를 확인해주세요.

640 변환, 학습/검증 분리, 증강 및 YOLO 학습은 아직 하지 않았습니다.
검수가 끝나면 이미지와 박스 좌표를 함께 변환합니다.
원본 Desktop/capture 파일은 수정하지 않았습니다. 외부 API 호출 없음.
''',encoding='utf-8-sig')
    summary=dict(captures=len(data['captures']),proposed_boxes=len(rows),classes=len(data['classes']),
        approved_boxes=0,edge_boxes=sum(r['touches_edge'] for r in rows),original_bytes_verified=True,
        coordinate_space='original_capture_pixels',training_started=False)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    zip_path=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and path.name not in {'automatic_seed_proposals.json','review_syntax_check.js','server.stdout.log','server.stderr.log'}:
                z.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert sum('/images/' in n and n.endswith('.png') for n in z.namelist())==110
    print(json.dumps({**summary,'zip_mib':round(zip_path.stat().st_size/1024**2,1)}))

if __name__=='__main__':main()
