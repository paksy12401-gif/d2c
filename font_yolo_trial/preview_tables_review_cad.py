"""Render the saved DWG roundtrip and explain the measured OCR progression."""
import html
import json
from pathlib import Path

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend, config, layout, pymupdf
from ezdxf.math import BoundingBox2d

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/cad_with_tables_ocr'


def main():
    manifest=json.loads((OUT/'conversion_report.json').read_text(encoding='utf-8'))
    assert manifest['dwg_roundtrip_text_verified']
    doc=ezdxf.readfile(OUT/'dwg_roundtrip.dxf')
    settings=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.COLOR,lineweight_scaling=.6)
    height=2383.93994140625
    previews=[
        ('overview.png',[0,0,12500.79,height],480,94),
        ('left_table.png',[5045,170,7140,1250],390,205),
        ('right_table.png',[7990,70,11170,2230],390,270),
        ('right_detail.png',[8000,90,11160,520],390,58),
    ]
    for name,(x0,y0,x1,y1),width,ph in previews:
        # Each export transforms the recorder; use a fresh one for each viewport.
        backend=pymupdf.PyMuPdfBackend()
        allowed=None
        if name=='left_table.png':allowed={'TABLE_LEFT_SOURCE','TABLE_LEFT_TEXT'}
        elif name.startswith('right_'):allowed={'TABLE_RIGHT_GRID','TABLE_RIGHT_TEXT'}
        Frontend(RenderContext(doc),backend,config=settings).draw_layout(
            doc.modelspace(),finalize=True,
            filter_func=(lambda entity:entity.dxf.layer in allowed) if allowed else None)
        box=BoundingBox2d([(x0/72,(height-y1)/72),(x1/72,(height-y0)/72)])
        (OUT/name).write_bytes(backend.get_pixmap_bytes(
            layout.Page(width,ph,margins=layout.Margins.all(2)),dpi=150,render_box=box))
        print('RENDERED',name,flush=True)
    progress=json.loads((OUT/'progress_metrics.json').read_text(encoding='utf-8'))
    metrics={(r['stage'],r['table']):r for r in progress['metrics']}
    stages=[
        ('raw_cell','① 원본 칸 OCR','표를 칸별로 잘라 OCR에 넣음. 얇은 획과 테두리·취소선, 작은 글자 때문에 혼동과 누락이 발생.'),
        ('clean_cell','② 입력 정리 추가','왼쪽은 PDF에 남은 글자 벡터 획만 검게 다시 그려 표선과 취소선을 OCR 입력에서 제외. 오른쪽은 이미지 글자 영역을 잘라 높이·여백을 맞춤.'),
        ('refined','③ 여러 OCR 결과 대조 추가','영문 OCR과 원본/굵은 획 입력, 일부 구간 재인식을 비교. 혼동 글자는 여러 결과가 지지할 때만 바꾸고 연속 숫자를 보호. 오른쪽 색상 코드 형식과 같은 도면 본문에서 YOLO로 읽은 커넥터 코드를 조건부 대조.'),
        ('font_verifier','④ 실제 표 글꼴 보조 학습 추가','학습용 5칸의 고유 글자 패치 23개로 O·0·I/l 공통 세로획·1을 구분하는 작은 CNN을 새로 학습. 기존 OCR 또는 YOLO 전체를 재학습한 것은 아님.'),
        ('guarded_server','⑤ 서버급 OCR의 조건부 보정 추가','서버급 모델은 로컬 실행. 두 입력의 OCR 문자가 일치하고 모양·높이 조건도 맞는 I/l 및 일부 대소문자만 보정. 서버 OCR 단독 결과로 전체를 교체하지 않음.'),
    ]
    rows=[]
    for key,title,detail in stages:
        values=[]
        for table in ['left','right']:
            m=metrics[key,table]
            values.append(f"{m['exact']}/{m['cells']} ({100*m['rate']:.1f}%)")
        rows.append(f'<tr><th>{title}</th><td>{detail}</td><td>{values[0]}</td><td>{values[1]}</td></tr>')
    report='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>표 포함 DWG와 성능 개선 과정</title>
<style>body{font:16px system-ui;line-height:1.75;color:#203040;max-width:1450px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd3dc;padding:12px;text-align:left;vertical-align:top}th{min-width:140px}td:nth-last-child(-n+2){min-width:130px}img{display:block;width:100%;border:1px solid #ddd;margin:15px 0}.note{padding:16px;background:#eef5fc}.warning{padding:16px;background:#fff1e8}a{color:#125ba7}code{background:#eef1f4;padding:2px 4px}</style>
<h1>표까지 OCR 문자를 넣은 DWG</h1>
<p><a href="drawing_with_tables_yolo_ocr.dwg">DWG 열기 / 다운로드</a> · <a href="drawing_with_tables_yolo_ocr.dxf">DXF</a> · <a href="conversion_report.json">저장·재열기 검증 기록</a> · <a href="../cad_table_font_training/report.html">글자별 OCR 검증 보고서</a></p>
<p class="note">AutoCAD 2018 형식. 기존 본문 문자 48개·테두리 표기 104개·PDF 원래 문자 59개를 유지하고, 왼쪽 160칸·오른쪽 720칸을 각각 편집 가능한 TEXT로 추가했습니다. 총 TEXT/MTEXT 1,091개입니다. 오른쪽 표선 84개도 LINE으로 재구성해 외부 이미지 파일 없이 DWG만으로 열 수 있습니다.</p>
<p>실제 AutoCAD로 DWG를 저장하고 다시 열어 DXF로 재내보냈습니다. 880칸의 문자열과 기존 211개 문자 내용, 레이어 상태, 칸 안 배치, CAD 구조 검사를 확인했습니다. 이는 저장·변환의 무결성 검사이며 모든 OCR 결과가 정답이라는 뜻은 아닙니다.</p>
<h2>처음 방식에서 무엇을 더했나</h2>
<p>본문은 처음에 도면 전체의 작은 글자를 YOLO/OCR로 찾는 데서 시작했습니다. 이후 글자 영역을 먼저 나누고, 벡터가 있는 곳은 PDF 획을 이용해 글자 입력을 분리하고 크기·획 두께를 조절했습니다. 영문·숫자·기호 후보와 한글 OCR을 영역·위치에 맞춰 합쳤으며, 특정 한글 모음의 혼동에는 OCR 후보와 원본 획 구조가 함께 맞을 때만 좁은 범위의 보정을 적용했습니다. 벡터가 없는 이미지 영역은 별도로 이미지 OCR을 사용합니다. PDF 전체에 원래 문자 좌표나 사람이 쓴 획 순서가 보존된다는 전제는 사용하지 않습니다.</p>
<p>문자가 읽혔어도 CAD에서 잘 안 보이던 문제는 별도로 해결했습니다. 영문 글꼴을 Arial로 바꾸고 배경에 맞춰 보이는 색상을 사용했으며, 인식된 문자에 대응하는 원본 획은 별도 꺼진 레이어에 보존했습니다. 이 조치는 표시 개선이며 OCR 정답률 향상과 구분해야 합니다.</p>
<p><strong>아래는 표의 단계별 저장 결과를 동일한 왼쪽 98칸·오른쪽 210칸으로 다시 비교한 수치입니다.</strong> 학습용 5칸과 동일 문구 1칸은 제외했습니다. 칸 전체 문자열의 대소문자·숫자·기호가 모두 같아야 정답이며, 공백 차이만 제외했습니다. 이전 74칸 결과에 추가 검사 24칸을 합친 것이므로 예전 보고서와 분모가 다릅니다.</p>
<table><tr><th>단계</th><th>추가한 처리와 이유</th><th>왼쪽 98칸</th><th>오른쪽 210칸</th></tr>ROWS</table>
<p>⑤의 ‘문맥’은 문장 뜻으로 답을 추측하는 LLM 보정이 아닙니다. 칸 전체를 읽은 OCR 결과를 개별 획 모양·글자 높이와 대조한 제한적 보정입니다. 학습한 보조 모델은 I와 l이 같은 세로획 모양일 때 둘을 이미지 한 글자만으로 구별하지 않습니다. 서버급 OCR 단독은 기존 74칸 왼쪽 검사에서 45~46칸만 맞아 전체 교체를 하지 않았습니다.</p>
<h2>남아 있는 오류와 검증 범위</h2>
<p class="warning">확인한 308칸 중 한 칸은 <code>TYCO</code>를 <code>TY0O</code>로 읽습니다. 왼쪽 표 PC NO 21의 DB PART NAME 칸입니다. 숫자 <code>12296694</code>는 맞습니다. 실제 인식 성능을 숨기지 않도록 이 오류를 손으로 수정하지 않았습니다. 같은 PDF·같은 글꼴에서의 검사이며, 전체 880칸 또는 다른 도면의 정확도가 99~100%라는 뜻은 아닙니다.</p>
<p>본문은 이전 감사에서 선택한 48문구의 문자 오류 0건으로 기록됐지만, 5문구의 띄어쓰기 문제가 남아 있습니다. 이번 작업에서 본문 인식 문자열은 바꾸지 않았습니다. 사용자가 제외한 동그라미 안 숫자는 원본 선 상태로 유지했습니다. 실제 mm 치수 보정은 하지 않았습니다.</p>
<h2>레이어와 원본 비교</h2>
<ul><li><code>TABLE_LEFT_TEXT</code>, <code>TABLE_RIGHT_TEXT</code>: 새 편집 가능한 표 문자. 어두운 배경에서 흰색, 밝은 배경에서 검은색.</li><li><code>TABLE_LEFT_SOURCE</code>: 원본 왼쪽 표선·제목·빨간 취소선. 제목은 원래 선 모양을 유지했습니다.</li><li><code>TABLE_RIGHT_GRID</code>: 새로 만든 오른쪽 표선.</li><li><code>TABLE_SOURCE_OUTLINES</code>: 왼쪽 칸 원본 문자 획, 기본 꺼짐. 새 문자 레이어를 끄고 이 레이어를 켜서 원본과 비교할 수 있습니다.</li></ul>
<h2>DWG 재열기 후 미리보기</h2><p>아래 이미지는 원본 PDF가 아니라 저장한 DWG를 다시 내보낸 DXF의 렌더링입니다.</p>
<h3>전체 도면</h3><img src="overview.png"><h3>왼쪽 표</h3><img src="left_table.png"><h3>오른쪽 표</h3><img src="right_table.png"><h3>오른쪽 표 확대</h3><img src="right_detail.png">
</html>'''.replace('ROWS',''.join(rows))
    (OUT/'report.html').write_text(report,encoding='utf-8')
    print('DELIVERY_REPORT_READY',flush=True)


if __name__=='__main__':
    main()
