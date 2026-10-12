# 산업 제어회로 데이터 재선정 기록 — 2026-10-03

목표는 Y–Δ 전동기 기동 회로와 같은 접촉기·NO/NC 접점·타이머·과부하 계전기·단자·배선이 있는 완성된 회로도입니다. 건축 평면도, 손그림 저항 회로, 기호만 나열한 이미지를 목표 데이터로 간주하지 않습니다.

| 후보 | 확인한 내용 | 이번 판단 |
|---|---|---|
| [AITEE](https://github.com/CKnievel/aitee-dataset) | 앞선 교육용 전자회로 데이터 | 목적 불일치. 학습/UI 기본값 제외 |
| [QElectroTech 공식 예제](https://github.com/qelectrotech/qelectrotech-source-mirror/tree/master/examples) | 산업 제어·비상정지·PLC·단자대가 포함된 실제 CAD 예제 문서 | 내려받은 현재 및 0.100 배포의 ELEMENTS.LICENSE에 ML 학습 불허 문구. 학습 제외 |
| [Industrial Diagram Benchmark](https://github.com/codesknight/Industrial-Diagram-Benchmark) | 산업 배선·제어 관련 도면. 안정 공개 단위는 수작업 검토한 14개 topology panel | 목표에 가까운 조사 후보. 즉시 YOLO 학습 가능한 기호 bbox 데이터라는 근거는 확인되지 않음. 원본 권리·정답 의미·분할 추가 검토 필요 |
| [circuit-vision](https://github.com/zakaneki/circuit-vision) | 변압기·차단기·스위치·MV선 4종 OBB. README 학습 데이터는 합성 200/50장 | 완성 산업 제어회로 실데이터의 대체재로 선택하지 않음 |
| [covaga electrical-symbols-dataset](https://huggingface.co/datasets/covaga/electrical-symbols-dataset) | EPLAN 계열 단일 기호 이미지 모음 | 완성 도면·연결 정답이 있는 데이터가 아님. 권리와 보조 사전학습 적합성 별도 검토 |

웹에 공개된 제조사 회로 PDF는 곧바로 라벨 학습 데이터가 되는 것이 아닙니다. 원본 사용 조건과 함께 기호 위치·클래스·접속/비접속·핀 대응 정답이 필요합니다.

## 준비할 라벨

1. 클래스 예: contactor_coil, relay_coil, timer_coil, contact_no, contact_nc, pushbutton_no, pushbutton_nc, overload_relay, breaker, fuse, motor, terminal, ground, splice. 실제 현장 표기에 맞춰 확정합니다.
2. 검출: detect `class cx cy w h`, OBB `class x1 y1 ... x4 y4` (모두 0~1 정규화). OBB 박스 방향은 핀 방향과 같지 않습니다.
3. OCR: 문자열과 polygon/bbox, 별도 모델 학습·평가. 검출 YOLO 라벨은 OCR 라벨을 대체하지 않습니다.
4. 연결: node/pin/wire 그래프, 교차 비접속과 접속 점을 구분합니다.
5. 개정 비교: 프로젝트/페이지/개정 계보 및 객체 대응. 같은 프로젝트의 개정본은 서로 다른 split에 넣지 않습니다.

이번 세션에서 산업 제어회로 모델 학습 완료 또는 정확도 수치를 보고하지 않습니다. 파일 확보·도메인 적합성·권리·라벨 검증을 통과한 데이터가 아직 확정되지 않았기 때문입니다.

추가 확인: Industrial Diagram Benchmark의 Hugging Face `main` 파일 API를 직접 확인했을 때 39개 항목 중 파일은 README/manifest/report 등 메타데이터였고 PNG/DWG/DXF 원본은 반환된 목록에 없었습니다. manifest의 `image_exists=true`는 저자의 환경에 대한 기록이며, 현재 공개 다운로드가 가능하다는 검증으로 사용하지 않습니다. 데이터 카드에는 Apache-2.0이 명시되어 있습니다. 따라서 이 자료도 지금 바로 내려받아 학습한 데이터라고 보고하지 않습니다.
