# 첨부 자료 적용 및 실측 결과 — 2026-10-06

사용자가 제공한 ZIP을 `references/`에 보존하고 선택한 코드만 검토해 적용했습니다. 외부 이미지 API, Ollama, 참조 저장소의 CLI는 실행하지 않았습니다.

| 자료 | 실제 적용 | 적용하지 않은 부분 |
|---|---|---|
| Image2CAD (Aditya Intwala) | 원 후보를 영상 잉크와 대조하고 문자/기하를 나누는 접근을 참고. 포함된 기계도면 4개는 회귀 비교 입력으로 실행 | OpenCV 3 전용 전체 프로그램, Tesseract 하드코딩 경로. README의 90% 수치를 우리 성능으로 사용하지 않음 |
| PaddleOCR_Engineering_Drawings (Binesh Kumar) | OCR 결과와 구조화 단계를 분리하는 구조 검토. 원본 처리와 의미 해석의 로컬 경계를 유지 | PaddleOCR-VL + Qwen 전체 페이지 메타데이터 모델은 이번 실측에 실행하지 않음. 출력에 좌표가 없어 기호/배선 복원에 바로 대체할 수 없음 |
| drawing-to-dxf | `skeleton_graph`, `geometry_model`, `segment_types`, `vector_fit`를 `reference_vector/`로 분리해 내부 실험에 적용 | OpenAI 이미지 전송, 부품 패널 분할, 배선을 지울 수 있는 수평 치수선 제거, 긴 선만 유지하는 기본 필터 |

최초 비교는 EasyOCR로 수행했고, 2026-10-06에 기본 OCR을 로컬 `PP-OCRv5_server_det` + `korean_PP-OCRv5_mobile_rec`로 교체했습니다. 첨부 저장소의 **PaddleOCR-VL/Qwen 전체 페이지 메타데이터 모델은 실행하지 않았습니다.**

변경한 부분: 대각 픽셀의 불필요한 연결 방지, 원본 1픽셀 선 보존, 짧은 접점 보존, 전체 곡선의 잉크 지지를 확인한 원호 피팅, 영상 경계 검사, DXF LWPOLYLINE 지원, OCR TEXT의 상자 폭 맞춤. 사각 틀을 거대한 원호로 잘못 바꾸던 참조 피팅 결과를 영상 근거로 거부하도록 했습니다.

## Y–Δ 실제 입력 결과

입력은 사용자가 올린 640×512 회로도입니다. 모델 학습 입력으로 사용한 것이 아니라 변환·비교 입력으로 사용했습니다.

| 항목 | 기존 Hough | PaddleOCR + 중심선 |
|---|---:|---:|
| 직선 | 55 | 128 |
| 꺾인 선/곡선 폴리라인 | 0 | 113 |
| 원 후보 | 0 | 9 |
| 편집 가능한 OCR TEXT | 0 | 35 |
| 원본 선 흔적 재현율, 2px 허용 | 23.18% | 98.69% |
| 선 흔적 precision | 98.46% | 99.95% |
| 선 흔적 F1 | 37.53% | 99.32% |
| 벡터 추출 시간 | 0.008초 | 0.348초 |

PaddleOCR는 CPU에서 모델 로드 15.21초, 첫 인식 11.40초, 같은 모델을 재사용한 인식 11.18초였습니다. 이전 EasyOCR는 22개 TEXT/웜 인식 6.85초였고, PaddleOCR는 35개 TEXT를 남겨 속도는 느리지만 작은 기기명과 한글 누락이 줄었습니다. DXF 저장·렌더링·UI 시간은 제외한 단일 샘플 측정입니다.

**98.69%는 회로 정확도가 아닙니다.** OCR 영역을 두 방법 모두에서 같은 마스크로 제외한 다음, 원본 이진 잉크의 중심선과 저장된 DXF를 다시 읽어 그린 기하 픽셀을 비교했습니다. 문자 상자 내부의 누락은 이 수치에 드러나지 않으며, 부품 의미·접점 상태·전기적 연결·치수·OCR 정답률을 평가하지 않습니다. 참조 코드 원본 실행과의 우열 평가도 아닙니다. 기존 프로젝트 Hough 방식 대비 통합 파이프라인 비교입니다.

눈으로 확인한 실패 사례:

- 제목에서 `△`가 누락되어 `Y-기동시퀀스회로`로 인식됨.
- `PB off/on`, 접촉기 하위 문자는 `PB-oP`, `PBov`, `M-t`, `M2-n`으로 오독.
- 작은 접점 일부를 `g`, `C`, `0`으로 문자 오검출.
- OCR이 인식하지 못한 글자는 일부 선/폴리라인으로 남음.
- 접속점의 전기적 의미와 기호 클래스/BLOCK은 확정하지 않음.

## 다른 참고 입력

Image2CAD의 기계도면 4개도 실행했습니다. 동일 잉크 재현율은 각각 기존 26.7/46.2/28.1/52.4% → 개선 99.6/100.0/100.0/99.9%였습니다. 작은 저해상도 4장에 대한 2px 허용 결과이며, 문자·제작 치수·기계 도면 이해 정확도를 의미하지 않습니다. 이 자료를 전기 회로 학습 데이터로 바꾸어 보고하지 않습니다.

## 보기와 재현

- Streamlit 첫 화면: 원본 / 기존 DXF / 개선 DXF, 실측 지표, 인식 문자열, DXF 다운로드.
- 파일: `output/reference_comparison/y_delta_paddle/comparison.png`, `improved.dxf`, `report.json`.
- 모든 비교의 원본 이미지는 미리보기 용도로만 별도 저장합니다. DXF에는 래스터 IMAGE를 넣지 않습니다.
- 시험 축척 0.1 mm/px는 **미보정**입니다. 실제 제작 크기로 사용하려면 기준 치수 보정이 필요합니다.

```powershell
.\.venv-ml\Scripts\python.exe -X utf8 benchmark_preview.py data/evaluation/user_y_delta.png --output output/reference_comparison/y_delta_paddle --ocr-engine paddle
.\.venv-ml\Scripts\python.exe -X utf8 -m unittest -v test_vector_preview.py test_pipeline.py test_app.py
.\start_streamlit.ps1
```

검사 13개 통과. 폴리라인의 Y축 반전, 실제 DXF 렌더 크기, 얇은 선 보존, 지원하지 않는 원호 거부, 경계 검사, 기존 개인정보 전송 경계를 포함합니다. 1회 공개 모델 다운로드 이후 PaddleOCR 추론은 네트워크 차단 하에서 실행했습니다. API 호출은 0회입니다.

OCR 추가 설치 때 Conda에서 상속한 SciPy의 비대각 역행렬 연산에서도 네이티브 오류가 발견되어, 프로젝트 안에 독립 SciPy 1.17.1 wheel을 설치했습니다. 사전 검사를 비대각 행렬로 강화했습니다. 사용자의 기존 Conda 환경은 바꾸지 않았습니다.

## 출처와 내부 사용 조건

- Image2CAD: Copyright © 2016 Aditya Intwala, 첨부 Apache-2.0 LICENSE 보존. 논문: *Image to CAD: Feature Extraction and Translation of Raster Image of CAD Drawing to DXF CAD Format*, CVIP 2019.
- PaddleOCR_Engineering_Drawings: 첨부 스크립트/README의 MIT 표기, 저자 Binesh Kumar. 제공 PDF는 별도 Medtronic 라이선스 자료로, 이번 학습·비교에 사용하지 않았습니다.
- drawing-to-dxf: 첨부 README가 interview/internal work 사용·수정을 허용하며 재배포에는 라이선스 파일 추가를 언급합니다. 별도 표준 LICENSE 파일은 없어 이 통합본을 공개 배포 완료한 것으로 간주하지 않습니다.
- EasyOCR 1.7.2: [공식 저장소](https://github.com/JaidedAI/EasyOCR), Apache-2.0. CRAFT 검출 모델 및 Korean G2 인식 모델을 사용했습니다.

ZIP SHA-256:

```
Image2CAD-master.zip                       3c73d27c5e37792e546c65fd9ea1916c0b76f78726c2dd4d79d69ad6bebf0768
PaddleOCR_Engineering_Drawings-master.zip   426b5f694d2bff19839024a2e9d65d239a171a5e344118706ae36e98c9d4cf6f
drawing-to-dxf-main.zip                    d2f016f5656879be2df3fe80007a6deb9046c0a649a11772ac23838ec24a90d0
```
