# 전기 배선·와이어 하네스 도면 벡터화 조사

조사일: 2026-10-02. 공개 논문·공식 저장소·공식 문서만 조사했다. 회사 도면은 열거나 업로드하지 않았고, 논문 코드를 설치·실행하거나 성능을 재현하지 않았다.

## 결론

이 프로젝트는 **벡터 PDF 직접 추출 → 로컬 중심선 추적 기준선 → 학습 벡터화 비교 실험** 순서가 적절하다. 배선에서 중요한 것은 두꺼운 선의 외곽을 예쁘게 복제하는 것보다 **하나의 전선 중심선과 실제 접속 관계를 보존하는 것**이다. 아래 우선순위는 확인한 출력 형식과 사용자 목표를 바탕으로 한 설계 판단이다.

`Deep Vectorization of Technical Drawings`는 로컬 기하 추출 후보가 될 수 있다. 그러나 이 연구의 출력만으로 전선의 전기적 연결, 커넥터 핀, 치수 의미 또는 DWG까지 완성되지는 않는다. 특히 입력에서 문자를 제거한다고 가정하며, 출력 primitive는 직선과 2차 Bézier 곡선이다. [논문 §3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf)

## 1. 정확한 논문과 구현

- **제목:** Deep Vectorization of Technical Drawings
- **저자:** Vage Egiazarian, Oleg Voynov, Alexey Artemov, Denis Volkhonskiy, Aleksandr Safin, Maria Taktasheva, Denis Zorin, Evgeny Burnaev
- **발표:** ECCV 2020, pp. 582–598. [저자 논문 페이지](https://artonson.github.io/publications/2020-deep-vectorization/)
- **1차 자료:** [ECVA 본문 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf), [보충자료 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579-supp.pdf), [arXiv 2003.05471](https://arxiv.org/abs/2003.05471), [공식 PyTorch 저장소](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings).

### 방법과 실제 출력

| 단계 | 확인한 내용 | 이 프로젝트에서의 의미 |
|---|---|---|
| Cleaning | U-Net으로 배경·노이즈 제거 및 끊긴 선 보충 | 보충된 선이 실제 전기 연결인지는 별도 검증 필요 |
| 초기 primitive 추정 | 64×64 patch를 ResNet 특징으로 인코딩하고 Transformer가 좌표·선폭·존재 confidence 예측 | OCR·기호 의미 해석과 분리할 수 있는 로컬 기하 모듈 |
| Refinement | raster와 primitive 사이의 인력 및 중복 억제 에너지를 Adam으로 최적화 | 초기 신경망 좌표를 그대로 CAD 좌표로 사용하지 않음 |
| Merging | patch 사이 직선·곡선을 휴리스틱으로 결합 | 분기점/미접속 교차부를 지우지 않는지 확인 필요 |

방법 근거: [논문 §3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf), [보충자료 A–C](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579-supp.pdf). 오른쪽 열은 하네스 도면 적용에 대한 설계 판단이다.

공식 `run_pipeline.py`는 `primitive_type`을 `line` 또는 `curve`로 선택한다. 저장 코드에서 curve는 `PT_QBEZIER`이며 최종 파일은 SVG이다. 별도의 의미 객체, DWG, `CIRCLE`, 원형 `ARC`를 직접 출력한다고 해석하면 안 된다. [실행 코드](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/run_pipeline.py), [곡선 병합·SVG 저장 코드](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/merging/merging_for_curves.py)

이 프로젝트에 연결한다면 직선은 `LINE`, Bézier는 제어점·차수를 유지한 `SPLINE`으로 매핑한다. Bézier **제어점**을 `cad.py`의 현재 `spline.points`인 **보간점**에 그대로 넣으면 다른 곡선이 된다. 모델 연동 때 별도의 제어점 표현을 추가해야 한다. 원/원호는 별도의 원형 피팅과 오차 검증을 통과했을 때만 해당 CAD 객체로 바꾼다. 이것은 구현 제안이며 논문에서 완료한 기능이 아니다.

### 성능 주장: 확인된 범위

논문 Table 1의 IoU 두 값은 평균 선폭/예측 선폭 조건이다. 아래 값은 **저자 보고값**이다.

| 데이터 | 평가 표본 | 전체 파이프라인 IoU | 해석 |
|---|---:|---:|---|
| PFP 평면도 | 40장 | 86% / 88% | 전기·하네스 도면 평가가 아님 |
| ABC 기계 부품 투영도 | 50장 | 77% / 77% | 비교법 PVF의 IoU 89%보다 낮지만 primitive 수는 적음 |
| DLD 열화 평면도 | 15장 | 79% / 82% | 수작업 정리한 raster 정답과 비교 |

ABC ablation에서 `NN → +refinement → +merging`의 IoU는 **65 → 91 → 77%**, primitive 수는 **309 → 240 → 97**이었다. 따라서 병합이 항상 정확도를 개선한다는 주장은 성립하지 않는다. [본문 §4, Tables 1·3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf)

보충자료의 단일 Globe 예시는 병합 전 IoU 91%에 5.5분, 수렴 92%에 12분, 병합 포함 최종 76%에 26분을 보고한다. 서로 다른 실행 환경이고 최적화된 구현이 아니라고 저자가 밝힌다. 이 수치를 현재 GPU 성능이나 페이지당 보장 처리시간으로 쓰면 안 된다. [보충자료 D, Table 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579-supp.pdf)

공식 QA에는 ABC 결과의 일부 수직선이 점선처럼 끊어지는 현상과 원인이 확정되지 않았다는 설명이 남아 있다. [공식 QA](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/QA.md)

### 코드·가중치·데이터·로컬 실행 조건

| 항목 | 확인 결과 |
|---|---|
| 코드 | 공식 repo 공개, 라이선스 파일은 **MPL-2.0**. 적용 파일의 수정·배포 조건은 해당 원문 기준으로 검토해야 함. [LICENSE](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/LICENSE) |
| 가중치 | README에 line/curve pretrained 다운로드 링크가 있음. line 링크는 `model_lines.weights` 페이지까지 확인. curve 호스트는 조사 도구에서 열기 실패했으므로 현재 다운로드 성공·무결성은 미검증. [README Models](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings#models), [line weight 페이지](https://disk.yandex.ru/d/FKJuMvNJuy-K9g) |
| 가중치 사용 조건 | 외부 가중치의 별도 명시적 라이선스를 이번에 확인하지 못함. repo 코드의 MPL 표기를 모든 가중치·데이터 권리의 보증으로 간주하지 않음 |
| Cleaning | README는 제공 실행 흐름에 cleaning model이 포함되지 않는다고 명시. 논문 그림의 전체 자동화가 바로 실행되는 것으로 취급하지 않음. [README How to run](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings#how-to-run) |
| 실행 환경 | README는 Linux/Python 3. requirements는 `torch==1.7.1`, `torchvision==0.8.2`, `numpy==1.20.0` 등 구형 패키지를 고정. 현재 프로젝트의 Windows/Python 3.14 환경과 바로 혼합하지 않고 별도 Linux 환경에서 재현하는 편이 현실적임. [requirements](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/requirements.txt) |
| CPU 지원 | CLI 도움말은 CPU를 언급하지만 실제 `vector_estimation`에서 `.cuda()`가 하드코딩되어 있음. CPU 실행을 검증된 기능으로 소개하면 안 됨. [run_pipeline.py](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/run_pipeline.py) |
| 학습 비용 | 저자 설정에서 모델당 NVIDIA V100 한 장으로 약 4일. 이는 재학습 실험의 보고값이지 추론 최소 장비 사양이 아님. [보충자료 B](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579-supp.pdf) |
| 데이터 분할 | repo는 ABC SVG train/test/validation과 PFP PDF, cleaning 데이터 다운로드 흐름을 제공. `golden_set`은 cleaning 데이터의 부분집합이므로 학습·검증에서 제외하도록 명시. [dataset README](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/dataset/README.md) |
| 데이터 권리 | ABC 공식 사이트는 CAD 모델 저작권이 제작자에게 있고 Onshape 약관을 따르도록 안내. PFP·배경·DLD와 가중치의 기업 이용 조건을 이 안내 하나로 확정할 수 없음. [ABC 공식 License](https://deep-geometry.github.io/abc-dataset/) |

**회사 도면을 API에 보내야 하는 구조는 아니다.** 확인한 inference 진입점은 로컬 이미지·checkpoint를 읽고 로컬 최적화를 수행한다. 다만 의존성 목록에는 `wandb` 등도 있으므로 일부 코드 읽기만으로 전체 배포물의 무통신을 증명할 수는 없다. 도입 시 모델·패키지를 미리 확보하고, 처리 프로세스의 외부 네트워크를 차단한 상태로 합성 도면을 실행해 검증해야 한다. [실행 코드](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/run_pipeline.py), [의존성 목록](https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings/blob/master/requirements.txt)

## 2. 실용적인 대안 3개와 학습 계열의 위치

| 후보 | 실제 지원·장점 | 하네스 적용 판단 |
|---|---|---|
| **벡터 PDF 직접 추출 — PyMuPDF** | `Page.get_drawings()`가 기존 draw command를 경로 사전으로 반환. 선·곡선·선폭 등을 raster 재인식 없이 활용. [공식 recipe](https://pymupdf.readthedocs.io/en/latest/recipes-drawing-and-graphics.html) | 원본 PDF에 벡터가 있으면 최우선. 스캔 bitmap에는 새 벡터가 생기지 않는다. clipping/hidden path 처리와 문자 outline을 구별하고 원본 render와 대조해야 한다. AGPL/상용 이중 라이선스 확인 필요. [라이선스](https://pymupdf.io/licensing) |
| **OpenCV 중심선 추적 + LSD/Hough + 피팅** | LSD는 선분 좌표, `HoughLinesP`는 선분, `HoughCircles`는 원 검출 기능 제공. `ximgproc.thinning`은 이진 blob을 골격으로 축소. [선·원 검출](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html), [thinning](https://docs.opencv.org/4.x/df/d2d/group__ximgproc.html) | 로컬 기준선으로 권장. thinning 이후 polyline 추적·분기 판정·선/곡선 피팅은 우리가 구현·검증할 부분이다. LSD나 Canny 결과만 쓰면 두꺼운 전선 양쪽 경계가 중복 선으로 잡힐 수 있으므로 중심선 품질을 별도 확인 |
| **Potrace** | 이진 bitmap을 SVG/PDF/DXF 등으로 변환. 로컬 Windows 실행 파일 제공, GPL 및 상용 라이선스 경로 존재. [공식 사이트](https://potrace.sourceforge.net/) | 로고·채움 기호 외곽선에는 유용. 공식 FAQ가 중심선 추적 미지원이라고 명시하므로 전선 네트워크의 주 추출기로는 부적합. DXF 출력 지원이 전선 중심선·전기 의미 지원을 뜻하지 않음. [공식 FAQ](https://potrace.sourceforge.net/faq.html) |

학습 계열에서는 우선 위 ECCV 2020 모델을 OpenCV 기준선과 비교하는 것으로 충분하다. 추가 연구용으로 `diffvg`는 로컬 미분 가능한 rasterizer와 SVG refinement를 제공하며 Apache-2.0이다. 그러나 준비된 SVG와 raster target을 받는 최적화 기반 도구이고, 도면에서 커넥터·핀·전선 그래프를 바로 추출하는 pretrained CAD 인식기는 아니다. 학습에 연결 가능한 부품과 완성된 인식 모델을 구별해야 한다. [diffvg 공식 저장소](https://github.com/BachiLi/diffvg)

## 3. 와이어 하네스와 revision 비교에 필요한 추가 계층

아래는 논문의 달성 결과가 아니라, 사용자 목적에 맞춘 구현 제안이다.

1. **OCR와 기하 입력을 분리한다.** 원본을 로컬 보존하고 OCR 문자열·bbox를 먼저 확보한다. 벡터화에 사용하는 문자 마스크가 실제 전선을 지우지 않았는지 표시한다. 검출한 기호 영역도 별도 마스크와 후보 연결점을 보존한다.
2. **중심선 그래프를 만든다.** 노드는 커넥터 핀, 단자, 접지, 스플라이스, 끝점이며 edge는 전선/하네스 경로이다. 단순 X자 교차만으로 접속을 확정하지 않는다. junction dot, jump/bridge, 기호 규약, 핀 접촉을 함께 검사하고 모호한 교차는 미확정 상태로 남긴다.
3. **기하와 의미를 따로 보존한다.** 원래 페이지 좌표, 정렬 좌표, 경로 polyline, 논리적 연결, 라벨, 기입 치수, 픽셀에서 환산한 길이를 별도 필드에 둔다. 비축척 도식에서 화면 경로 길이를 실물 전선 길이로 대체하지 않는다.
4. **revision 매칭은 ID가 새로 생성되어도 가능해야 한다.** 커넥터명+핀, 부품번호 후보, 이웃 연결, 분기명 등을 로컬에서 조합한다. 동일 이름의 중복·OCR 오독·이름 변경은 유일 매칭이 확정될 때까지 보류한다.
5. **변경을 분리해서 보고한다.** 기준점으로 페이지 이동/회전을 정렬한 후, `표기값 변경`, `배치 이동`, `경로 형상 변경`, `연결·분기 변경`, `분기명 변경`, `부품 추가·삭제·속성 변경`을 분리한다. 좌표만 달라졌다고 부품 삭제+추가 또는 접속 변경으로 기록하지 않는다.
6. **보안 경계는 그대로 둔다.** 회사 도면, 중심선 좌표 전체, 전체 그래프, 부품표를 외부 LLM에 넘기지 않는다. 필요한 경우 회사 정책상 허용된 최소한의 추상화된 후보 정보로만 의미 판단을 보조하고, 그래프 비교·CAD 생성·결과 검증은 로컬에서 수행한다. 이 범위까지 전송이 금지라면 규칙과 로컬 모델로 처리한다.

### 실제 비교 실험의 통과 기준

논문의 raster IoU 하나로 revision 비교를 승인하지 않는다. 합성 전기 도면부터 다음 사례를 검사하고, 이후 승인된 내부 샘플은 내부 환경에서만 평가한다.

| 합성 변경 | 기대하는 판정 |
|---|---|
| 도면 전체 이동·회전, 접속 유지 | 정렬 후 의미 변경 없음 |
| 동일 두 핀 사이 경로만 굽힘 | 경로 형상 변경, 접속 변경 없음 |
| 동일 위치에 junction dot 추가 | 연결/분기 변경 후보 |
| 평행 전선 두 개, 두꺼운 단일 전선 | 각각 edge 2개 / edge 1개 유지 |
| 치수 `150→170`, 배선 선은 동일 | 표기 치수 변경 |
| 분기명 변경, 부품 추가/삭제 | 해당 의미 변경으로 분류 |
| OCR 가림·스캔 단절 | 자동 연결/삭제 확정 대신 검토 필요 |

측정 항목은 중심선 거리, 잘못 생성한 중복 edge 수, 끝점·분기·접속 precision/recall, 허위 연결 수, 변경 종류별 precision/recall, 원본 대비 로컬 overlay, CAD 재열기 성공 여부이다. IoU는 시각적 보조 지표로 함께 사용한다.
