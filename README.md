# 전기 제어회로·하네스 도면 → 로컬 CAD

과제 1 PDF→블록 DWG와 과제 2 벡터 PDF 변경 비교를 합친 웹은 `start_cad_workbench.ps1`로 실행합니다. 브라우저 주소는 `http://127.0.0.1:8502`입니다. 사용법과 범위는 [CAD_WORKBENCH.md](CAD_WORKBENCH.md)에 정리했습니다.

VS Code에서 `harness_workflow.ipynb`를 열고 커널을 `.venv-ml\Scripts\python.exe`로 선택하세요.
현재 상태는 **검토 가능한 파이프라인 구현**입니다. 회사 도면의 완전 자동 복원 모델을 학습 완료한 상태가 아닙니다.

## 실행

프로젝트 폴더의 PowerShell에서:

```powershell
.\start_streamlit.ps1
```

브라우저: http://127.0.0.1:8501

상단의 `도면 두 개 자동 변경 비교`에서 전도면과 후도면 PDF/이미지를 함께 올릴 수 있습니다. 앱은 두 페이지를 ORB+RANSAC으로 정렬하고, 빨간 변경 주석이 있으면 로컬 PaddleOCR로 문구를 읽어 변경 목록과 동일 좌표의 전후 확대 이미지를 보여 줍니다. 빨간 주석이 없는 문서는 2픽셀 허용 선 흔적 차이를 후보로 표시합니다. 저장된 결과 선택에서 `블라인드 샘플 · 빨간 정답 없음`과 `표시본 샘플 · 빨간 주석 OCR`을 바로 비교할 수 있습니다.

빨간 표시가 없는 제공 샘플을 100 DPI로 블라인드 비교한 결과 정렬 특징점 9,271개, RANSAC 내부점 비율 99.02%, 변경 선 흔적 1.82%를 얻었습니다. 113개 군집 중 큰 후보 30개가 검출된 변경 픽셀의 99.45%를 포함했습니다. 이 비율은 정답 대비 재현율이 아니라 내부 검출 픽셀의 분포이며, 부품 의미와 전기 연결 변경의 정확도를 뜻하지 않습니다.

같은 샘플의 OCR 점검은 150 DPI에서 긴 배선을 제거하고 가로 문자열 후보만 Korean PP-OCRv5 인식기에 넣습니다. 전도면 306개 후보 중 242개, 후도면 314개 후보 중 248개를 신뢰도 0.2 이상으로 읽었습니다. 신뢰도 0.8 이상에서는 전·후도면 동일 위치 107건의 정규화 문자열이 91.6% 일치했습니다. 이는 정답 정확도가 아니라 대부분 그대로인 영역의 반복 인식 안정성입니다. 단일 문자·세로 문자·필터가 제안하지 못한 문자의 재현율은 포함하지 않습니다.

1. API 키 없이 `합성 예제 불러오기`로 의미 JSON·DXF 흐름을 확인할 수 있습니다.
2. 첫 화면의 실제 도면 비교에서 Y–Δ 회로 원본·기존 Hough·개선 결과와 DXF를 확인합니다. 이미지/PDF 업로드도 지원합니다.
3. 개선 모드의 기본 OCR은 로컬 PaddleOCR PP-OCRv5입니다. EasyOCR는 비교용으로 남겨 두었습니다. 기호 의미 검출에는 별도로 학습한 YOLO `.pt`가 필요합니다.
4. API 키 입력 후 `합성 JSON으로 API 연결 확인`을 누르면 실제 Responses API를 호출합니다. 실제 도면은 이 버튼의 입력이 아닙니다. 유효한 키·계정 사용 한도·모델 접근 권한·인터넷 연결이 필요합니다.
5. 도면 의미 보조 API는 표시된 범주 JSON에 대해 전송 허용을 선택하고 요청해야 호출됩니다.
6. 의미와 축척을 검토한 뒤 DXF를 다운로드합니다. DWG는 설치된 AutoCAD/ZWCAD와 `pywin32`가 있어야 생성됩니다.

API 실호출은 키를 제공받지 않아 미검증입니다. HTTP 요청 본문과 응답 검증은 모의 API로 테스트했습니다. DWG COM은 실제 CAD 설치로 검증하지 않았습니다.

## 첨부 저장소 반영 — 2026-10-06

실제 첨부 Y–Δ 회로의 PaddleOCR 결과는 128 LINE, 113 LWPOLYLINE, 9 CIRCLE, 35 TEXT입니다. 같은 2픽셀 허용 기준의 선 흔적 재현율은 23.2%에서 98.7%로 개선됐습니다. 이 수치는 OCR 영역을 제외한 시각적 재현율이며 회로 의미·접속 정확도가 아닙니다.

결과: `output/reference_comparison/y_delta_paddle/comparison.png`, `improved.dxf`. 측정 조건과 EasyOCR 비교는 [REFERENCE_INTEGRATION.md](REFERENCE_INTEGRATION.md)에 정리했습니다. 학습 완료를 의미하지 않습니다.

## Python 오류 수정 — 2026-10-06

기존 `.venv-ml`이 Conda `TORCH_PY311`의 NumPy 2.4.6을 상속했고, NumPy 단독 행렬 곱셈에서도 Windows 네이티브 예외 `0xc06d007f`로 종료됐습니다. 학습 로그의 실패 지점은 Ultralytics의 좌표 변환 행렬 곱셈이었습니다. 특정 DLL 이름까지는 규명하지 않았습니다.

프로젝트 `.venv-ml` 안에 PyPI NumPy 2.4.4 wheel을 별도로 설치했습니다. 이후 OCR 설치 과정에서 상속한 SciPy의 비대각 역행렬에서도 네이티브 오류를 발견해 SciPy 1.17.1 독립 wheel을 설치했습니다. 사전 검사를 실제 BLAS 연산이 필요한 비대각 행렬로 강화했습니다. 사용자 Conda 환경은 수정하지 않았습니다.

```powershell
.\.venv-ml\Scripts\python.exe -X utf8 check_runtime.py
```

실행 스크립트와 학습 스크립트는 이 자식 프로세스 검사가 실패하면 중단합니다. 검사 프로세스의 Windows 오류 대화상자는 비활성화하지만, 종료 코드와 faulthandler 오류 로그는 유지합니다. 이것은 모든 종류의 네이티브 충돌을 해결한다는 보장이 아닙니다.

동일 환경을 다시 만들 때(이미 설치된 현재 환경에서는 불필요):

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-preview.txt
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-paddle.txt
# PaddleOCR 설치 뒤 프로젝트에서 검증한 수치 연산·OpenCV wheel을 마지막에 고정
.\.venv-ml\Scripts\python.exe -m pip install --ignore-installed --no-deps numpy==2.4.4 scipy==1.17.1 opencv-python-headless==5.0.0.93
# OCR 모델 최초 다운로드(도면을 입력받거나 전송하지 않음)
.\.venv-ml\Scripts\python.exe setup_preview_models.py
.\.venv-ml\Scripts\python.exe setup_paddle_models.py
```

새 PC는 Python 3.11 독립 가상환경과 그 PC의 드라이버에 맞는 PyTorch를 먼저 설치하세요. 기본 `.venv`는 GPU 학습 환경이 아닙니다.

## LLM 적극 활용 변경 해석

변경 비교의 새 기본 설계는 LLM에 후보별 실제 OCR 문자열·수치·부품번호·좌표·픽셀 변화·주변 객체·연결 근거를 충분히 전달합니다. 원본 PDF와 원본/crop 이미지는 계속 로컬에 둡니다. LLM은 모든 후보를 `confirmed`, `probable`, `uncertain`, `rendering_noise`로 판정하고 전후 값과 변경 종류를 구조화해서 반환합니다. 선택한 경우 같은 근거와 1차 결과를 다시 보내 독립 재검토합니다.

현재 블라인드 샘플은 30개 후보, 전후 OCR 연결 179개, 약 67KB의 풍부한 JSON을 만듭니다. Streamlit에서 JSON 전체를 확인하고 실제 OCR·좌표·객체·연결 근거의 전송을 명시적으로 허용한 뒤 호출합니다. 응답은 로컬 스키마로 검증하며 OCR이 읽지 못한 값은 발명하지 못하도록 했습니다.

기존 `privacy.py`의 최소 범주 관계 문의는 레거시 기능으로 유지합니다. 새 변경 분석은 `llm_revision.py`를 사용합니다. 실제 OCR·부품·좌표도 회사의 파생 정보이므로 외부 전송 허용 여부를 확인해야 합니다. 상세 흐름은 [ARCHITECTURE_LLM.md](ARCHITECTURE_LLM.md)에 정리했습니다.

## 데이터와 학습 상태

앞서 받은 AITEE 교육용 회로는 사용자가 요청한 Y–Δ 기동·접촉기·타이머 제어회로와 맞지 않아 기본값에서 제외했습니다. 실패한 학습 로그를 성공 결과로 취급하지 않습니다. **현재 검증된 산업 제어회로 학습 가중치는 없습니다.**

QElectroTech 산업 회로는 자료 유형은 맞지만 내려받은 `ELEMENTS.LICENSE`에 머신러닝 학습 사용 불허 문구가 있어 학습에 사용하지 않았습니다. 다른 후보와 제외 사유는 `research_datasets.md`에 기록했습니다. 적합한 라벨 데이터 확보와 재학습은 남아 있습니다.

`prepare_public_data.py`는 이전 AITEE 자료 조사 코드이며 현재 실행 경로가 아닙니다. `train_public.py`는 `--data`를 명시해야 실행됩니다.

```powershell
.\.venv-ml\Scripts\python.exe -X utf8 train_public.py --data data\my_control_circuits\dataset.yaml --epochs 100 --weights models\yolo11n.pt
```

OBB 학습은 노트북의 `train_yolo(..., task="obb")`를 사용합니다. 같은 도면의 페이지·타일·개정판·증강본이 train/val/test에 섞이지 않게 원본 프로젝트별로 분리하세요. 기호 검출 정확도와 배선 연결·CAD 복원 정확도는 따로 평가합니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `harness_workflow.ipynb` | 단계별 실행·설명·학습·API·CAD·개정 비교 |
| `app.py` | 로컬 Streamlit 입력·검토·다운로드 화면 |
| `vision.py` | PDF/이미지 전처리, 직선 후보, 로컬 오버레이 |
| `vectorize_local.py`, `reference_vector/` | 중심선 추적, 원·원호 검증, 폴리라인 보존 |
| `paddle_ocr_local.py` | 기본 한글·영문 PP-OCRv5 로컬 추론 |
| `ocr_local.py` | 비교용 EasyOCR 로컬 추론 |
| `benchmark_preview.py` | 실제 DXF 재렌더링, 원본 비교와 제한적인 픽셀 평가 |
| `models_local.py` | 로컬 YOLO/PaddleOCR, 학습 라벨 검증, 네트워크 차단 |
| `pipeline.py`, `rules.json` | 단위 규칙, 연결 후보, 검토한 최종 JSON |
| `privacy.py` | 레거시 최소 범주 관계 문의와 응답 검증 |
| `llm_revision.py` | 실제 OCR·좌표·객체·연결 근거를 포함한 LLM 변경 판정과 2차 재검토 |
| `topology.py` | 미확정 연결 그래프와 명시적 대응 기반 개정 비교 |
| `cad.py` | DXF 객체·BLOCK·LAYER·속성, 선택적 DWG COM 변환 |

## 구현 범위

- 개선 모드는 얇은 선 보존, OCR 영역 제외, 중심선 추적, 원·원호 적합을 지원합니다. 적합하지 않은 곡선은 폴리라인으로 보존합니다. 최대 400만 픽셀이며 접속 의미는 확정하지 않습니다.
- 기본 OCR은 `PP-OCRv5_server_det` + `korean_PP-OCRv5_mobile_rec`입니다. Windows 네이티브 엔진의 한글 경로 문제를 피하기 위해 공개 모델은 `Documents/CAD_OCR_MODELS`에 저장합니다. 첨부 PaddleOCR-VL/Qwen 모델은 실행하지 않았습니다.
- `LINE/LWPOLYLINE/CIRCLE/ARC/SPLINE/TEXT/BLOCK` 변환과 Y축 반전·축척·레이어·XDATA 속성 저장을 구현했습니다. 기호 BLOCK은 별도의 실제 DXF 라이브러리와 클래스 매핑이 필요합니다.
- 도면 교차점은 전기적 접속으로 자동 확정하지 않습니다. 접점·핀·배선은 로컬 검토가 필요합니다.
- 개정 비교는 사람이 확인한 객체 대응을 사용합니다. 위치 변화는 정합 정보 없이 확정하지 않습니다. 배선 그래프 비교에는 노드 대응과 확인된 연결이 필요합니다.
- 이 구현은 개인 PC의 순차 사용 기준입니다. 로컬 모델의 네트워크 차단은 프로세스 전체에 적용되므로 다중 사용자 서비스는 별도 워커 격리가 필요합니다.

## 검증

```powershell
.\.venv-ml\Scripts\python.exe -X utf8 -m unittest -v test_pipeline.py test_app.py test_vector_preview.py
.\.venv-ml\Scripts\python.exe -X utf8 -c "import vision,topology,models_local; vision.self_check(); topology.self_check(); models_local.self_check()"
```

논문별 적용 범위와 공식 소스는 `research_detection.md`, `research_vectorization.md`, `research_connectivity.md`에서 확인하세요.
