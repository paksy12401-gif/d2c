# 과제 1·2 통합 Streamlit 웹

## 실행

프로젝트 폴더의 PowerShell에서 실행한다.

```powershell
.\start_cad_workbench.ps1
```

브라우저에서 `http://127.0.0.1:8502`를 연다. 이 스크립트는 프로젝트의 `.venv-ml` Python을 사용한다. 새 환경에서 실행한다면 먼저 `requirements.txt`를 설치해야 한다.

## 과제 1 · PDF → 블록 DWG

같은 양식의 **1페이지 벡터 PDF**를 업로드한다. 앱은 심볼 검색, AutoCAD PDFIMPORT, 블록화를 실행하고 DWG·검출 표시 PDF·작업 기록을 ZIP으로 내려준다. 이전에 검증한 샘플 PDF와 파일 내용이 완전히 같으면 검증된 DWG를 제공한다. 새 PDF의 텍스트 OCR과 배선 분리는 아직 검증된 자동 복구가 아니다. 자세한 과정은 [PDF_TO_DWG_PROCESS.md](PDF_TO_DWG_PROCESS.md)에 있다.

새 PDF 변환에는 이 PC의 AutoCAD Core Console과 `dataset_640`, `labels_640` 폴더가 필요하다. 사이드바에서 심볼 데이터 폴더를 지정한다.

## 과제 2 · 도면 변경 비교

ZIP `Co_drawing2cad-main.zip`의 `drawing_compare` 비교 엔진을 통합했다. **기존/신규 벡터 PDF**를 각각 업로드하면 전역 위치를 맞추고 미세 이동을 제외한 뒤 벡터 차이를 찾는다. 화면에서 오버레이와 변경 영역을 볼 수 있다. 결과 ZIP에는 `report.html`, `result.json`, `overlay.png`, 변경 영역 이미지, 변경 위치가 표시된 페이지 이미지가 들어 있다. ZIP을 푼 상태에서 `report.html`을 열어야 이미지가 함께 보인다.

빨강은 기존 도면 전용 선, 초록은 신규 도면 전용 선, 회색은 공통 선이다. 스캔 이미지만 있는 PDF는 두 작업 모두 지원하지 않는다. 과제 2의 변경 설명은 기본적으로 로컬 형상 통계에 근거한다. `DRAWING_COMPARE_LLM_URL`, `DRAWING_COMPARE_LLM_API_KEY`, `DRAWING_COMPARE_LLM_MODEL`이 설정되어 있을 때 화면에서 **LLM 설명을 선택하면** 변경 영역 이미지와 수치가 해당 API로 전송된다. LLM은 변경 위치를 판정하지 않는다.

## 확인

```powershell
.\.venv-ml\Scripts\python.exe -m unittest -v test_drawing_comparison_workflow.py test_pdf_to_dwg_workflow.py test_blockify_pdf_symbols.py
```

과제 2의 샘플 비교와 원본 ZIP에 포함된 비교 테스트도 확인했다. 실제 대형 신규 PDF 두 장의 처리 시간과 정확도는 도면 쌍에 따라 달라지므로 별도 검증이 필요하다.
