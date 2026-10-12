# 전기 배선·와이어 하네스 도면: YOLO11-OBB + Donut 조사

조사일: 2026-10-02. 공개 논문·저자 저장소·모델 제공자 문서를 조사했다. 회사 도면은 읽거나 업로드하지 않았다. 논문의 보고 성능을 재현 실행한 것은 아니다.

## 적용 판단

**YOLO11-OBB로 주석 영역을 찾고 로컬 Donut으로 JSON을 읽는 방식은 유효한 후보다. 하지만 조사한 제조도면 논문은 기계부품의 치수·GD&T 중심이다. 전기 배선/하네스의 핀 연결, 접속/비접속 교차, SQ, 색상, 분기명을 자동으로 이해한다는 근거가 아니다.** 이 프로젝트에서는 영역 검출·주석 해석 모듈의 참고 연구로 사용하고, 배선 그래프·CAD 형상 복원·Revision 비교는 별도로 설계해야 한다.

Donut은 이미지 입력을 받는 모델이다. Donut을 사내 PC에서 실행하면 원본과 크롭 모두 로컬에서 처리할 수 있다. 반대로 외부 Donut 서비스나 클라우드 VLM에 크롭을 보내면 원본 일부가 외부에 전달된다. 아래 논문을 적용한다고 해서 이미지 미전송이 자동으로 보장되는 것은 아니다.

## 1. 요청한 논문의 정확한 식별

**Automated Parsing of Engineering Drawings for Structured Information Extraction Using a Fine-tuned Document Understanding Transformer**

- 저자: Muhammad Tayyab Khan, Zane Yong, Lequn Chen, Jun Ming Tan, Wenhe Feng, Seung Ki Moon.
- 2025년; arXiv v1 2025-05-02, 확인한 v3 2025-09-03. arXiv에는 IEEE IEEM 게재 승인으로 표시된다.
- [arXiv:2505.01530v3](https://arxiv.org/abs/2505.01530v3), [본문 PDF](https://arxiv.org/pdf/2505.01530v3), [arXiv DOI](https://doi.org/10.48550/arXiv.2505.01530).

| 항목 | 확인 내용 |
|---|---|
| YOLO11-OBB | 회전된 주석의 영역·범주 검출 및 크롭 |
| Donut | 크롭 이미지→구조화 JSON; 단일 모델과 범주별 모델 비교 |
| 데이터 | 기계도면 1,367장→11,469패치; 수동 라벨 1,000패치를 6,000개로 증강 |
| 범주 | GD&T, 일반공차, 재질, 치수, 비고, 반지름, 표면거칠기, 나사, 표제란 |
| 학습 | RTX 4090 한 대, batch 1, 30 epoch; 단일 모델 약 20시간 |
| 해석 주의 | GD&T precision 94.77%, hallucination 5.23%; 초록의 F1 97.3%를 전체 CAD 정확도로 쓰면 안 됨 |
| 취약 범주 | 단일 모델 표제란 F1 72.9%; 재질/일반공차 hallucination 80.0%/66.7% |

5쪽 본문에는 명시적인 train/validation/test 분할 조건을 확인하지 못했다. 높은 수치만으로 독립 도면 일반화나 하네스 성능을 판단할 수 없다. 위 내용은 논문 §II–III에 근거한다. [원문](https://arxiv.org/pdf/2505.01530v3)

## 2. 함께 봐야 할 확장 연구와 수치 차이

**From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D Engineering Drawings into Structured Manufacturing Knowledge** — Muhammad Tayyab Khan, Lequn Chen, Zane Yong, Jun Ming Tan, Wenhe Feng, Seung Ki Moon.

- [arXiv:2506.17374v2](https://arxiv.org/abs/2506.17374v2), [본문](https://arxiv.org/pdf/2506.17374v2).
- 저널판: *Robotics and Computer-Integrated Manufacturing* 99, 103186, 2026; [DOI 10.1016/j.rcim.2025.103186](https://doi.org/10.1016/j.rcim.2025.103186). 출판사 검색 결과의 서지·초록은 확인했으나 출판사 전체 본문 직접 열기는 실패하여 세부 평가는 arXiv v2 본문을 기준으로 했다.

| 기준 | Donut precision | recall | F1 | hallucination |
|---|---:|---:|---:|---:|
| arXiv 랜딩 초록 | 88.5% | 99.2% | 93.5% | 11.5% |
| v2 PDF Table 5·저널 초록 | 89.2% | 99.2% | 94.0% | 10.8% |

이 차이를 숨기거나 서로 섞어 인용하면 안 된다. Table 5는 11,469패치의 10% 테스트 부분집합에서 JSON key-value 정확 일치로 평가한다. hallucination은 `1 − precision` 정의이며 일반적인 사실성 평가와 동일하지 않다. YOLO 검출은 도면 단위 90:10 분할을 명시하지만, 이를 파서 패치 분할의 누출 방지 근거로 확대할 수 없다. 표제란 F1은 53.3%, 일반공차는 66.7%다. [v2 §4.1–4.2](https://arxiv.org/pdf/2506.17374v2)

이 연구의 rule-based 후처리는 제조공정·공구 선택이다. 전기 네트워크 생성, 편집가능 DWG, Revision 변경 검출을 검증한 논문은 아니다.

## 3. 공개 자산과 사내 적용 가능성

| 자산 | 확인 결과 | 적용 시 의미 |
|---|---|---|
| 위 두 논문의 전용 코드 | 논문·arXiv·제목 검색에서 저자 공개 저장소를 확인하지 못함 | 논문 그대로 즉시 실행 가능한 패키지로 간주하지 않음 |
| 위 두 논문의 도면 특화 가중치·라벨 데이터 | 다운로드 위치·배포 라이선스를 확인하지 못함 | 공개 출처에서 수집했다는 서술이 학습 데이터 공개를 뜻하지 않음 |
| Donut 일반 구현 | [공식 GitHub](https://github.com/clovaai/donut), MIT | 로컬 학습·추론 구현 재사용 가능 |
| Donut 일반 base 가중치 | [공식 모델 카드](https://huggingface.co/naver-clova-ix/donut-base), MIT 표기 | 하네스 전용 가중치가 아님 |
| YOLO11-OBB 구현·일반 가중치 | [공식 YOLO11 문서](https://docs.ultralytics.com/models/yolo11/)에 학습·추론·export 지원 | 하네스 클래스용 라벨링·재학습 필요 |
| Ultralytics 이용 조건 | AGPL-3.0 / Enterprise. 공급자는 비공개 상용 결합에 Enterprise를 안내 | 사내 배포 방식에 맞는 라이선스 검토 필요; 회사 도면을 공개해야 한다고 단정하는 뜻은 아님 |

라이선스 근거: [Donut 구현](https://github.com/clovaai/donut#license), [Donut 모델](https://huggingface.co/naver-clova-ix/donut-base), [Ultralytics 공식 안내](https://www.ultralytics.com/license). 논문 PDF의 저작권과 코드·가중치·데이터의 이용 조건은 각각 별개다. “찾지 못함”은 비공개임을 확정한 표현이 아니다.

사내 추론에서는 검증한 가중치·토크나이저·설정 파일을 사전에 준비하고 로컬 경로만 사용한다. 자동 모델 다운로드, 원격 inference, 외부 실험 추적을 끈 상태에서 실행하고, egress 차단 환경에서 동작하는지 확인한다. 이것은 이 프로젝트에 대한 권고이며 논문이 보증한 보안 특성이 아니다.

## 4. 추가로 읽을 1차 자료

| 자료 | 핵심 내용 | 이 프로젝트에서의 위치 |
|---|---|---|
| Geewook Kim 외, **OCR-free Document Understanding Transformer**, ECCV 2022; [arXiv:2111.15664](https://arxiv.org/abs/2111.15664) | 별도 OCR 없이 문서 이미지에서 구조화 결과 생성. 공식 구현·모델·합성 데이터 제공 | Donut 학습 원리 참고. 기본 모델이 하네스 규격을 안다고 가정하지 않음 |
| Khan 외, **Fine-Tuning Vision-Language Model for Automated Engineering Drawing Information Extraction**, 2024; [arXiv:2411.03707](https://arxiv.org/abs/2411.03707), [본문](https://arxiv.org/pdf/2411.03707) | 기계도면 400장으로 0.23B Florence-2를 GD&T 추출에 미세조정. 폐쇄형 모델은 zero-shot 비교 | 도메인 학습의 가치. 서로 다른 학습 조건이므로 모델 자체 우열이나 하네스 성능으로 일반화하지 않음. [일반 Florence-2 가중치](https://huggingface.co/microsoft/Florence-2-base)는 MIT 표기 |
| Javier Villena Toro, Mehdi Tarkian, **Optimizing Text Recognition in Mechanical Drawings: A Comprehensive Approach**, *Machines* 13(3), 254, 2025; [DOI](https://doi.org/10.3390/machines13030254) | eDOCr2: 도면 구획화+OCR. 연구 벤치마크 text recall 93.75%, CER <1%; 저자 코드·데이터 공개 안내 | OCR baseline 후보. [저자 저장소](https://github.com/javvi51/edocr2)는 MIT이며 recognizer는 Releases 안내. 외부 GPT 연동 예제는 채택하지 않고 로컬 경로만 사용 |

eDOCr2 출판사 본문은 직접 열기 오류가 있었고, 출판사 검색 인덱스의 본문·초록과 저자 저장소/라이선스를 함께 확인했다. recognizer 자산의 실제 다운로드 및 실행 검증은 수행하지 않았다.

추가 후속 연구 [**A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model**](https://arxiv.org/abs/2510.21862v2)는 layout→OBB→Donut으로 나누고, 문자 영역과 수치 영역을 별도 모델로 처리한다. v2 본문은 수치 모델만 미세조정하고 문자 모델은 zero-shot으로 사용했다고 명시한다. F1 0.963과 0.672는 서로 다른 영역·학습 조건의 결과다. 전체 도면이 복잡할수록 먼저 구획화한다는 구조적 참고자료다. [본문 §4.3](https://arxiv.org/pdf/2510.21862v2)

## 5. 하네스·DWG·Revision 요구에 대한 설계 반영

아래는 문헌을 바탕으로 한 프로젝트 설계 판단이다. 현재 구현 또는 검증 완료를 뜻하지 않는다.

1. **검출과 판독을 분리한다.** 커넥터·단자·스플라이스·접지·접속점·주석/표/핀목록을 로컬에서 검출한다. 핀번호, 전선 ID, 색상, SQ/AWG, 길이, 분기명은 원문 값·정규화 값·근거 bbox를 함께 보관한다. Donut은 판독 후보이며 PaddleOCR 등 로컬 OCR baseline과 비교한다.
2. **OBB를 배선 경로로 오인하지 않는다.** OBB는 둘러싼 상자다. 중심선·끝점·곡선·분기·교차점·핀 접점은 별도 로컬 형상/그래프 단계에서 만든다. 겹쳐 지나가는 선과 전기 접속점을 반드시 구별한다.
3. **기하와 속성을 각각 검증한다.** `0.5SQ` 판독이 맞아도 어느 전선에 붙는지 틀릴 수 있다. 관계 ID·리더선·핀 연결·불확실성까지 검증한 뒤 CAD 속성에 반영한다. 치수 주석과 픽셀 거리가 다르면 주석으로 실제 길이를 추론할지 도면 배치를 보존할지 정책을 명시한다.
4. **DWG는 로컬에서 생성한다.** 검출/파서 출력에서 검증된 LINE/ARC/CIRCLE/SPLINE/TEXT/BLOCK과 속성을 로컬 CAD API로 생성한다. 미해결 연결·핀번호를 LLM 추측으로 채우지 않는다. 원본 위에 재투영한 로컬 overlay와 CAD 재열기로 편집성·좌표·단위를 확인한다.
5. **Revision은 픽셀 차이와 의미 차이를 나눈다.** 먼저 도면을 정합하여 스캔 이동·회전·축척 변화를 보정한다. 이어 커넥터/핀/전선/분기의 대응을 만들고 치수값·전선 규격·색상·분기명·부품 추가삭제·연결 변경을 비교한다. 이름만 변경된 객체를 삭제+추가로 처리하지 않도록 공간·주변 연결을 함께 본다.

## 6. 최소 검증 데이터와 지표

회사 도면 검증은 사내에서만 실시한다. train/test는 같은 원도면의 크롭이나 증강본이 섞이지 않게 원도면/설계계열 단위로 분리하고, Revision 쌍도 관련 도면끼리 묶는다.

| 검증 대상 | 반드시 따로 측정할 항목 |
|---|---|
| 문자·속성 | 핀번호/부품번호 exact match, SQ/색상/길이/분기명 정확도; CER만으로 대체하지 않음 |
| 연결 | 핀↔전선, 스플라이스↔분기 edge precision/recall; 교차/접속 혼동 건수 |
| 기하 | 기준점·선·원·곡선의 좌표 오차와 축척 오차; 끊김·중복선 |
| CAD 결과 | 편집 가능한 객체 종류·개수·레이어·속성 보존 및 파일 재열기 |
| Revision | 변경 종류별 precision/recall, 실제 변화 없는 스캔본의 오탐률 |
| 정보 유출 | 회사 자료 처리 중 네트워크 미사용; 선택 API 요청은 허용 스키마와 실제 직렬화 바이트 검사 |

외부 API가 받는 것이 최소 JSON뿐이라면 그 JSON 안의 의미 후보만 판단할 수 있다. 원본 누락·OCR 오독·잘못된 접속점을 API가 시각적으로 검증할 수 없으므로 최종 일치 검증은 로컬에 남겨야 한다.
