# 전기 배선·와이어 하네스 도면: 연결 추출과 Revision 비교 조사

확인일: 2026-10-02. 공개 논문·공식 데이터·저자 저장소만 읽었으며, 회사 도면을 열거나 업로드하지 않았다. 아래 성능은 논문 보고값이며 이 프로젝트에서 재현한 수치가 아니다.

## 판단

**PID2Graph는 P&ID용 데이터셋이다. 전기 배선·와이어 하네스용 완성 모델로 도입할 수 없다.** 참고할 부분은 선분 목록을 넘어 접속점을 포함한 그래프를 만들고, 객체 검출과 연결 정확도를 따로 평가하는 설계다. 하네스에서는 커넥터 외곽보다 **핀↔전선↔스플라이스↔접지** 연결을 복원하는 별도 정의가 필요하다.

## PID2Graph의 정확한 출처와 방법

| 항목 | 확인 결과 |
|---|---|
| 최종 논문 | *From Engineering Diagrams to Graphs: Digitizing P&IDs with Transformers* |
| 저자 | Jan Marius Stürmer, Marius Graumann, Tobias Koch |
| 발표 | IEEE DSAA 2025; 최초 preprint 2024-11-21, 최신 v3 2025-12-19 |
| 최초 제목 | *Transforming Engineering Diagrams: A Novel Approach for P&ID Digitization using Transformers* |
| 식별자 | [arXiv:2411.13929](https://arxiv.org/abs/2411.13929), [출판 DOI](https://doi.org/10.1109/DSAA65442.2025.11248012) |

본문에서는 Relationformer가 객체 토큰과 관계 토큰으로 심볼·연결을 함께 예측한다. 꺾임·교차·패치 경계 노드를 추가하고, 겹치는 패치를 NMS/WBF로 병합한다. 비교 방법은 심볼/문자 검출→문자 필터링→형태학/Hough 선분→끝점을 심볼·선에 연결하는 방식이다. [본문 III](https://arxiv.org/html/2411.13929v3)

| OPEN100 전체 도면 병합 후 | 심볼 mAP | 노드 AP | 연결 edge mAP |
|---|---:|---:|---:|
| 모듈형 비교 방법 | 76.99 | 52.14 | 45.89 |
| Relationformer | 73.14 | 83.63 | 75.46 |

OPEN100은 P&ID 12장이며 점선이 없다. 노드 검출 IoU 기준은 0.5, edge mAP은 Hungarian 노드 매칭 기반 별도 지표다. 75.46−45.89는 **29.57%p** 차이다. 자동차 핀 연결 정확도나 CAD 치수 오차를 뜻하지 않는다. 본문은 큰 심볼의 패치 절단, 오분류, 잘못된 대각 연결을 한계로 보고한다. [본문 III–V, 표 III](https://arxiv.org/html/2411.13929v3)

## 공개 자산과 로컬 실행 가능성

| 자산 | 확인 상태와 도입 조건 |
|---|---|
| PID2Graph 데이터 | [저자 Zenodo 배포](https://zenodo.org/records/14803338), DOI `10.5281/zenodo.14803338`, 9.3GB ZIP. 전체/패치 이미지와 GraphML 정답. 노드 bbox·라벨, edge 라벨, connector·border node 포함. |
| 데이터 권리 | Zenodo는 Dataset-P&ID와 Synthetic 심볼 출처에 CC BY-SA 4.0을 명시한다. OPEN100은 별도 출처를 표시한다. 확인한 페이지에서 전체 묶음의 동일 라이선스를 명확히 확인하지 못했으므로 전부 동일 조건이라고 단정하지 않는다. |
| PID2Graph 논문 저자 구현·가중치 | 논문·Zenodo·저자 기관 페이지 및 이름 검색에서 공식 구현/가중치 다운로드를 확인하지 못했다. “비공개임이 확정”이 아니라 “공개 경로 미확인”이다. [저자 기관 서지](https://elib.dlr.de/221317/) |
| 기반 Relationformer | 원저자 [공식 저장소](https://github.com/suprosanna/relationformer)는 Apache-2.0 코드와 road/vessel/scene graph 분기 제공. **이 저장소가 PID2Graph 적응 코드나 하네스 가중치를 제공한다는 뜻은 아니다.** |

이미지→그래프 연산 자체는 로컬 모델로 구현할 수 있다. 다만 현재 확인만으로 PID2Graph 모델의 다운로드 즉시 추론·완전 재현을 보장할 수 없다. 공개 데이터는 회사망으로 가져와 로컬 연구에 쓰는 방향이고, 회사 도면을 공개 모델 서비스로 전송할 이유는 없다.

## 관련 대안 3개

### 1. PIDCon: 추출한 객체·선을 받아 연결을 학습

Yaxuan Hu, Zhongyuan Wang, Yan Xiong의 *Connectivity Relationship Recognition in Piping and Instrumentation Diagrams Using Graph Neural Networks*. PRCV 2025, 온라인 출판 2026-01-13, [DOI 10.1007/978-981-95-5676-2_4](https://doi.org/10.1007/978-981-95-5676-2_4). P&ID 대상이며 점선·점프선·밀집 배치를 다룬다. 확인한 공식 서지에서 arXiv 번호는 찾지 못했다.

[저자 README](https://github.com/sad123-yx/PIDCon)는 600개 P&ID, 82개 부품 클래스, 7,212개 연결쌍 및 코드·데이터 링크를 제공한다. [models](https://github.com/sad123-yx/PIDCon/tree/main/models)에 `.pth` 파일이 있다. 데이터 전체 공개 예정 문구가 남아 있으며, 현재 공개된 주석 범위와 실제 내려받은 전체 데이터는 구분해야 한다. 저장소 루트에서 명시적 LICENSE는 확인하지 못했다.

[test.py](https://raw.githubusercontent.com/sad123-yx/PIDCon/main/test.py)를 읽으면 bbox·클래스·선 좌표가 든 JSON을 입력해 객체쌍의 연결을 예측한다. README F1=0.963은 원본 이미지부터의 종단간 성능으로 해석하면 안 된다. 코드의 `Accuracy`는 `TP/(TP+FN)`로 Recall과 같다. 도면별 test split/검출 오차 조건을 별도로 맞춰야 PID2Graph와 비교할 수 있다.

**적용 판단:** 로컬 YOLO/OCR/선 추출 뒤 관계 후보를 점수화하는 연구 후보. 전기 핀·스플라이스·비접속 교차로 다시 라벨링해야 한다. 지금 설치한 MVP에 연결된 상태는 아니다.

### 2. V2G-Audit: 벡터 CAD를 그래프로 바꾸고 규칙 검증

Chengwei Ma 외 7명, *Beyond Pixels: A Vector-to-Graph Framework for Reliable Schematic Auditing*, ICASSP 2026. [arXiv:2602.11678v2](https://arxiv.org/abs/2602.11678), [DOI](https://doi.org/10.1109/ICASSP55912.2026.11460425). 전력 전기회로의 접지·배선·단자표기 검사가 대상이며 자동차 하네스 Revision 벤치마크는 아니다.

[본문](https://arxiv.org/html/2602.11678v2)은 ezdxf로 LINE/ARC/TEXT/INSERT를 읽고, 끝점 허용오차와 교차 필터·LLM 해석으로 속성 그래프를 만드는 방법이다. 규칙 검증은 결정적 그래프 연산을 사용한다. 60개 기본 사례의 증강 약 900개에서 보고된 종합 정확도 12%→47%는 감사 태스크 결과이며, 도형 복원 정확도가 아니다. 회전은 OR, 다른 증강은 다수결로 기본 사례 단위 집계하므로 일반 이미지 테스트셋과 직접 비교할 수 없다.

[공식 저장소](https://github.com/gm-embodied/V2G-Audit)는 실제 Python 모듈이 있으나 README에는 미출판/공개 예정 문구와 2025 서지가 남아 최신 논문과 불일치한다. [CLI 소스](https://raw.githubusercontent.com/gm-embodied/V2G-Audit/main/v2g_audit/cli.py)는 로컬 DXF·설정·rules JSON을 입력해 그래프·보고서를 만드는 경로를 보여준다. 루트 LICENSE는 확인하지 못했다. 전체 코드 실행 및 외부 호출 감사는 수행하지 않았다.

**적용 판단:** 원본 벡터 DXF가 있을 때 우선 참고. 스캔 이미지의 누락된 접속점을 해결해 주지 않는다. 논문의 LLM 사용 방식이나 전체 그래프 전송을 그대로 채택하지 말고 로컬 규칙 검증 부분을 분리한다.

### 3. sknw: 스켈레톤에서 끝점·분기점·경로 추출

[Image-Py/sknw 공식 README](https://github.com/Image-Py/sknw)는 BSD-3-Clause 코드로, 가늘어진 이진 이미지에서 NetworkX 그래프를 만든다. 경로 픽셀을 edge에 보존하고 multigraph·닫힌 고리 옵션을 제공한다. 논문 정확도 주장이나 학습 가중치를 가진 하네스 모델은 아니다.

[소스](https://raw.githubusercontent.com/Image-Py/sknw/master/sknw/sknw.py)에서는 이웃 수가 2인 픽셀을 경로로, 나머지를 노드 후보로 분리하고 경로를 추적한다. **적용 판단:** 로컬 선 연결 기준선을 만드는 가장 작은 출발점이다. 교차한 픽셀을 전기적 접속으로 잘못 합칠 수 있으므로 junction dot/점프/비접속 교차 판정은 별도로 해야 한다. 끊어진 스캔·점선·글자 마스킹·심볼 내부 선은 전처리/검증 대상이다.

## 이 프로젝트에 필요한 로컬 그래프 — 설계 제안

이 절은 위 논문의 검증 결과가 아니라 **전기 배선·와이어 하네스라는 사용자 범위에 맞춘 제안**이다.

원본 흐름의 요소 추출 다음에 **접속/경로 그래프 생성**을 넣는다. API가 좌표·선 경로를 보지 않으면 실제 연결 관계를 직접 검증할 수 없으므로, 이 단계는 로컬에서 완결하거나 불확실 상태를 유지해야 한다.

```text
로컬 PDF/이미지
  → 객체·문자·기하 후보
  → 핀/끝점/스플라이스/접지 검출 + 선 추적
  → 접속 교차와 비접속 교차 구별
  → 전기 연결 그래프 + 하네스 분기/배치 그래프
  → 속성 연결·단위 정규화·로컬 검토
  → CAD 생성 또는 Revision 그래프 비교
```

- **연결 노드:** 커넥터 개체와 핀을 구별한다. 핀, splice, ground, wire endpoint가 실제 연결 단위다. 전체 커넥터 bbox에 닿았다는 이유로 모든 핀을 연결하지 않는다.
- **교차:** `junction`, `nonconnecting_crossing`, `unknown_crossing`을 구분한다. junction dot/브리지/선 연속성 및 회사 도면 규약으로 판단한다. 점이 없는 교차도 회사 규약에 따라 달라질 수 있으므로 보편 규칙으로 확정하지 않는다.
- **하네스 분기:** 외형상 묶음 경로의 분기와 전기적 splice는 별개다. 한 bundle 안의 전선들이 서로 전기적으로 연결되었다고 처리하지 않는다. `bundle_segment`와 `wire_edge`를 구분하고 membership으로 연결한다.
- **길이:** OCR로 읽은 표기 길이, CAD/픽셀 경로 길이, 실제 절단 길이를 별도 속성으로 둔다. schematic 경로의 화면 길이만으로 실물 전선 길이를 산출하지 않는다.
- **근거:** OCR 원문, bbox/페이지, 검출 confidence, 적용 규칙, 사용자 확정 여부를 로컬 보존한다. OCR confidence만으로 관계 confidence를 대체하지 않는다.
- **최종 CAD:** polyline의 꺾임점과 그래프 분기점은 다르다. 곡선·원호의 실제 기하와 심볼 블록/속성은 topology와 별도로 보존해야 편집 가능한 DWG를 생성할 수 있다.

아래는 회사정보가 아닌 가상 로컬 데이터 예시다. 외부 API 전송 형식이 아니다.

```json
{
  "nodes": [
    {"id": "P1", "kind": "connector_pin", "connector_id": "C1", "pin_no": "1"},
    {"id": "S1", "kind": "splice"}
  ],
  "wire_edges": [
    {
      "id": "W1", "from": "P1", "to": "S1",
      "path_id": "PATH1", "wire_size_sq_mm": 0.5,
      "annotated_length_mm": 150,
      "measured_path_length_mm": null,
      "cut_length_mm": null,
      "status": "needs_review", "evidence_ids": ["OCR1", "LINE1"]
    }
  ],
  "branches": [
    {"id": "BR1", "kind": "bundle_branch", "name": "가상분기A", "wire_ids": ["W1"]}
  ],
  "crossings": [
    {"path_ids": ["PATH1", "PATH2"], "connectivity": "unknown", "evidence_ids": ["REGION2"]}
  ]
}
```

## Revision 전후 비교 — 설계 제안

1. **정렬:** 페이지·도면 뷰를 먼저 대응시킨다. 공통 기준점으로 이동·회전·축척을 정렬하고 정렬 오차를 저장한다. 유연한 이미지 변형으로 실제 배선 변경을 지워 버리지 않도록 한다.
2. **개체 대응:** 신뢰할 수 있는 커넥터/핀/부품 식별자를 우선 사용하되, 재명명 가능성을 고려해 종류·이웃 연결·정렬 좌표로 보완한다. 두 revision에서 새로 부여된 `LINE_0001` 같은 순번 ID를 동일 개체로 간주하지 않는다.
3. **속성 비교:** 단위 정규화 후 `150→180 mm`, `0.5→0.75 SQ`, 부품번호, 분기명 등을 비교한다. OCR 흔들림은 `변경 의심`으로 남기고 사람이 원문을 확인할 수 있게 한다.
4. **연결 비교:** 대응된 핀/스플라이스 ID를 기준으로 edge 추가·삭제·재연결을 찾는다. 분기점 신설/삭제 및 bundle membership 변화도 따로 비교한다. 단순히 선을 여러 조각으로 검출한 차이는 degree-2 경로 단순화로 흡수한다.
5. **배치 비교:** 연결이 같아도 경로가 바뀌면 `경로 변경`으로 기록한다. 연결까지 달라졌는지, 단순 이동/회전인지 구별한다. 의미 비교용 단순화 그래프와 원래 CAD 기하는 둘 다 보존한다.
6. **로컬 시각화:** 추가/삭제/속성/연결/경로 변경 레이어 및 전후 값 표를 생성한다. 변경 개체에서 원본의 해당 영역으로 연결하고, 매칭 불확실성도 표시한다.

검증용으로 `동일 도면 이동/회전`, `치수만 변경`, `핀 하나 재연결`, `교차점에 접속점 추가`, `분기명만 변경`, `부품 추가/삭제`, `같은 연결의 경로만 이동` 쌍을 따로 준비한다. 개발/평가는 도면 또는 프로젝트 단위로 나누고, 같은 도면의 패치가 학습/시험 양쪽에 섞이지 않도록 한다. 지표는 핀 연결 precision/recall, 접속 교차 오판, 속성값 정확도, 개체 매칭 정확도, 변경 종류별 precision/recall을 사용한다.

외부 API는 승인된 최소 정보의 의미 분류나 **이미 로컬에서 계산한** 변경 설명을 생성할 수 있다. 원본과 전체 구조를 받지 않은 API가 누락 선·잘못 연결된 핀·최종 DWG의 원본 충실도를 검증했다고 주장하면 안 된다. 그런 검증과 최종 도면/변경 결과 생성은 로컬 그래프·기하·근거 데이터로 수행한다.
