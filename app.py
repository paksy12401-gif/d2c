"""Run locally: python -m streamlit run app.py"""
import json
import os
from io import BytesIO
from pathlib import Path
import tempfile
import cv2
import models_local  # Heavy ML imports happen only when explicitly enabled.
from PIL import Image

import streamlit as st

from cad import build_dxf, dxf_to_dwg
from pipeline import analyze, finalize, load_rules, suggest_links
from privacy import make_payload, request_selection
from vision import extract_geometry, preprocess, draw_overlay
from topology import build_topology, compare_revisions
from revision_compare import compare_documents, serializable_report
from llm_revision import build_revision_evidence, request_revision_review
from unified_revision import run_local_analysis

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="로컬 하네스 도면 분석", layout="wide")
st.title("로컬 하네스 도면 분석")
st.caption("구조 검증용 초기 구현 · 원본 처리와 결과 생성은 이 PC에서 수행합니다.")
st.info("학습한 로컬 YOLO와 선택 OCR을 연결할 수 있습니다. 검증된 산업 제어회로 모델은 아직 포함되어 있지 않습니다. "
        "배선·교차·핀 연결과 CAD 축척은 원본을 보며 검토해야 합니다.")

with st.sidebar:
    st.subheader("OpenAI API 설정")
    api_key = st.text_input("OpenAI API 키", type="password", key="api_key",
                            help="이 브라우저 세션에서만 사용합니다. 환경변수 OPENAI_API_KEY도 지원합니다.")
    model_id = st.text_input("API 모델", value=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), key="model_id")
    st.caption("키 없이도 로컬 분석은 실행됩니다. 키는 저장 파일·도면 JSON에 넣지 않습니다.")
    if st.button("합성 JSON으로 API 연결 확인", key="api_test"):
        try:
            answer = request_selection({"token_kind": "length_unit", "unit": "mm", "options": [
                {"target": "a", "kind": "wire", "evidence": "dimension_line", "roles": ["length"]}]},
                model_id, enabled=True, api_key=api_key)
            st.success("API 호출과 JSON 응답 검증 성공")
            st.json(answer)
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("로컬 인식 모델")
    vector_mode = st.selectbox("벡터화 방식", ["중심선·원·곡선 (개선)", "Hough 직선 (기존)"])
    from paddle_ocr_local import model_dirs, models_ready
    paddle_ready = models_ready()
    easy_ready = (ROOT / "models/easyocr/korean_g2.pth").is_file()
    engines = (["PaddleOCR PP-OCRv5 (기본)"] if paddle_ready else []) + \
              (["EasyOCR (비교용)"] if easy_ready else []) + ["OCR 사용 안 함"]
    ocr_engine = st.selectbox("OCR 엔진", engines, index=0,
                              help="모두 이 PC에서만 실행하며 도면을 외부로 전송하지 않습니다.")
    use_ocr = ocr_engine.startswith("PaddleOCR")
    use_easyocr = ocr_engine.startswith("EasyOCR")
    ocr_confidence = st.slider("OCR 최소 신뢰도", 0.05, 0.95, 0.50, 0.05)
    use_yolo = st.checkbox("학습한 YOLO 모델 사용", value=False)
    weights_path = st.text_input("로컬 가중치 경로", value=str(ROOT / "models/control_circuit_best.pt"))
    detector_task = st.selectbox("검출 방식", ["detect", "obb"], help="일반 사각형 라벨은 detect, 회전 사각형 라벨은 obb입니다.")
    confidence = st.slider("기호 검출 신뢰도", 0.05, 0.95, 0.25, 0.05)
    default_det, default_rec = model_dirs()
    det_dir = st.text_input("PaddleOCR 검출 모델 폴더", value=str(default_det))
    rec_dir = st.text_input("PaddleOCR 인식 모델 폴더", value=str(default_rec))
    block_library = st.text_input("CAD BLOCK 라이브러리 DXF", value="")
    block_mapping_path = st.text_input("기호→BLOCK 매핑 JSON", value="")


@st.cache_resource
def cached_easy_reader():
    from ocr_local import load_reader
    return load_reader()


@st.cache_resource
def cached_paddle_reader(detector_dir, recognizer_dir):
    from models_local import load_ocr
    reader = load_ocr(detector_dir, recognizer_dir)
    reader._cad_model_root = str(Path(detector_dir).parent.parent)
    return reader


def process_image(data, extension, page_index=0, angle=0):
    improved = vector_mode.startswith("중심선")
    gray = preprocess(data, extension, page_index, angle, preserve_thin_lines=improved)
    symbols, texts = [], []
    if use_yolo or use_ocr:
        from models_local import load_yolo, detect_symbols, load_ocr, recognize_texts
        if use_yolo:
            detector = load_yolo(weights_path, task=detector_task)
            class_map = json.loads(Path(block_mapping_path).read_text(encoding="utf-8")) if block_mapping_path else {"GND": {"kind": "ground"}}
            symbols = detect_symbols(gray, detector, class_map=class_map, conf=confidence, imgsz=640)
        if use_ocr:
            ocr = cached_paddle_reader(det_dir, rec_dir)
            texts = recognize_texts(gray, ocr, min_confidence=ocr_confidence)
    if use_easyocr and not use_ocr:
        from ocr_local import read_text
        texts = read_text(gray, cached_easy_reader())
    masked = gray.copy()
    for item in symbols + ([] if improved else texts):
        x, y, w, h = item["bbox"]
        masked[max(0, int(y)):min(gray.shape[0], int(y+h+1)),
               max(0, int(x)):min(gray.shape[1], int(x+w+1))] = 255
    if improved:
        from vectorize_local import extract_vectors
        result = extract_vectors(masked, texts)
    else:
        result = extract_geometry(masked)
    result["objects"].extend(symbols)
    result["texts"] = texts
    if not use_ocr and not use_easyocr:
        result["warnings"].append("OCR을 사용하지 않았습니다. 문자 의미 분석에는 로컬 OCR 모델 또는 구조 JSON이 필요합니다.")
    result["graph"] = build_topology(gray, result["objects"], texts)
    return gray, suggest_links(result)

for key, default in {"structure": None, "preview": None, "decisions": {}, "suggestions": {},
                     "revision": 0, "dxf": None, "dwg": None, "loaded_name": "",
                     "visual_revision": None, "visual_revision_source": None,
                     "llm_revision_review": None, "unified_revision": None,
                     "unified_revision_review": None,
                     "unified_revision_source": None}.items():
    st.session_state.setdefault(key, default)

comparison_root = ROOT / "output/reference_comparison"
comparisons = sorted((p.parent for p in comparison_root.glob("*/report.json")),
                     key=lambda p:(p.name!='y_delta_paddle',p.name))
load_comparison = False
if comparisons:
    st.subheader("실제 입력 변환 비교")
    selected = st.selectbox("비교 도면", comparisons, format_func=lambda p:p.name)
    report = json.loads((selected / "report.json").read_text(encoding="utf-8"))
    st.caption(f"OCR: {report.get('ocr_engine', 'easy')} · 기존/개선 DXF를 다시 그린 결과 · 축척 0.1 mm/px는 미보정 시험값")
    cols = st.columns(3)
    for col,name,label in zip(cols,["original","baseline","improved"],["원본","기존 DXF","개선 DXF"]):
        col.image(str(selected / f"{name}.png"),caption=label,width="stretch")
    old,new = report["methods"]["baseline"],report["methods"]["improved"]
    stats = st.columns(3)
    stats[0].metric("원본 선 흔적 재현율 (2px)",f"{new['geometry_ink_recall']:.1%}",
                    delta=f"{(new['geometry_ink_recall']-old['geometry_ink_recall'])*100:.1f}%p")
    stats[1].metric("추출 도형 / OCR",f"{sum(new['entities'].values())} / {new['ocr_texts']}")
    stats[2].metric("OCR + 벡터화 시간",f"{report['ocr_seconds_including_load_if_uncached']+new['vectorization_seconds']:.1f}초")
    st.warning("재현율은 OCR 영역을 제외한 원본 잉크와의 픽셀 비교입니다. 회로 정확도·접속 정확도·OCR 정답률이 아닙니다. 작은 문자와 접점은 원본과 대조하세요.")
    st.download_button("개선 DXF 바로 다운로드",(selected / "improved.dxf").read_bytes(),
                       file_name=f"{selected.name}_review.dxf",mime="application/dxf")
    load_comparison = st.button("이 결과를 아래에서 검토",key="load_comparison")
    with st.expander("인식된 문자와 원본 점검"):
        st.dataframe([{"문자":t["text"],"모델 신뢰도":round(t["confidence"],3)} for t in report["texts"]],hide_index=True)


def _png_bytes(image):
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def show_revision_result(report, overview, after_aligned, crop_lookup, *, key_prefix,
                         original_after=None):
    """Render a compact list and enlarged before/after crops."""
    count = report["change_count"]
    difference_stats = report.get("difference_stats")
    metrics = st.columns(4 if difference_stats else 3)
    metrics[0].metric("변경 후보", count)
    metrics[1].metric("정렬 일치점", report["alignment"]["matched_features"])
    metrics[2].metric("정렬 내부점 비율", f"{report['alignment']['inlier_ratio']:.1%}")
    if difference_stats:
        metrics[3].metric("변경 선 흔적", f"{difference_stats['changed_page_ratio']:.2%}")
    st.image(overview, caption="정렬된 후도면 · 주황 테두리: 변경 후보", width="stretch")
    with st.container(horizontal=True):
        st.download_button("정렬된 변경후 도면 PNG", _png_bytes(after_aligned),
                           file_name="after.aligned.png", mime="image/png",
                           key=f"{key_prefix}_after_png", icon=":material/download:")
        st.download_button("변경 위치 표시 도면 PNG", _png_bytes(overview),
                           file_name="after.changes.png", mime="image/png",
                           key=f"{key_prefix}_overview_png", icon=":material/download:")
        if original_after is not None:
            st.download_button("원본 변경후 도면", original_after[0],
                               file_name=original_after[1], mime="application/octet-stream",
                               key=f"{key_prefix}_original_after", icon=":material/download:")
    rows = [{"번호": item["id"], "인식한 변경 내용": item["label"],
             "OCR 신뢰도": None if item["ocr_confidence"] is None else round(item["ocr_confidence"], 3),
             "검출 방식": item["source"]} for item in report["changes"]]
    st.dataframe(rows, hide_index=True, width="stretch")
    has_red_annotations = any(item["source"] == "red_annotation" for item in report["changes"])
    if has_red_annotations:
        st.caption("아래 목록은 전도면과 정렬된 후도면의 같은 위치를 확대합니다. 빨간 글씨는 후도면에 포함된 변경 주석입니다.")
    else:
        stats = report.get("difference_stats", {})
        detail = (f" 전체 군집 {stats.get('all_region_count')}개 중 큰 후보 {stats.get('shown_region_count')}개가 "
                  f"변경 픽셀 {stats.get('shown_changed_pixel_coverage', 0):.1%}를 포함합니다.") if stats else ""
        st.caption("아래 목록은 빨간 정답 없이 검출한 선 흔적 차이입니다. 각 항목에서 전도면과 후도면의 동일 위치를 확대합니다." + detail)
    for item in report["changes"]:
        confidence = "" if item["ocr_confidence"] is None else f" · OCR {item['ocr_confidence']:.1%}"
        with st.expander(f"{item['id']} · {item['label']}{confidence}"):
            before_crop, after_crop = crop_lookup(item["id"])
            left, right = st.columns(2)
            left.image(before_crop, caption="변경 전", width="stretch")
            right.image(after_crop, caption="변경 후", width="stretch")
            st.caption(f"픽셀 영역: {item['bbox_px']}")
    st.download_button("변경점 JSON 다운로드", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="visual_changes.local.json", mime="application/json",
                       key=f"{key_prefix}_download")
    for warning in report.get("warnings", []):
        st.warning(warning)


st.header("전·후도면 통합 AI 변경 분석")
st.caption(
    "전도면과 후도면을 넣고 한 번만 실행하세요. 픽셀 차이, PaddleOCR 문자·숫자, "
    "선택적 YOLO 기호, 배선 연결 후보를 로컬에서 합친 뒤 LLM이 실제 변경과 렌더링 노이즈를 분류합니다."
)
st.info(
    "PDF와 원본 이미지는 외부로 보내지 않습니다. LLM 사용 시에는 실제 OCR 문자열, 좌표, "
    "기호·배선 후보, 픽셀 통계가 포함된 구조화 JSON만 OpenAI API로 전송됩니다."
)

with st.form("unified_revision_compare"):
    upload_columns = st.columns(2)
    before_revision = upload_columns[0].file_uploader(
        "전도면 PDF/이미지", type=["pdf", "png", "jpg", "jpeg"], key="unified_before")
    after_revision = upload_columns[1].file_uploader(
        "후도면 PDF/이미지", type=["pdf", "png", "jpg", "jpeg"], key="unified_after")
    option_columns = st.columns(4)
    sensitivity_label = option_columns[0].selectbox(
        "변경 검출 민감도", ["매우 민감", "균형", "엄격"], index=0,
        help="매우 민감은 작은 차이를 넓게 잡고 LLM이 뒤에서 걸러냅니다.")
    render_dpi = option_columns[1].selectbox(
        "렌더링 DPI", [100, 120, 144], index=0,
        help="작은 글자가 빠지면 DPI를 올리세요. 처리 시간과 메모리도 증가합니다.")
    run_llm = option_columns[2].checkbox(
        "LLM 최종 판정", value=bool(api_key or os.environ.get("OPENAI_API_KEY")),
        help="원본 대신 구조화 근거 JSON을 전송합니다.")
    review_passes = option_columns[3].selectbox(
        "LLM 검토", [2, 1],
        format_func=lambda value: "2회 교차검토" if value == 2 else "1회 분석")
    unified_clicked = st.form_submit_button(
        "전체 변경 분석 실행", type="primary", icon=":material/manage_search:",
        use_container_width=True)

if unified_clicked:
    try:
        if before_revision is None or after_revision is None:
            raise ValueError("전도면과 후도면 파일 두 개를 선택하세요.")
        if max(before_revision.size, after_revision.size) > 20 * 1024 * 1024:
            raise ValueError("각 파일은 20MB 이하여야 합니다.")
        sensitivity = {"매우 민감": "high", "균형": "balanced", "엄격": "strict"}[sensitivity_label]
        with st.status("전·후도면 전체 분석 중", expanded=True) as status:
            st.write("1/4 도면 렌더링·정렬 및 고감도 픽셀 후보 검출")
            reader = cached_paddle_reader(det_dir, rec_dir) if use_ocr else None
            detector = None
            class_map = None
            if use_yolo:
                if not Path(weights_path).is_file():
                    st.warning("YOLO 가중치가 없어 기호 단계만 건너뜁니다. 픽셀·OCR·배선 비교는 계속합니다.")
                else:
                    from models_local import load_yolo
                    detector = load_yolo(weights_path, task=detector_task)
                    class_map = (json.loads(Path(block_mapping_path).read_text(encoding="utf-8"))
                                 if block_mapping_path else {})
            st.write("2/4 PaddleOCR 문자·숫자와 선택적 YOLO 기호를 전후 동일 좌표로 추출")
            unified = run_local_analysis(
                before_revision.getvalue(), after_revision.getvalue(),
                Path(before_revision.name).suffix, Path(after_revision.name).suffix,
                dpi=render_dpi, sensitivity=sensitivity,
                ocr_reader=reader, ocr_confidence=ocr_confidence,
                detector=detector, detector_confidence=confidence,
                class_map=class_map,
            )
            st.write("3/4 픽셀·OCR·기호·배선 근거를 후보별 JSON으로 결합")
            review = unified["local_review"]
            if run_llm and not unified["evidence"]["candidates"]:
                st.info("검출된 변경 후보가 없어 LLM 호출을 생략했습니다.")
            elif run_llm:
                if not (api_key or os.environ.get("OPENAI_API_KEY")):
                    st.warning("API 키가 없어 로컬 후보 판정까지만 완료했습니다.")
                else:
                    st.write("4/4 LLM이 후보를 배치별로 판정하고 최종 변경 설명 생성")
                    try:
                        review = request_revision_review(
                            unified["evidence"], model_id, enabled=True,
                            api_key=api_key, passes=review_passes, batch_size=20)
                    except (ValueError, PermissionError) as exc:
                        st.warning(f"LLM 판정은 실패했지만 로컬 결과는 보존했습니다: {exc}")
            unified["review"] = review
            st.session_state.unified_revision = unified
            st.session_state.unified_revision_review = review
            st.session_state.unified_revision_source = (
                after_revision.getvalue(), after_revision.name)
            source = "LLM 최종 판정" if review.get("analysis_source") == "openai_llm" else "로컬 예비 판정"
            status.update(
                label=f"변경 후보 {len(unified['report']['changes'])}개 · {source} 완료",
                state="complete", expanded=False)
    except (ValueError, TypeError, ImportError, OSError, cv2.error) as exc:
        st.error(str(exc))

if st.session_state.unified_revision is not None:
    unified = st.session_state.unified_revision
    report = unified["report"]
    review = unified["review"]
    review_by_id = {item["change_id"]: item for item in review["changes"]}
    verdict_labels = {
        "confirmed": "확정", "probable": "가능성 높음", "uncertain": "불확실",
        "rendering_noise": "렌더링 노이즈",
    }
    type_labels = {
        "text_value_changed": "문자/수치 변경", "text_added": "문자 추가",
        "text_removed": "문자 삭제", "text_unreadable_changed": "판독 불가 문자 변경",
        "symbol_added": "기호 추가", "symbol_removed": "기호 삭제",
        "symbol_changed": "기호 변경", "geometry_added": "형상 추가",
        "geometry_removed": "형상 삭제", "geometry_moved": "형상 이동",
        "connection_added": "연결 추가", "connection_removed": "연결 삭제",
        "mixed_change": "복합 변경", "no_semantic_change": "의미 변화 없음",
        "unknown": "분류 필요",
    }
    counts = {name: sum(item["verdict"] == name for item in review["changes"])
              for name in verdict_labels}
    metric_columns = st.columns(6)
    metric_columns[0].metric("전체 픽셀 군집", (report.get("difference_stats") or {}).get("all_region_count", report["change_count"]))
    metric_columns[1].metric("상세 분석 후보", report["change_count"])
    metric_columns[2].metric("확정", counts["confirmed"])
    metric_columns[3].metric("가능성 높음", counts["probable"])
    metric_columns[4].metric("불확실", counts["uncertain"])
    metric_columns[5].metric("노이즈", counts["rendering_noise"])
    source_label = ("OpenAI LLM 구조화 판정" if review.get("analysis_source") == "openai_llm"
                    else "로컬 예비 판정")
    st.success(f"판정 방식: {source_label} · {review['summary']}")
    st.image(unified["visual"]["overview"],
             caption="정렬된 후도면 · 주황 상자: 고감도 변경 후보", width="stretch")

    show_noise = st.checkbox("렌더링 노이즈 판정도 목록에 표시", value=False)
    final_rows = []
    for item in review["changes"]:
        if item["verdict"] == "rendering_noise" and not show_noise:
            continue
        final_rows.append({
            "번호": item["change_id"],
            "판정": verdict_labels.get(item["verdict"], item["verdict"]),
            "무엇이 달라졌나": type_labels.get(item["change_type"], item["change_type"]),
            "변경 전": item["before_value"], "변경 후": item["after_value"],
            "신뢰도": round(item["confidence"], 3),
            "사람 검토": "필요" if item["needs_review"] else "선택",
            "판정 근거": item["reason"],
        })
    st.subheader("최종 변경 목록")
    st.dataframe(final_rows, hide_index=True, width="stretch")

    with st.container(horizontal=True):
        st.download_button(
            "최종 판정 JSON", json.dumps(review, ensure_ascii=False, indent=2),
            file_name="revision.final.json", mime="application/json",
            key="unified_final_json", icon=":material/download:")
        st.download_button(
            "LLM 전달 근거 JSON", json.dumps(unified["evidence"], ensure_ascii=False, indent=2),
            file_name="revision.evidence.json", mime="application/json",
            key="unified_evidence_json", icon=":material/download:")
        st.download_button(
            "변경 위치 표시 PNG", _png_bytes(unified["visual"]["overview"]),
            file_name="revision.overview.png", mime="image/png",
            key="unified_overview_png", icon=":material/download:")
        if st.session_state.unified_revision_source:
            st.download_button(
                "변경후 원본 다운로드", st.session_state.unified_revision_source[0],
                file_name=st.session_state.unified_revision_source[1],
                mime="application/octet-stream", key="unified_after_original",
                icon=":material/download:")

    st.subheader("변경 영역 확대 확인")
    items_by_id = {item["id"]: item for item in unified["visual"]["items"]}
    evidence_by_id = {item["change_id"]: item for item in unified["evidence"]["candidates"]}
    for row in final_rows:
        change_id = row["번호"]
        decision = review_by_id[change_id]
        item = items_by_id[change_id]
        evidence = evidence_by_id[change_id]
        before_value = decision["before_value"] or "인식값 없음"
        after_value = decision["after_value"] or "인식값 없음"
        title = (f"{change_id} · {row['판정']} · {row['무엇이 달라졌나']} · "
                 f"{before_value} → {after_value}")
        with st.expander(title):
            crop_columns = st.columns(2)
            crop_columns[0].image(item["before_crop"], caption="변경 전", width="stretch")
            crop_columns[1].image(item["after_crop"], caption="변경 후", width="stretch")
            st.write(decision["reason"])
            st.caption(
                f"OCR 전: {[x['text'] for x in evidence['before']['ocr']]} · "
                f"OCR 후: {[x['text'] for x in evidence['after']['ocr']]} · "
                f"픽셀 +{evidence['visual'].get('added_ink_pixels') or 0} / "
                f"-{evidence['visual'].get('removed_ink_pixels') or 0}")

    with st.expander("처리 상세·전송 경계"):
        st.json({
            "local_timings": unified["timings"],
            "before_ocr_count": len(unified["before_texts"]),
            "after_ocr_count": len(unified["after_texts"]),
            "before_symbol_count": len(unified["before_structure"]["objects"]),
            "after_symbol_count": len(unified["after_structure"]["objects"]),
            "llm_received_original_pdf_or_image": False,
            "llm_received_structured_evidence": review.get("analysis_source") == "openai_llm",
            "llm_batch_count": review.get("batch_count", 0),
            "llm_passes_per_batch": review.get("passes_per_batch", 0),
        })
        for warning in report.get("warnings", []) + unified.get("warnings", []) + review.get("global_warnings", []):
            st.warning(warning)

saved_revision_root = ROOT / "output/revision_comparison"
saved_revisions = sorted(
    (path.parent for path in saved_revision_root.glob("*/report.json")),
    key=lambda path: (path.name != "blind_sample", path.name),
)
if saved_revisions:
    with st.expander("개발 진단용 저장 결과 보기"):
        saved_labels = {
            "blind_sample": "블라인드 샘플 · 빨간 정답 없음",
            "sample": "표시본 샘플 · 빨간 주석 OCR",
        }
        selected_revision = st.selectbox(
            "저장 결과", saved_revisions,
            format_func=lambda path: saved_labels.get(path.name, path.name))
        sample_report = json.loads((selected_revision / "report.json").read_text(encoding="utf-8"))
        show_revision_result(
            sample_report, str(selected_revision / "overview.png"),
            str(selected_revision / "after_aligned.png"),
            lambda item_id: (str(selected_revision / "crops" / f"{item_id}_before.png"),
                             str(selected_revision / "crops" / f"{item_id}_after.png")),
            key_prefix=f"saved_{selected_revision.name}",
        )

st.subheader("1. 로컬 입력")
with st.form("load"):
    upload = st.file_uploader("도면 또는 로컬 구조 JSON", type=["png", "jpg", "jpeg", "pdf", "json"])
    page = st.number_input("PDF 페이지 (1부터 시작)", min_value=1, value=1, step=1)
    rotation = st.number_input("수동 회전 (반시계 방향, 도)", value=0.0, min_value=-360.0, max_value=360.0)
    submitted = st.form_submit_button("업로드 파일 로컬 처리")
demo = st.button("합성 예제 불러오기", key="demo")
public_samples = sorted((ROOT / "data/public/control_circuits/images/test").glob("*.png"))
public_demo = st.button("공개 회로 시험 이미지 분석", disabled=not public_samples, key="public_demo")

if submitted or demo or public_demo or load_comparison:
    try:
        if load_comparison:
            from PIL import Image
            import numpy as np
            structure = suggest_links(json.loads((selected / "improved.json").read_text(encoding="utf-8")))
            preview = np.array(Image.open(selected / "original.png").convert("L"))
            loaded_name = report["input_name"]
        elif demo:
            structure = json.loads((ROOT / "sample_structure.json").read_text(encoding="utf-8"))
            preview, loaded_name = None, "합성 예제"
        elif public_demo:
            source = public_samples[0]
            preview, structure = process_image(source.read_bytes(), source.suffix)
            loaded_name = "공개 산업 제어회로: " + source.name
        elif upload is None:
            raise ValueError("파일을 먼저 선택하세요.")
        elif upload.size > 20 * 1024 * 1024:
            raise ValueError("20MB 이하 파일만 처리합니다.")
        elif upload.name.lower().endswith(".json"):
            structure = json.loads(upload.getvalue().decode("utf-8-sig"))
            preview, loaded_name = None, upload.name
        else:
            preview, structure = process_image(upload.getvalue(), Path(upload.name).suffix, int(page) - 1, rotation)
            loaded_name = upload.name
        analyze(structure)  # Reject malformed input before replacing the current document.
        st.session_state.update(structure=structure, preview=preview, decisions={}, suggestions={},
                                dxf=None, dwg=None, loaded_name=loaded_name,
                                revision=st.session_state.revision + 1)
    except (ValueError, TypeError, UnicodeDecodeError, ImportError, OSError) as exc:
        st.error(str(exc))

if st.session_state.structure is None:
    st.stop()

structure = st.session_state.structure
rows, rules = analyze(structure), load_rules()
revision = st.session_state.revision
st.caption(f"현재 처리 결과: {st.session_state.loaded_name}")
if st.session_state.preview is not None:
    overlay = draw_overlay(st.session_state.preview, structure["objects"], structure["texts"])
    st.image(overlay, caption="로컬 검출 검토 · 빨강: 직선 후보 / 파랑: 기호 / 초록: OCR", width=900)
for warning in structure.get("warnings", []):
    st.warning(warning)
with st.expander("전체 구조 JSON · 외부 전송하지 않음"):
    st.json(structure)
st.download_button("로컬 구조 JSON 다운로드", json.dumps(structure, ensure_ascii=False, indent=2),
                   file_name="structure.local.json", mime="application/json")

st.subheader("2. 의미와 대상 객체 검토")
st.caption("단위 규칙은 의미 후보만 만듭니다. 연결 대상과 OCR 값은 원본을 보며 이 PC에서 확인하세요.")
if rows:
    text_id = st.selectbox("검토할 문자", [r["text_id"] for r in rows],
                           format_func=lambda tid: next(f"{r['text_id']}: {r['raw_text']}" for r in rows if r["text_id"] == tid),
                           key=f"text_{revision}")
    row = next(r for r in rows if r["text_id"] == text_id)
    st.json(row)
    if row["roles"]:
        role = st.selectbox("속성 의미", row["roles"], key=f"role_{revision}_{text_id}")
        targets = [obj["id"] for obj in structure["objects"] if obj["kind"] in rules["targets"][role]]
        if targets:
            target = st.selectbox("연결할 로컬 객체", targets, key=f"target_{revision}_{text_id}_{role}")
            if st.button("값·단위·대상을 확인하고 적용", key="confirm"):
                updated = {**st.session_state.decisions, text_id: {"role": role, "object_id": target}}
                try:
                    finalize(structure, updated)
                    st.session_state.update(decisions=updated, dxf=None, dwg=None)
                    st.success("로컬 검토 결과를 적용했습니다.")
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.warning("이 의미를 연결할 수 있는 객체가 없습니다.")
        if text_id in st.session_state.decisions and st.button("이 문자 확인 취소"):
            del st.session_state.decisions[text_id]
            st.session_state.update(dxf=None, dwg=None)
    else:
        st.warning("현재 rules.json에 없는 문자입니다. OCR 값·분류 규칙을 로컬에서 보완하세요.")

    with st.expander("레거시: 최소 정보로 API에 관계 후보 문의"):
        st.caption("실제 값·문자·좌표·도면·전체 연결망은 제외합니다. 아래 범주 정보 자체도 외부로 나가므로 "
                   "회사 정책상 파생 정보도 반출 금지라면 사용하지 마세요. 결과는 자동 적용되지 않습니다.")
        try:
            payload, mapping = make_payload(row)
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.json(payload)
            enabled = st.checkbox("위 범주 JSON의 외부 전송을 허용", value=False, key=f"allow_{revision}_{text_id}")
            if st.button("표시된 후보 1건만 API에 문의", disabled=not enabled or not model_id):
                try:
                    selection = request_selection(payload, model_id, enabled=enabled, api_key=api_key)
                    st.session_state.suggestions[text_id] = {
                        "role": selection["role"], "object_id": mapping.get(selection["target"]),
                        "status": "needs_review"}
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
            if text_id in st.session_state.suggestions:
                st.json(st.session_state.suggestions[text_id])
                st.caption("참고 제안입니다. 위의 로컬 검토에서 확인해야 적용됩니다.")
else:
    st.warning("검출된 문자가 없습니다. 로컬 OCR 설정 또는 구조 JSON을 확인하세요.")

st.subheader("3. 로컬 결과 생성")
final = finalize(structure, st.session_state.decisions)
if "graph" in structure:
    final["graph"] = structure["graph"]
st.json(final)
st.download_button("의미 JSON 다운로드", json.dumps(final, ensure_ascii=False, indent=2),
                   file_name="semantic.local.json", mime="application/json")
st.caption("미확인 항목은 needs_review로 남습니다. 의미 JSON과 CAD 생성 시 API를 호출하지 않습니다.")
scale = st.number_input("검증한 축척 (mm / 처리 이미지 pixel)", min_value=0.000001, value=0.1, format="%.6f")
st.caption("기본 0.1은 합성 예제용입니다. 실제 도면은 기준 치수(mm) ÷ 그 치수의 픽셀 거리로 보정하세요. "
           "하네스 표시 길이와 화면상의 선 길이는 다를 수 있습니다.")
geometry_ok = st.checkbox("원본과 도형·축척을 로컬에서 비교했습니다", key=f"geometry_{revision}")
if st.button("검토용 DXF 생성", disabled=not geometry_ok):
    try:
        st.session_state.update(dxf=build_dxf(structure, scale, block_library=block_library or None,
                                           semantics=final), dwg=None)
        st.session_state["dxf_scale"] = scale
    except ValueError as exc:
        st.error(str(exc))
if st.session_state.dxf:
    st.caption(f"마지막 생성 축척: {st.session_state.dxf_scale:g} mm/pixel · 출력은 검토용입니다.")
    st.download_button("DXF 다운로드", st.session_state.dxf, file_name="review.dxf", mime="application/dxf")
    with st.expander("설치된 로컬 CAD로 DWG 저장"):
        st.caption("pywin32와 정식 설치된 CAD가 필요합니다. 실제 CAD 버전별 연동은 아직 검증하지 않았습니다.")
        prog_id = st.selectbox("로컬 CAD", ["AutoCAD.Application", "ZWCAD.Application"])
        if st.button("로컬 CAD를 호출해 DWG 생성"):
            try:
                with tempfile.TemporaryDirectory(prefix="local-cad-") as folder:
                    source, destination = Path(folder) / "source.dxf", Path(folder) / "result.dwg"
                    source.write_bytes(st.session_state.dxf)
                    dxf_to_dwg(source, destination, prog_id)
                    st.session_state.dwg = destination.read_bytes()
            except Exception:
                st.error("DWG 변환에 실패했습니다. CAD 설치·라이선스·COM ProgID·pywin32를 확인하세요. DXF는 유지됩니다.")
        if st.session_state.dwg:
            st.download_button("DWG 다운로드", st.session_state.dwg, file_name="result.dwg", mime="application/octet-stream")

st.subheader("4. 검토 완료 JSON의 의미·연결 비교")
st.caption("각 도면을 먼저 로컬 분석·검토한 뒤 다운로드한 의미 JSON을 넣으세요. 자동 생성된 ID는 서로 같은 객체라는 뜻이 아닙니다.")
with st.form("revision_comparison"):
    before_file = st.file_uploader("개정 전 의미 JSON", type=["json"], key="before_json")
    after_file = st.file_uploader("개정 후 의미 JSON", type=["json"], key="after_json")
    match_text = st.text_area("확인한 객체 대응표 (개정 전 ID → 개정 후 ID)", value='{"WIRE_03": "WIRE_03", "CIRCLE_01": "CIRCLE_01"}')
    match_complete = st.checkbox("모든 객체의 대응·추가·삭제 여부를 검토했습니다")
    compare_clicked = st.form_submit_button("로컬에서 변경점 비교")
if compare_clicked:
    try:
        if not before_file or not after_file:
            raise ValueError("개정 전후 JSON 두 개를 선택하세요.")
        if max(before_file.size, after_file.size) > 20 * 1024 * 1024:
            raise ValueError("각 JSON은 20MB 이하여야 합니다.")
        changes = compare_revisions(json.loads(before_file.getvalue()), json.loads(after_file.getvalue()),
                                    json.loads(match_text), matches_complete=match_complete)
        st.json(changes)
        st.download_button("변경점 JSON 다운로드", json.dumps(changes, ensure_ascii=False, indent=2),
                           file_name="changes.local.json", mime="application/json")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
