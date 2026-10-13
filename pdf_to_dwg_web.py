"""Run locally: python -m streamlit run pdf_to_dwg_web.py --server.port 8502"""

import os
from pathlib import Path

import pymupdf
import streamlit as st

from drawing_comparison_workflow import compare_pdfs, inspect_compare_pdf
from pdf_to_dwg_workflow import DEFAULT_DATA_ROOT, ROOT, convert_pdf, inspect_pdf


def preview_pdf(data: bytes, caption: str, *, max_width: int = 1000, max_height: int = 560) -> None:
    with pymupdf.open(stream=data, filetype="pdf") as document:
        page = document[0]
        factor = min(max_width / page.rect.width, max_height / page.rect.height, 1)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(factor, factor), alpha=False)
        st.image(pixmap.tobytes("png"), caption=caption)


def render_conversion() -> None:
    st.header("과제 1 · PDF → 블록 DWG")
    st.caption("같은 양식의 1페이지 벡터 도면용 · 이 PC의 AutoCAD와 심볼 템플릿을 사용합니다.")
    st.info("검증된 샘플 PDF는 표·문구까지 복구한 DWG를 제공합니다. 새 PDF는 심볼 블록화까지 자동 처리하며, 표·윤곽 문자는 원본 형태로 보존합니다.")

    with st.sidebar:
        st.subheader("과제 1 작업 환경")
        data_root = Path(st.text_input("심볼 데이터 폴더", value=str(DEFAULT_DATA_ROOT)))
        st.caption("폴더 안에 dataset_640 및 labels_640이 있어야 합니다.")
        st.caption("업로드한 PDF는 이 PC에서 처리합니다.")

    upload = st.file_uploader("변환할 PDF", type=["pdf"], key="conversion_pdf")
    if upload is not None:
        data = upload.getvalue()
        try:
            info = inspect_pdf(data)
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.container(border=True):
            st.write(f"**파일:** {upload.name}")
            st.write(f"벡터 경로 {info['vector_paths']:,}개 · 포함 이미지 {info['embedded_images']}개")
            preview_pdf(data, "업로드한 도면 전체 미리보기", max_width=1200, max_height=650)

        if st.button("DWG 변환 시작", type="primary", icon=":material/play_arrow:"):
            with st.status("도면 처리 중", expanded=True) as status:
                try:
                    result = convert_pdf(data, data_root=data_root, progress=st.write)
                except Exception as exc:
                    status.update(label="변환 실패", state="error")
                    st.error(str(exc))
                else:
                    st.session_state["pdf_dwg_result"] = result
                    st.session_state["pdf_dwg_hash"] = info["sha256"]
                    status.update(label="변환 완료", state="complete")

        result = st.session_state.get("pdf_dwg_result")
        if result and st.session_state.get("pdf_dwg_hash") == info["sha256"]:
            report = result["report"]
            st.subheader("변환 결과")
            st.write(f"심볼 블록 **{report['symbol_blocks']}개** · 검출 후보 **{report['candidate_count']}개**")
            if report["mode"] == "verified_identical_pdf":
                st.success("검증된 샘플 PDF와 내용이 동일하여 복구 완료 DWG를 사용했습니다. 표와 문구 텍스트도 포함됩니다.")
            else:
                st.warning("새 도면의 심볼 블록은 자동 생성됐습니다. 배선 분리와 윤곽 글자·이미지 표의 텍스트 복구는 검증되지 않았으므로 결과 PDF와 DWG를 확인하세요.")
            st.download_button("DWG와 검출 결과 ZIP 다운로드", result["archive"].read_bytes(),
                               file_name="pdf_to_block_dwg.zip", mime="application/zip",
                               icon=":material/download:")
            st.caption(f"작업 기록: {result['folder']}")

    with st.expander("과제 1 처리 과정"):
        st.markdown("""
1. **입력 검사:** PDF가 한 페이지이고 벡터 선이 있는지 확인합니다.
2. **심볼 검색:** 기존 640×640 정답박스 템플릿을 PDF에서 형상 비교하고, 작은 심볼은 고해상도 구역으로 다시 찾습니다.
3. **CAD 변환:** AutoCAD가 PDF의 선과 이미지를 DWG 객체로 가져옵니다. 화면 전체를 640×640으로 축소하지 않습니다.
4. **블록화:** 강한 심볼 후보의 범위에 완전히 들어온 벡터 객체만 개별 BLOCK으로 묶고 나머지 배선은 원래 객체로 둡니다.
5. **검증:** 생성된 블록 수와 변환 전후 객체를 비교하고, 원본 PDF·결과 DWG·검출 표시 PDF·기록을 ZIP으로 제공합니다.

기존 샘플과 **파일 내용이 완전히 같은 PDF**는 앞서 검증한 최종 DWG를 재사용합니다. 같은 양식의 **새 PDF**에서는 심볼 블록화까지 자동 실행하며, 표·윤곽 문구 OCR의 정확성은 아직 일반화 검증이 되지 않아 원본 형태로 보존합니다.
""")
        st.download_button("전체 처리 과정 문서 다운로드", (ROOT / "PDF_TO_DWG_PROCESS.md").read_bytes(),
                           file_name="PDF_TO_DWG_PROCESS.md", mime="text/markdown")


def render_comparison() -> None:
    st.header("과제 2 · 기존/신규 도면 비교")
    st.caption("두 벡터 PDF의 선을 정렬해 비교합니다. 빨강은 기존 도면 전용, 초록은 신규 도면 전용, 회색은 공통 선입니다.")
    old_col, new_col = st.columns(2)
    with old_col:
        old_upload = st.file_uploader("기존 도면 PDF", type=["pdf"], key="compare_old")
    with new_col:
        new_upload = st.file_uploader("신규 도면 PDF", type=["pdf"], key="compare_new")

    if old_upload is None or new_upload is None:
        st.info("기존 도면과 신규 도면 PDF를 각각 올려 주세요.")
        return

    old_data, new_data = old_upload.getvalue(), new_upload.getvalue()
    try:
        old_info, new_info = inspect_compare_pdf(old_data), inspect_compare_pdf(new_data)
        if old_info["pages"] != new_info["pages"]:
            raise ValueError("기존 도면과 신규 도면의 페이지 수가 같아야 합니다.")
    except ValueError as exc:
        st.error(str(exc))
        return

    with old_col:
        st.caption(f"{old_info['pages']}페이지 · 벡터 경로 {old_info['vector_paths']:,}개")
        preview_pdf(old_data, "기존 도면 첫 페이지", max_width=600, max_height=400)
    with new_col:
        st.caption(f"{new_info['pages']}페이지 · 벡터 경로 {new_info['vector_paths']:,}개")
        preview_pdf(new_data, "신규 도면 첫 페이지", max_width=600, max_height=400)

    llm_ready = all(os.getenv(name) for name in (
        "DRAWING_COMPARE_LLM_URL", "DRAWING_COMPARE_LLM_API_KEY", "DRAWING_COMPARE_LLM_MODEL"
    ))
    use_llm = st.checkbox("변경 영역을 LLM으로 설명", value=False, disabled=not llm_ready)
    if llm_ready:
        st.caption("선택하면 변경 영역 이미지와 측정값이 설정된 외부 LLM API로 전송됩니다. API 호출이 실패하면 로컬 설명으로 대체됩니다.")
    else:
        st.caption("현재는 로컬 형상 분석으로 설명합니다. 외부 LLM 설정값은 없습니다.")

    if st.button("두 도면 비교 시작", type="primary", icon=":material/compare_arrows:"):
        with st.status("벡터 도면 비교 중", expanded=True) as status:
            try:
                result = compare_pdfs(old_data, new_data, use_llm=use_llm)
            except Exception as exc:
                status.update(label="비교 실패", state="error")
                st.error(str(exc))
            else:
                st.session_state["comparison_result"] = result
                st.session_state["comparison_hashes"] = (old_info["sha256"], new_info["sha256"], use_llm)
                status.update(label="비교 완료", state="complete")

    result = st.session_state.get("comparison_result")
    if not result or st.session_state.get("comparison_hashes") != (
        old_info["sha256"], new_info["sha256"], use_llm
    ):
        return

    comparison = result["result"]
    if not comparison.overlay_path.is_file() or not result["archive"].is_file():
        st.error("비교 결과 이미지가 없습니다. '두 도면 비교 시작'을 다시 눌러 결과를 생성해 주세요.")
        return
    st.subheader("비교 결과")
    st.write(f"변경 영역 **{len(comparison.regions)}개** · 미세 이동 제외 **{comparison.suppressed_minor_regions}개**")
    st.caption("도면 전체 정렬 후의 좌표와 형상 차이를 비교한 결과입니다. 실제 변경 여부는 결과 이미지를 확인해 주세요.")
    st.image(comparison.overlay_path.read_bytes(), caption="전체 오버레이 · 빨강 삭제 / 초록 추가 / 회색 공통")
    for annotated in comparison.annotated_pages:
        st.image((comparison.output_dir / annotated).read_bytes(), caption=f"변경 위치 · {annotated}")
    if comparison.regions:
        st.dataframe([{
            "ID": region.id,
            "분류": region.classification,
            "설명": region.description,
            "추가 픽셀": region.added_pixels,
            "삭제 픽셀": region.deleted_pixels,
            "신뢰도": region.confidence,
        } for region in comparison.regions], hide_index=True)
    st.download_button("비교 보고서 ZIP 다운로드", result["archive"].read_bytes(),
                       file_name="drawing_comparison.zip", mime="application/zip",
                       icon=":material/download:")
    st.caption("ZIP을 풀고 report.html을 열면 영역별 이미지와 상세 설명을 볼 수 있습니다.")
    st.caption(f"작업 기록: {result['folder']}")


st.set_page_config(page_title="CAD 도면 작업실", layout="wide")
st.title("CAD 도면 작업실")
st.caption("과제 1 PDF→DWG 블록화 · 과제 2 기존/신규 PDF 변경 비교")
conversion_tab, comparison_tab = st.tabs(["과제 1 · PDF→DWG", "과제 2 · 도면 비교"])
with conversion_tab:
    render_conversion()
with comparison_tab:
    render_comparison()
