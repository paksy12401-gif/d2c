"""Run locally: python -m streamlit run pdf_to_dwg_web.py --server.port 8502"""

from pathlib import Path

import pymupdf
import streamlit as st

from pdf_to_dwg_workflow import DEFAULT_DATA_ROOT, ROOT, convert_pdf, inspect_pdf


st.set_page_config(page_title="PDF 도면 → 블록 DWG", layout="wide")
st.title("PDF 도면 → 블록 DWG")
st.caption("같은 양식의 1페이지 벡터 도면용 · 이 PC의 AutoCAD와 심볼 템플릿을 사용합니다.")
st.info("검증된 샘플 PDF는 표·문구까지 복구한 DWG를 제공합니다. 새 PDF는 심볼 블록화까지 자동 처리하며, 표·윤곽 문자는 원본 형태로 보존합니다.")

with st.sidebar:
    st.subheader("작업 환경")
    data_root = Path(st.text_input("심볼 데이터 폴더", value=str(DEFAULT_DATA_ROOT)))
    st.caption("폴더 안에 dataset_640 및 labels_640이 있어야 합니다.")
    st.caption("업로드한 PDF는 이 PC에서만 처리합니다. API 전송은 없습니다.")

upload = st.file_uploader("변환할 PDF", type=["pdf"])
if upload is not None:
    data = upload.getvalue()
    try:
        info = inspect_pdf(data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.container(border=True):
        st.write(f"**파일:** {upload.name}")
        st.write(f"벡터 경로 {info['vector_paths']:,}개 · 포함 이미지 {info['embedded_images']}개")
        with pymupdf.open(stream=data, filetype="pdf") as document:
            page = document[0]
            factor = min(1200 / page.rect.width, 650 / page.rect.height, 1)
            preview = page.get_pixmap(matrix=pymupdf.Matrix(factor, factor), alpha=False)
            st.image(preview.tobytes("png"), caption="업로드한 도면 전체 미리보기")

    if st.button("변환 시작", type="primary", icon=":material/play_arrow:"):
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
        st.subheader("결과")
        st.write(f"심볼 블록 **{report['symbol_blocks']}개** · 검출 후보 **{report['candidate_count']}개**")
        if report["mode"] == "verified_identical_pdf":
            st.success("검증된 샘플 PDF와 내용이 동일하여 복구 완료 DWG를 사용했습니다. 표와 문구 텍스트도 포함됩니다.")
        else:
            st.warning("새 도면의 심볼 블록은 자동 생성됐습니다. 배선 분리와 윤곽 글자·이미지 표의 텍스트 복구는 검증되지 않았으므로 결과 PDF와 DWG를 확인하세요.")
        st.download_button("DWG와 검출 결과 ZIP 다운로드", result["archive"].read_bytes(),
                           file_name="pdf_to_block_dwg.zip", mime="application/zip",
                           icon=":material/download:")
        st.caption(f"작업 기록: {result['folder']}")

st.divider()
st.subheader("처리 과정")
st.markdown("""
1. **입력 검사:** PDF가 한 페이지이고 벡터 선이 있는지 확인합니다.
2. **심볼 검색:** 기존 640×640 정답박스 템플릿을 PDF에서 형상 비교하고, 작은 심볼은 고해상도 구역으로 다시 찾습니다.
3. **CAD 변환:** AutoCAD가 PDF의 선과 이미지를 DWG 객체로 가져옵니다. 화면 전체를 640×640으로 축소하지 않습니다.
4. **블록화:** 강한 심볼 후보의 범위에 완전히 들어온 벡터 객체만 개별 BLOCK으로 묶고 나머지 배선은 원래 객체로 둡니다.
5. **검증:** 생성된 블록 수와 변환 전후 객체를 비교하고, 원본 PDF·결과 DWG·검출 표시 PDF·기록을 ZIP으로 제공합니다.

기존 샘플과 **파일 내용이 완전히 같은 PDF**는 앞서 검증한 최종 DWG를 재사용합니다. 이 결과에는 부자재 표, 회로 표, 심볼 아래 문구의 복구 텍스트가 포함됩니다. 같은 양식의 **새 PDF**에서는 심볼 블록화까지 자동 실행하며, 표·윤곽 문구 OCR의 정확성은 아직 일반화 검증이 되지 않아 원본 형태로 보존합니다.
""")
st.download_button("전체 처리 과정 문서 다운로드", (ROOT / "PDF_TO_DWG_PROCESS.md").read_bytes(),
                   file_name="PDF_TO_DWG_PROCESS.md", mime="text/markdown")
