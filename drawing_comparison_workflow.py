"""Run assignment 2's vector-PDF comparison from the combined local web app."""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import uuid
import zipfile

import pymupdf

from drawing_compare.compare import compare_files


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "drawing_comparison"
MAX_PDF_BYTES = 40 * 1024 * 1024


def inspect_compare_pdf(data: bytes) -> dict:
    if not data or len(data) > MAX_PDF_BYTES:
        raise ValueError("각 PDF는 1바이트 이상, 40MB 이하여야 합니다.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass or not document.page_count:
                raise ValueError("암호가 없고 페이지가 있는 PDF만 비교할 수 있습니다.")
            vector_paths = sum(len(page.get_drawings()) for page in document)
            if not vector_paths:
                raise ValueError("벡터 선이 없는 PDF입니다. 이미지 스캔 도면은 비교할 수 없습니다.")
            return {
                "sha256": hashlib.sha256(data).hexdigest(),
                "pages": document.page_count,
                "vector_paths": vector_paths,
            }
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError("올바른 PDF 파일을 선택해 주세요.") from exc


def compare_pdfs(
    old_data: bytes,
    new_data: bytes,
    *,
    use_llm: bool = False,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    if inspect_compare_pdf(old_data)["pages"] != inspect_compare_pdf(new_data)["pages"]:
        raise ValueError("기존 도면과 신규 도면의 페이지 수가 같아야 합니다.")

    run_dir = output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    old_path, new_path = run_dir / "old.pdf", run_dir / "new.pdf"
    old_path.write_bytes(old_data)
    new_path.write_bytes(new_data)
    result_dir = run_dir / "result"
    result = compare_files(old_path, new_path, result_dir, use_llm=use_llm)

    archive = run_dir / "comparison_result.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(result_dir.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(result_dir))
    return {
        "folder": run_dir,
        "result": result,
        "archive": archive,
    }
