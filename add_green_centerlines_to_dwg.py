"""Draw the traced green and red centerlines inside a copy of the checked DWG."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import pymupdf

import blockify_pdf_symbols as cad


ROOT = Path(__file__).resolve().parent
TRACE = ROOT / "green_centerline_trace"
SOURCE_DWG = ROOT / "verified_dwg" / "20261016" / "symbols_60_bom_title_fit.dwg"
SOURCE_AUDIT = ROOT / "verified_dwg" / "20261016" / "entities_after.csv"
SOURCE_PDF = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
OUTPUT_DWG = TRACE / "symbols_60_with_full_centerlines.dwg"
CONFIRMED_LAYER = "GREEN_CENTERLINE_CONFIRMED"
INFERRED_LAYER = "GREEN_CENTERLINE_INFERRED"


def endpoints(row: dict, page_height: float) -> tuple[tuple[float, float], tuple[float, float]]:
    position, start, end = (float(row[key]) for key in
                            ("position_pdf_pt", "start_pdf_pt", "end_pdf_pt"))
    if row["axis"] == "H":
        pdf_points = ((start, position), (end, position))
    elif row["axis"] == "V":
        pdf_points = ((position, start), (position, end))
    else:
        raise ValueError(row["axis"])
    return tuple((x / 72.0, (page_height - y) / 72.0) for x, y in pdf_points)


def command_script(groups: list[tuple[str, tuple[tuple[float, float], tuple[float, float]]]]) -> Path:
    lines = [
        '(setvar "CMDECHO" 0)', '(setvar "FILEDIA" 0)',
        f'(if (not (tblsearch "LAYER" "{CONFIRMED_LAYER}")) '
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{CONFIRMED_LAYER}") '
        '(cons 70 0) (cons 62 4) (cons 6 "Continuous"))))',
        f'(if (not (tblsearch "LAYER" "{INFERRED_LAYER}")) '
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{INFERRED_LAYER}") '
        '(cons 70 0) (cons 62 30) (cons 6 "Continuous"))))',
        '(setq trace_fp (open (strcat (getvar "DWGPREFIX") "full_centerline_creation.csv") "w"))',
        '(write-line "layer,handle" trace_fp)',
    ]
    for layer, (a, b) in groups:
        lines += [
            '(setq trace_ent (entmakex (list (cons 0 "LWPOLYLINE") '
            '(cons 100 "AcDbEntity") '
            f'(cons 8 "{layer}") (cons 100 "AcDbPolyline") '
            '(cons 90 2) (cons 70 0) '
            f'(cons 10 (list {a[0]:.9f} {a[1]:.9f})) '
            f'(cons 10 (list {b[0]:.9f} {b[1]:.9f})))))',
            f'(write-line (strcat "{layer}," '
            '(if trace_ent (cdr (assoc 5 (entget trace_ent))) "FAILED")) trace_fp)',
        ]
    lines += ['(close trace_fp)', '(command "_.QSAVE")', '(princ)', '_.QUIT', '_N']
    path = TRACE / "add_full_centerlines.scr"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def main() -> Path:
    for path in (SOURCE_DWG, SOURCE_AUDIT, SOURCE_PDF, TRACE / "trace_report.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_DWG.exists():
        raise FileExistsError(OUTPUT_DWG)
    report = json.loads((TRACE / "trace_report.json").read_text(encoding="utf-8"))
    with pymupdf.open(SOURCE_PDF) as doc:
        height = doc[0].rect.height
    groups = [(CONFIRMED_LAYER, endpoints(row, height)) for row in report["confirmed"]]
    groups += [(INFERRED_LAYER, endpoints(row, height)) for row in report["inferred"]]
    if not report["confirmed"] or report.get("red_centerline_pdf_paths", 0) != 4:
        raise ValueError("Expected vector centerlines were not extracted")
    TRACE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DWG, OUTPUT_DWG)
    script = command_script(groups)
    cad.run_console(OUTPUT_DWG, script, TRACE / "add_full_centerlines.log")
    with (TRACE / "full_centerline_creation.csv").open(encoding="utf-8", newline="") as stream:
        created = list(csv.DictReader(stream))
    if len(created) != len(groups) or any(row["handle"] == "FAILED" for row in created):
        raise RuntimeError("AutoCAD failed to create every centerline")
    after = cad.read_entities(cad.audit(OUTPUT_DWG, TRACE, "full_centerlines_after"))
    original = cad.read_entities(SOURCE_AUDIT)
    original_by_handle = {row["handle"]: row for row in original}
    after_by_handle = {row["handle"]: row for row in after}
    if not original_by_handle.keys() <= after_by_handle.keys():
        raise RuntimeError("Some existing DWG entities disappeared")
    new_handles = after_by_handle.keys() - original_by_handle.keys()
    if new_handles != {row["handle"] for row in created}:
        raise RuntimeError("New DWG entities differ from creation report")
    new_rows = [after_by_handle[handle] for handle in new_handles]
    by_layer = Counter(row["layer"] for row in new_rows)
    expected_layers = Counter({CONFIRMED_LAYER: len(report["confirmed"]),
                               INFERRED_LAYER: len(report["inferred"])})
    if by_layer != expected_layers:
        raise RuntimeError(f"Unexpected centerline layer counts: {by_layer}")
    for (layer, (first, second)), result in zip(groups, created):
        entity = after_by_handle[result["handle"]]
        expected = (min(first[0], second[0]), min(first[1], second[1]),
                    max(first[0], second[0]), max(first[1], second[1]))
        if entity["type"] != "LWPOLYLINE" or entity["layer"] != layer or any(
                abs(a - b) > 1e-5 for a, b in zip(entity["bounds"], expected)):
            raise RuntimeError(f"Saved geometry mismatch: {result['handle']}")
    block_count = sum(row["type"] == "INSERT" and row["block_name"].startswith("SYM_")
                      for row in after)
    if block_count != 60:
        raise RuntimeError(f"Existing symbol block count changed: {block_count}")
    verification = {"output_dwg": str(OUTPUT_DWG), "existing_symbol_blocks": block_count,
                    "original_entities_preserved": len(original_by_handle),
                    "confirmed_centerlines": by_layer[CONFIRMED_LAYER],
                    "inferred_bridges": by_layer[INFERRED_LAYER]}
    (TRACE / "full_dwg_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return OUTPUT_DWG


if __name__ == "__main__":
    main()
