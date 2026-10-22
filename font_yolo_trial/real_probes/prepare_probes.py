"""Render held-out, qualitative font probes from the existing source PDF.

This only reads source PDFs and the prior verified text-restoration script.
It does not produce character annotations or training inputs.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
REFERENCE = ROOT / "complete_text_restoration.py"
SCALE = 4.0
PROBES = [
    ("splice_point", (2675, 1575, 2825, 1635), ["SPLICE POINT"]),
    ("caption_11", (1958, 1475, 2200, 1547), ["FLEECE TAPE"]),
    ("middle_callout", (3350, 1310, 3730, 1420), ["FLEECE TAPE", "COLDMELT"]),
]
VISUAL_REVIEW = {
    "splice_point": "Angular, single-line uppercase glyphs visually resemble the supplied A-Z strip. Glyphs are tightly set and some edges touch; leader lines lie directly below the word and a circled numeral is retained.",
    "caption_11": "Angular, single-line uppercase glyphs visually resemble the supplied A-Z strip. English letters have very tight spacing; Korean text and a numeral remain in the crop, with a small leader fragment at the right edge.",
    "middle_callout": "Angular, single-line uppercase glyphs visually resemble the supplied A-Z strip. Two text rows include Korean characters, and adjacent colored drawing lines, arrows and large numerals remain. Latin glyphs can touch, making segmentation harder than the spaced reference alphabet.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Read constants only: importing the CAD script would have unnecessary side effects.
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    regions = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REGIONS"
            for target in node.targets
        ):
            regions = ast.literal_eval(node.value)
            break
    if regions is None:
        raise RuntimeError("No verified regions found in reference script")
    region_by_name = {region["name"]: region for region in regions}
    source_inventory = []
    for path in sorted((ROOT / "source_inputs").glob("*.pdf")):
        with fitz.open(path) as pdf:
            source_inventory.append({
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "page_sizes_pdf_points": [[page.rect.width, page.rect.height] for page in pdf],
            })
    manifest = {
        "purpose": "Held-out real-drawing qualitative probes for the A-Z font trial",
        "usage": "Inference/visual inspection only; do not include in train or validation datasets",
        "limitation": "No per-character labels; not a labeled full-drawing detection metric",
        "coordinate_system": "PDF points, top-left origin, x rightwards, y downwards",
        "source_pdf": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "reference_script": str(REFERENCE.resolve()),
        "reference_script_sha256": sha256(REFERENCE),
        "source_inventory": source_inventory,
        "probes": [],
    }
    with fitz.open(SOURCE) as pdf:
        page = pdf[0]
        for name, rect, expected in PROBES:
            region = region_by_name[name]
            output = OUT / f"{name}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), clip=fitz.Rect(rect), alpha=False)
            pixmap.save(output)
            manifest["probes"].append({
                "id": name,
                "image": output.name,
                "image_sha256": sha256(output),
                "source_path": str(SOURCE.resolve()),
                "source_sha256": manifest["source_sha256"],
                "page_index": 0,
                "source_page_size_pdf_points": [page.rect.width, page.rect.height],
                "crop_pdf_rect": list(rect),
                "render_scale_pixels_per_pdf_point": SCALE,
                "render_dpi": SCALE * 72,
                "image_size_pixels": [pixmap.width, pixmap.height],
                "expected_english_portions": expected,
                "known_source_texts": [line[2] for line in region["lines"]],
                "reference_region_name": name,
                "reference_text_region_pdf_rect": list(region["box"]),
                "preserved_context": "Unaltered PDF crop, including surrounding linework and non-Latin characters",
                "evaluation_note": "Qualitative only; expected text is from a prior visually checked source script, not new character annotations",
                "visual_inspection": VISUAL_REVIEW[name],
            })
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "probes": [p["image"] for p in manifest["probes"]]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
