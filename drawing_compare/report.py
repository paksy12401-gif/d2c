from __future__ import annotations

import html
from pathlib import Path

from .models import CompareResult


def _evidence_html(evidence: dict[str, object]) -> str:
    items = [
        f"범위: {float(evidence.get('width', 0)):.1f} x {float(evidence.get('height', 0)):.1f}",
        f"추가/삭제 선 길이: {float(evidence.get('added_length', 0)):.1f} / {float(evidence.get('deleted_length', 0)):.1f}",
        f"추가/삭제 묶음: {int(evidence.get('added_components', 0))} / {int(evidence.get('deleted_components', 0))}",
        f"추가/삭제 폐곡선: {int(evidence.get('added_closed_contours', 0))} / {int(evidence.get('deleted_closed_contours', 0))}",
    ]
    shift = evidence.get("estimated_local_shift")
    if isinstance(shift, dict):
        items.append(
            f"추정 이동: X {float(shift.get('dx', 0)):+.2f}, Y {float(shift.get('dy', 0)):+.2f}"
        )
    residual = evidence.get("aligned_residual_ratio")
    if residual is not None:
        items.append(f"이동 보정 후 잔차: {float(residual):.1%}")
    return "<br>".join(html.escape(item) for item in items)


def write_report(result: CompareResult) -> Path:
    rows = []
    for region in result.regions:
        bbox = ", ".join(f"{value:.2f}" for value in region.bbox_drawing)
        rows.append(
            "<tr>"
            f"<td>{html.escape(region.id)}</td>"
            f'<td><img src="{html.escape(region.image or "")}" loading="lazy"></td>'
            f"<td>{html.escape(region.classification)}</td>"
            f"<td>{html.escape(region.description)}</td>"
            f"<td>{_evidence_html(region.evidence)}</td>"
            f"<td>{bbox}</td>"
            f"<td>{region.deleted_pixels:,}</td>"
            f"<td>{region.added_pixels:,}</td>"
            f"<td>{region.confidence:.0%}</td>"
            "</tr>"
        )
    empty = '<tr><td colspan="9" class="empty">중요 변경 영역이 검출되지 않았습니다.</td></tr>'
    annotated = "".join(
        f'<figure><img class="annotated" src="{html.escape(path)}" alt="변경 주석 도면"><figcaption>중요 변경 위치 - {index}페이지</figcaption></figure>'
        for index, path in enumerate(result.annotated_pages, 1)
    )
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>도면 비교 결과</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, Pretendard, "Malgun Gothic", sans-serif; }}
body {{ margin:0; background:#101820; color:#e9eef2; }}
main {{ max-width:1500px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 8px; font-size:28px; }}
.meta {{ color:#aeb9c3; margin-bottom:22px; }}
.legend {{ display:flex; gap:18px; margin:12px 0; }}
.dot {{ width:11px; height:11px; display:inline-block; border-radius:50%; margin-right:6px; }}
.overlay,.annotated {{ width:100%; max-height:82vh; object-fit:contain; border:1px solid #31404c; background:#fff; }}
figure {{ margin:16px 0 28px; }} figcaption {{ color:#aeb9c3; margin-top:7px; }}
.summary {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:20px 0; }}
.card {{ background:#19232c; padding:16px; border-radius:10px; }}
.card strong {{ display:block; font-size:24px; }}
table {{ width:100%; border-collapse:collapse; background:#19232c; }}
th,td {{ padding:10px; border-bottom:1px solid #31404c; text-align:left; vertical-align:top; }}
th {{ position:sticky; top:0; background:#24313b; }}
td img {{ width:180px; max-height:120px; object-fit:contain; background:#111820; }}
.empty {{ text-align:center; padding:35px; color:#aeb9c3; }}
@media(max-width:800px) {{ .summary {{ grid-template-columns:1fr; }} table {{ font-size:12px; }} }}
</style>
</head>
<body><main>
<h1>벡터 PDF 자동 비교 결과</h1>
<div class="meta">기존: {html.escape(result.old_file)}<br>신규: {html.escape(result.new_file)}</div>
<h2>중요 변경 주석 도면</h2>
<div class="legend">
  <span><i class="dot" style="background:#189c56"></i>추가</span>
  <span><i class="dot" style="background:#dc2a30"></i>삭제</span>
  <span><i class="dot" style="background:#2563eb"></i>이동</span>
  <span><i class="dot" style="background:#ed7e1c"></i>이동+수정</span>
  <span><i class="dot" style="background:#8b45e0"></i>형상·문자·치수 변경</span>
</div>
{annotated if annotated else '<p class="meta">표시할 중요 변경 영역이 없습니다.</p>'}
<h2>전체 벡터 비교 오버레이</h2>
<div class="legend">
  <span><i class="dot" style="background:#eb1e1e"></i>삭제/기존 전용</span>
  <span><i class="dot" style="background:#28cd28"></i>추가/신규 전용</span>
  <span><i class="dot" style="background:#9b9b9b"></i>공통</span>
</div>
<img class="overlay" src="overlay.png" alt="도면 비교 오버레이">
<section class="summary">
  <div class="card"><span>중요 변경</span><strong>{len(result.regions)}</strong></div>
  <div class="card"><span>제외한 미세 이동</span><strong>{result.suppressed_minor_regions}</strong></div>
  <div class="card"><span>전체 정렬 이동</span><strong>{result.global_shift_drawing[0]:.3f}, {result.global_shift_drawing[1]:.3f}</strong></div>
</section>
<table>
<thead><tr><th>ID</th><th>미리보기</th><th>분류</th><th>상세 설명</th><th>판단 근거</th><th>도면 좌표</th><th>삭제</th><th>추가</th><th>신뢰도</th></tr></thead>
<tbody>{''.join(rows) if rows else empty}</tbody>
</table>
</main></body></html>"""
    path = result.output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
