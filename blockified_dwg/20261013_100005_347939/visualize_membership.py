"""Draw source DWG membership for manual symbol/pipe separation review."""

import csv
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


FOLDER = Path(__file__).resolve().parent
GROUPS = json.loads((FOLDER / "block_manifest.json").read_text("utf-8"))
AUDIT = {r["handle"]: r for r in csv.DictReader((FOLDER / "entities_before.csv").open())}
HANDLE = re.compile(r'\(5 \. "([0-9A-F]+)"\)')
TYPE = re.compile(r'\(0 \. "([A-Z]+)"\)')
VERTEX = re.compile(r'\(10 ([+-]?[\d.eE]+) ([+-]?[\d.eE]+)(?: [+-]?[\d.eE]+)?\)')
AXIS = re.compile(r'\(11 ([+-]?[\d.eE]+) ([+-]?[\d.eE]+)(?: [+-]?[\d.eE]+)?\)')


def code(line, number, default=0.0):
    found = re.search(rf'\({number} \. ([+-]?[\d.eE]+)\)', line)
    return float(found.group(1)) if found else default


def bulged_segment(a, b, bulge):
    if abs(bulge) < 1e-6:
        return [a, b]
    dx, dy = b[0] - a[0], b[1] - a[1]
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        return [a, b]
    offset = chord * (1 - bulge * bulge) / (4 * bulge)
    center = ((a[0] + b[0]) / 2 - dy / chord * offset,
              (a[1] + b[1]) / 2 + dx / chord * offset)
    start = math.atan2(a[1] - center[1], a[0] - center[0])
    sweep = 4 * math.atan(bulge)
    radius = math.hypot(a[0] - center[0], a[1] - center[1])
    steps = max(5, int(abs(sweep) * 12))
    return [(center[0] + radius * math.cos(start + sweep * i / steps),
             center[1] + radius * math.sin(start + sweep * i / steps))
            for i in range(steps + 1)]


def points(line, kind):
    vertices = [(float(x), float(y)) for x, y in VERTEX.findall(line)]
    if not vertices:
        return []
    if kind == "LWPOLYLINE":
        bulges = [float(value) for value in re.findall(r'\(42 \. ([+-]?[\d.eE]+)\)', line)]
        is_closed = int(code(line, 70)) & 1
        pairs = len(vertices) if is_closed else len(vertices) - 1
        path = []
        for i in range(pairs):
            path.extend(bulged_segment(vertices[i], vertices[(i + 1) % len(vertices)],
                                       bulges[i] if i < len(bulges) else 0.0)[1:] if path
                        else bulged_segment(vertices[i], vertices[(i + 1) % len(vertices)],
                                            bulges[i] if i < len(bulges) else 0.0))
        return path or vertices
    if kind == "CIRCLE" or kind == "ARC":
        cx, cy = vertices[0]
        radius = code(line, 40)
        start = code(line, 50) if kind == "ARC" else 0
        end = code(line, 51) if kind == "ARC" else 2 * math.pi
        if end < start:
            end += 2 * math.pi
        return [(cx + radius * math.cos(start + (end - start) * i / 80),
                 cy + radius * math.sin(start + (end - start) * i / 80))
                for i in range(81)]
    if kind == "ELLIPSE":
        axis = AXIS.search(line)
        if not axis:
            return []
        ax, ay = float(axis.group(1)), float(axis.group(2))
        cx, cy = vertices[0]
        ratio = code(line, 40)
        start, end = code(line, 41), code(line, 42, 2 * math.pi)
        if end < start:
            end += 2 * math.pi
        return [(cx + ax * math.cos(start + (end - start) * i / 80)
                 - ay * ratio * math.sin(start + (end - start) * i / 80),
                 cy + ay * math.cos(start + (end - start) * i / 80)
                 + ax * ratio * math.sin(start + (end - start) * i / 80))
                for i in range(81)]
    return []


geometry = {}
with (FOLDER / "entity_geometry.lspdata").open(encoding="cp949", errors="ignore") as stream:
    for line in stream:
        h, kind = HANDLE.search(line), TYPE.search(line)
        if h and kind and kind.group(1) in {"LWPOLYLINE", "CIRCLE", "ARC", "ELLIPSE"}:
            geometry[h.group(1)] = points(line, kind.group(1))

out_dir = FOLDER / "membership_review"
out_dir.mkdir(exist_ok=True)
font = ImageFont.load_default()
tiles = []
for group in GROUPS:
    box = group["bbox_dwg"]
    width, height = box[2] - box[0], box[3] - box[1]
    margin = max(0.12, 0.2 * max(width, height))
    x1, y1, x2, y2 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
    target_w, target_h = 500, 420
    scale = min((target_w - 30) / (x2 - x1), (target_h - 35) / (y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    to_pixel = lambda x, y: (round(target_w / 2 + (x - cx) * scale),
                              round(target_h / 2 - (y - cy) * scale))
    selected = set(group["handles"])
    excluded = set(group["left_as_pipe"])
    image = Image.new("RGB", (target_w, target_h), "white")
    draw = ImageDraw.Draw(image)
    for color, stroke, chosen in [("#cbd1d7", 1, "context"),
                                  ("#164476", 2, "selected"),
                                  ("#e65522", 3, "excluded")]:
        for h, row in AUDIT.items():
            bounds = [float(row[k]) for k in ("minx", "miny", "maxx", "maxy")]
            if bounds[2] < x1 or bounds[0] > x2 or bounds[3] < y1 or bounds[1] > y2:
                continue
            if (chosen == "selected" and h not in selected) or (chosen == "excluded" and h not in excluded):
                continue
            if chosen == "context" and (h in selected or h in excluded):
                continue
            path = geometry.get(h, [])
            if len(path) >= 2:
                draw.line([to_pixel(*point) for point in path], fill=color, width=stroke,
                          joint="curve")
    draw.text((6, 4), group["block_name"], fill="black", font=font)
    draw.text((6, 20), "blue=block  orange=pipe left out  gray=context", fill="black", font=font)
    image.save(out_dir / f"{group['block_name']}.png")
    tiles.append((group["block_name"], image))

for chunk_start in range(0, len(tiles), 12):
    chunk = tiles[chunk_start:chunk_start + 12]
    sheet = Image.new("RGB", (500 * 4, 420 * 3), "#dddddd")
    for i, (_, tile) in enumerate(chunk):
        sheet.paste(tile, ((i % 4) * 500, (i // 4) * 420))
    sheet.save(out_dir / f"sheet_{chunk_start // 12 + 1}.png")
print(f"Rendered {len(tiles)} blocks; parsed {len(geometry)} source entities")
