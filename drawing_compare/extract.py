from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf.disassemble import make_primitive, recursive_decompose

from .models import Drawing, Segment, TextItem


def _safe_handle(entity: object) -> str:
    try:
        return str(entity.dxf.handle or "")
    except Exception:
        return ""


def load_dxf(path: Path, flattening: float = 0.1) -> Drawing:
    document = ezdxf.readfile(path)
    modelspace = document.modelspace()
    segments: list[Segment] = []
    texts: list[TextItem] = []

    for entity in recursive_decompose(modelspace):
        entity_type = entity.dxftype()
        handle = _safe_handle(entity)
        if entity_type in {"TEXT", "ATTRIB", "ATTDEF"}:
            try:
                insert = entity.dxf.insert
                texts.append(
                    TextItem(
                        value=str(entity.dxf.text),
                        x=float(insert.x),
                        y=float(insert.y),
                        height=float(entity.dxf.height),
                        handle=handle,
                    )
                )
            except Exception:
                pass
            continue
        if entity_type == "MTEXT":
            try:
                insert = entity.dxf.insert
                texts.append(
                    TextItem(
                        value=entity.plain_text(),
                        x=float(insert.x),
                        y=float(insert.y),
                        height=float(entity.dxf.char_height),
                        handle=handle,
                    )
                )
            except Exception:
                pass
            continue

        try:
            primitive = make_primitive(entity, max_flattening_distance=flattening)
            vertices = list(primitive.vertices())
        except Exception:
            continue
        if len(vertices) < 2:
            continue
        for first, second in zip(vertices, vertices[1:]):
            segment = Segment(
                float(first.x),
                float(first.y),
                float(second.x),
                float(second.y),
                entity_type,
                handle,
            )
            if segment.length > 1e-9:
                segments.append(segment)

    if not segments:
        raise ValueError(f"비교 가능한 벡터를 찾지 못했습니다: {path}")
    return Drawing(str(path), segments, texts)
