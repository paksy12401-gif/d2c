"""Local CAD export. This module has no API/network calls.

Geometry is authoritative; inferred semantic attributes never change its scale.
Coordinates use image pixels (Y down), output uses millimetres (Y up).
"""

import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import ezdxf
from ezdxf.addons import Importer


LAYERS = {
    "wire": "WIRES", "line": "GEOMETRY", "circle": "GEOMETRY",
    "arc": "GEOMETRY", "spline": "GEOMETRY", "polyline": "GEOMETRY",
    "connector": "CONNECTORS", "terminal": "TERMINALS", "ground": "GROUND",
    "splice": "SPLICES", "symbol": "SYMBOLS",
}
SYMBOLS = {"connector", "terminal", "ground", "splice", "symbol"}
XDATA_APP = "HARNESS_AI"


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: 숫자가 필요합니다.")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name}: 지원 가능한 숫자 범위를 초과했습니다.") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: 유한한 {'양수' if positive else '숫자'}가 필요합니다.")
    return value


def build_dxf(structure: dict, mm_per_pixel: float, block_library=None, semantics=None) -> bytes:
    """Build UTF-8 R2010 DXF with LINE/CIRCLE/ARC/SPLINE/TEXT entities.

    ARC input angles increase clockwise in image coordinates; reflection swaps
    the start/end angles. SPLINE points are interpolation fit points.
    block_library is an approved local DXF with millimetre units. Symbol geometry
    needs insert and block_name; scale defaults to 1, angle is clockwise degrees.
    Library blocks are already CAD Y-up: their geometry is not mirrored.
    semantics accepts pipeline.finalize() output or {object_id: attributes}.
    Object IDs and attributes are preserved in HARNESS_AI XDATA as chunked JSON.
    """
    scale = _number(mm_per_pixel, "mm_per_pixel", positive=True)
    if not isinstance(structure, dict):
        raise ValueError("structure는 JSON 객체여야 합니다.")
    _number(structure.get("width"), "width", positive=True)
    height = _number(structure.get("height"), "height", positive=True)
    objects, texts = structure.get("objects", []), structure.get("texts", [])
    if not isinstance(objects, list) or not isinstance(texts, list):
        raise ValueError("objects와 texts는 배열이어야 합니다.")
    ids = [item.get("id") if isinstance(item, dict) else None for item in objects + texts]
    if any(not isinstance(key, str) or not key or len(key) > 128 for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("각 객체/문자에는 중복되지 않는 1~128자 id가 필요합니다.")
    if semantics is None:
        semantic_attrs = {}
    elif isinstance(semantics, dict) and isinstance(semantics.get("objects"), list):
        semantic_attrs = {}
        for entry in semantics["objects"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("object_id"), str):
                raise ValueError("semantic objects에는 object_id가 필요합니다.")
            if entry["object_id"] in semantic_attrs:
                raise ValueError("중복 semantic object_id입니다.")
            semantic_attrs[entry["object_id"]] = entry.get("attributes", {})
    elif isinstance(semantics, dict):
        semantic_attrs = semantics
    else:
        raise ValueError("semantics는 최종 의미 JSON 또는 {object_id: attributes}여야 합니다.")
    if set(semantic_attrs) - set(ids) or any(not isinstance(value, dict) for value in semantic_attrs.values()):
        raise ValueError("semantic attributes가 유효하지 않거나 존재하지 않는 객체를 가리킵니다.")

    def xy(point):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("좌표는 [x, y] 형식이어야 합니다.")
        x, y = (_number(v, "coordinate") for v in point)
        return (_number(x * scale, "CAD x"), _number((height - y) * scale, "CAD y"))

    def length(value, name):
        return _number(_number(value, name, positive=True) * scale, name, positive=True)

    document = ezdxf.new("R2010", units=ezdxf.units.MM)
    for name in set(LAYERS.values()) | {"TEXT"}:
        document.layers.new(name)
    document.styles.new("OCR", dxfattribs={"font": "malgun.ttf"})
    document.appids.new(XDATA_APP)
    library = None
    importer = None
    if block_library is not None:
        library_path = Path(block_library)
        if library_path.suffix.lower() != ".dxf" or not library_path.is_file():
            raise ValueError("block_library에는 존재하는 로컬 .dxf 파일이 필요합니다.")
        library = ezdxf.readfile(library_path)
        if library.units != ezdxf.units.MM:
            raise ValueError("BLOCK 라이브러리는 mm 단위로 정규화해야 합니다.")
        importer = Importer(library, document)

    def attach(entity, item, kind):
        attributes = semantic_attrs.get(item["id"], item.get("attributes", {}))
        if not isinstance(attributes, dict):
            raise ValueError("객체 attributes는 JSON 객체여야 합니다.")
        metadata = {"object_id": item["id"], "kind": kind, "attributes": attributes}
        try:
            # ASCII escaping is UTF-8-safe and preserves Korean on json.loads().
            encoded = json.dumps(metadata, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("CAD 속성은 유한한 숫자를 포함한 JSON 값이어야 합니다.") from exc
        if len(encoded) > 12000:
            raise ValueError("객체 속성이 XDATA 한도를 초과합니다. 상세 정보는 로컬 JSON sidecar에 보관하세요.")
        entity.set_xdata(XDATA_APP, [(1000, encoded[i:i + 240]) for i in range(0, len(encoded), 240)])

    model = document.modelspace()
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
            raise ValueError("각 객체에는 kind 문자열이 필요합니다.")
        kind, geometry = item["kind"], item.get("geometry")
        if kind not in LAYERS or not isinstance(geometry, dict):
            raise ValueError(f"지원하지 않는 객체/geometry: {kind}")
        attrs = {"layer": LAYERS[kind]}
        try:
            if kind in SYMBOLS:
                if library is None:
                    raise ValueError(f"{item['id']}: 승인된 로컬 BLOCK 라이브러리를 연결해야 합니다.")
                name = geometry["block_name"]
                if not isinstance(name, str) or not name or name.startswith("*") or name not in library.blocks:
                    raise ValueError(f"로컬 라이브러리에 BLOCK 정의가 없습니다: {name}")
                pending, checked = [name], set()
                while pending:
                    dependency = pending.pop()
                    if dependency in checked:
                        continue
                    checked.add(dependency)
                    if dependency not in library.blocks:
                        raise ValueError(f"중첩 BLOCK 정의 누락: {dependency}")
                    block = library.blocks[dependency]
                    if block.block.dxf.flags & 12 or block.units not in (0, ezdxf.units.MM):
                        raise ValueError("외부참조 또는 mm 이외 단위의 BLOCK은 지원하지 않습니다.")
                    for child in block:
                        if child.dxftype() not in {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "POINT", "SOLID", "TRACE", "3DFACE", "TEXT", "MTEXT", "ATTDEF", "INSERT", "HATCH"}:
                            raise ValueError(f"BLOCK 내부의 지원하지 않는 객체: {child.dxftype()}")
                        if child.dxftype() == "INSERT":
                            pending.append(child.dxf.name)
                if name not in document.blocks:
                    importer.import_block(name, rename=False)
                block_scale = _number(geometry.get("scale", 1), "block scale", positive=True)
                rotation = -_number(geometry.get("angle", 0), "block angle")
                entity = model.add_blockref(name, xy(geometry["insert"]), dxfattribs={
                    **attrs, "xscale": block_scale, "yscale": block_scale, "zscale": block_scale, "rotation": rotation,
                })
            elif kind in {"line", "wire"}:
                start, end = xy(geometry["start"]), xy(geometry["end"])
                if start == end:
                    raise ValueError("직선의 시작점과 끝점이 같습니다.")
                entity = model.add_line(start, end, dxfattribs=attrs)
            elif kind == "circle":
                entity = model.add_circle(xy(geometry["center"]), length(geometry["radius"], "radius"), dxfattribs=attrs)
            elif kind == "arc":
                start = _number(geometry["start_angle"], "start_angle") % 360
                end = _number(geometry["end_angle"], "end_angle") % 360
                if start == end:
                    raise ValueError("원호의 시작/끝 각도가 같습니다. 완전한 원은 circle로 지정하세요.")
                entity = model.add_arc(xy(geometry["center"]), length(geometry["radius"], "radius"),
                              (-end) % 360, (-start) % 360, dxfattribs=attrs)
            elif kind == "polyline":
                points = geometry["points"]
                if not isinstance(points, list) or not 2 <= len(points) <= 10000:
                    raise ValueError("polyline에는 2~10000개 점이 필요합니다.")
                points = [xy(p) for p in points]
                if any(a == b for a, b in zip(points, points[1:])):
                    raise ValueError("polyline에 연속 중복점이 있습니다.")
                closed = geometry.get("closed", False)
                if type(closed) is not bool:
                    raise ValueError("polyline closed는 bool이어야 합니다.")
                entity = model.add_lwpolyline(points, close=closed, dxfattribs=attrs)
            elif kind == "spline":
                points = geometry["points"]
                if not isinstance(points, list) or len(points) < 3:
                    raise ValueError("spline에는 최소 3개의 보간점이 필요합니다.")
                points = [xy(p) for p in points]
                if any(a == b for a, b in zip(points, points[1:])):
                    raise ValueError("spline에 연속으로 중복된 점이 있습니다.")
                entity = model.add_spline_control_frame(points, degree=min(3, len(points) - 1), dxfattribs=attrs)
            attach(entity, item, kind)
        except KeyError as exc:
            raise ValueError(f"{item.get('id', kind)}: geometry 필드 누락: {exc.args[0]}") from exc

    for item in texts:
        if not isinstance(item, dict):
            raise ValueError("각 text는 JSON 객체여야 합니다.")
        text, bbox = item.get("text"), item.get("bbox")
        if not isinstance(text, str) or not text or any(ord(c) < 32 for c in text):
            raise ValueError("TEXT에는 제어문자 없는 비어 있지 않은 한 줄 문자열이 필요합니다.")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("TEXT bbox는 [x, y, width, height] 형식이어야 합니다.")
        x, y = (_number(v, "bbox coordinate") for v in bbox[:2])
        box_width = _number(bbox[2], "bbox width", positive=True)
        box_height = _number(bbox[3], "bbox height", positive=True)
        # ponytail: OCR bbox approximates font height/baseline; use local font fitting if exact typography is required.
        entity = model.add_text(text, height=length(box_height * .72, "text height"),
                                dxfattribs={"layer": "TEXT", "style": "OCR"})
        from ezdxf.enums import TextEntityAlignment
        entity.set_placement(xy((x, y+box_height*.85)),xy((x+box_width,y+box_height*.85)),
                             align=TextEntityAlignment.FIT)
        attach(entity, item, "text")

    if importer is not None:
        importer.finalize()
    output = io.StringIO()
    document.write(output)
    return output.getvalue().encode("utf-8")


def dxf_to_dwg(dxf_path, dwg_path, prog_id="AutoCAD.Application") -> Path:
    """Convert locally through an installed, licensed Windows CAD COM server.

    Optional dependency: pywin32. AutoCAD Documents.Open/Document.SaveAs are
    used; ZWCAD requires its installed version's compatible ProgID/API.
    No CAD is launched by importing this module. This adapter is only exercised
    when called and still needs validation on the target CAD/version/license.
    It never closes pre-existing drawings or overwrites an existing output.
    The DWG magic check rejects renamed DXF; it is not a full DWG integrity check.
    """
    source, target = Path(dxf_path).resolve(), Path(dwg_path).resolve()
    if os.name != "nt":
        raise RuntimeError("DWG 변환에는 Windows와 로컬 CAD COM 서버가 필요합니다.")
    if source == target or source.suffix.lower() != ".dxf" or target.suffix.lower() != ".dwg":
        raise ValueError("서로 다른 .dxf 입력과 .dwg 출력 경로가 필요합니다.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    if not isinstance(prog_id, str) or not re.fullmatch(r"(?:AutoCAD|ZWCAD)\.Application(?:\.[\d.]+)?", prog_id):
        raise ValueError("설치된 AutoCAD.Application 또는 ZWCAD.Application ProgID를 지정하세요.")
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("DWG 변환을 사용하려면 pywin32를 설치하세요.") from exc

    pythoncom.CoInitialize()
    try:
        app = win32com.client.Dispatch(prog_id)
        for index in range(app.Documents.Count):
            name = app.Documents.Item(index).FullName
            if name and Path(name).resolve() == source:
                raise RuntimeError("입력 DXF가 CAD에서 이미 열려 있습니다. 닫은 뒤 다시 시도하세요.")
        with tempfile.TemporaryDirectory(prefix="cad-export-", dir=target.parent) as temporary:
            staged = Path(temporary) / "converted.dwg"
            opened = app.Documents.Open(str(source), True)
            try:
                opened.SaveAs(str(staged))
            finally:
                opened.Close(False)
            with staged.open("rb") as stream:
                signature = stream.read(6)
            if not re.fullmatch(rb"AC10[0-9]{2}", signature):
                raise RuntimeError("CAD 결과가 DWG 헤더를 갖고 있지 않습니다. 출력하지 않습니다.")
            # Windows rename fails if the destination appeared during conversion.
            staged.rename(target)
    finally:
        pythoncom.CoUninitialize()
    return target


def self_check():
    """Synthetic-only round trip; never starts CAD or reads company files."""
    drawing = {
        "width": 1000, "height": 800,
        "objects": [
            {"id": "W1", "kind": "wire", "geometry": {"start": [0, 0], "end": [100, 200]}},
            {"id": "C1", "kind": "circle", "geometry": {"center": [300, 400], "radius": 20}},
            {"id": "A1", "kind": "arc", "geometry": {"center": [300, 400], "radius": 20, "start_angle": 0, "end_angle": 90}},
            {"id": "S1", "kind": "spline", "geometry": {"points": [[0, 0], [10, 20], [30, 10]]}},
        ],
        "texts": [{"id": "T1", "text": "Φ12 / 24V / 한글", "bbox": [10, 20, 100, 10]}],
    }
    attributes = {"length": {"value": 150, "unit": "mm"}, "name": "합성속성" * 100}
    document = ezdxf.read(io.StringIO(build_dxf(drawing, 0.5, semantics={"W1": attributes}).decode("utf-8")))
    model = document.modelspace()
    assert document.units == ezdxf.units.MM
    assert model.query("LINE")[0].dxf.start.isclose((0, 400, 0))
    assert model.query("LINE")[0].dxf.end.isclose((50, 300, 0))
    metadata = json.loads("".join(tag.value for tag in model.query("LINE")[0].get_xdata(XDATA_APP)))
    assert metadata == {"object_id": "W1", "kind": "wire", "attributes": attributes}
    assert all(len(tag.value.encode("utf-8")) <= 240 for tag in model.query("LINE")[0].get_xdata(XDATA_APP))
    circle = model.query("CIRCLE")[0]
    assert circle.dxf.center.isclose((150, 200, 0)) and circle.dxf.radius == 10
    arc = model.query("ARC")[0]
    assert arc.dxf.start_angle == 270 and arc.dxf.end_angle == 0
    assert arc.start_point.isclose((150, 190, 0)) and arc.end_point.isclose((160, 200, 0))
    assert model.query("TEXT")[0].dxf.text == drawing["texts"][0]["text"]
    assert len(model.query("SPLINE")) == 1
    assert not document.audit().has_errors
    with tempfile.TemporaryDirectory(prefix="cad-self-check-") as temporary:
        template_path = Path(temporary) / "synthetic_blocks.dxf"
        template = ezdxf.new("R2010", units=ezdxf.units.MM)
        template.blocks.new("PIN").add_circle((0, 0), 1)
        template.blocks.new("CONNECTOR").add_blockref("PIN", (2, 0))
        template.blocks.new("UNUSED").add_line((0, 0), (1, 1))
        template.saveas(template_path)
        symbol = {"id": "CON1", "kind": "connector", "geometry": {
            "insert": [100, 200], "block_name": "CONNECTOR", "scale": 2, "angle": 90,
        }}
        final = {"objects": [{"object_id": "CON1", "attributes": {"part_number": "SYNTHETIC"}}]}
        result = ezdxf.read(io.StringIO(build_dxf({**drawing, "objects": [symbol]}, 0.5,
                                                 template_path, final).decode("utf-8")))
        inserted = result.modelspace().query("INSERT")[0]
        assert inserted.dxf.insert.isclose((50, 300, 0))
        assert inserted.dxf.xscale == 2 and inserted.dxf.rotation == -90
        assert "CONNECTOR" in result.blocks and "PIN" in result.blocks and "UNUSED" not in result.blocks
        assert json.loads("".join(t.value for t in inserted.get_xdata(XDATA_APP)))["attributes"] == final["objects"][0]["attributes"]
        assert not result.audit().has_errors
    for invalid_scale in [0, -1, float("nan"), float("inf"), True, "1"]:
        try:
            build_dxf(drawing, invalid_scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid scale accepted: {invalid_scale}")
    for bad in [{"kind": "circle", "geometry": {"center": [0, 0], "radius": -1}},
                {"kind": "symbol", "geometry": {"insert": [0, 0], "block_name": "TEST"}},
                {"kind": "line", "geometry": {"start": [0, 0]}}]:
        try:
            build_dxf({**drawing, "objects": [{"id": "BAD", **bad}]}, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/unsupported geometry accepted")
    print("CAD self-check passed (synthetic DXF only; DWG COM not exercised).")


if __name__ == "__main__":
    self_check()
