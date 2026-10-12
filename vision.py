"""Local raster/PDF preprocessing and provisional line candidates only.

YOLO and PaddleOCR are not integrated: they require separately installed local
models. No model downloads or network calls occur here. Local OCR results may
be supplied through the structured JSON's ``texts`` list in the application.
"""

from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import pypdfium2 as pdfium


MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000


def _check_size(width, height):
    if not (math.isfinite(width) and math.isfinite(height)) or min(width, height) <= 0:
        raise ValueError("이미지 또는 PDF 페이지 크기가 올바르지 않습니다.")
    if math.ceil(width) * math.ceil(height) > MAX_PIXELS:
        raise ValueError("처리 가능한 이미지 크기는 최대 2,000만 픽셀입니다.")


def preprocess(data: bytes, file_type: str, page_index: int = 0,
               rotation: float = 0, *, preserve_thin_lines: bool = False) -> np.ndarray:
    """Return a grayscale raster; PDF renders the selected page only at 144 dpi.

    Rotation is a manual counterclockwise angle in degrees. The expanded canvas
    prevents clipping; automatic deskew and drawing alignment are not provided.
    PDF rendering dpi is not a physical drawing scale: CAD needs calibration.
    """
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("비어 있지 않은 20MB 이하의 파일이 필요합니다.")
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)) or not math.isfinite(rotation):
        raise ValueError("회전 각도는 유한한 숫자여야 합니다.")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
        raise ValueError("페이지 번호는 0 이상의 정수여야 합니다.")
    if not isinstance(file_type, str):
        raise ValueError("PNG, JPEG 또는 PDF 형식을 지정하세요.")
    kind = file_type.lower().lstrip(".")
    if kind in ("pdf", "application/pdf"):
        try:
            with pdfium.PdfDocument(data) as document:
                if page_index >= len(document):
                    raise ValueError(f"PDF 페이지 범위를 벗어났습니다. 총 {len(document)}페이지입니다.")
                page = document[page_index]
                width, height = page.get_size()
                _check_size(width * 2, height * 2)
                bitmap = page.render(scale=2, grayscale=True)
                try:
                    gray = bitmap.to_numpy().copy()
                finally:
                    bitmap.close()
        except pdfium.PdfiumError as exc:
            raise ValueError("PDF를 읽을 수 없습니다. 손상 또는 암호화 여부를 확인하세요.") from exc
    elif kind in ("png", "image/png", "jpg", "jpeg", "image/jpeg"):
        if page_index != 0:
            raise ValueError("이미지 파일의 페이지 번호는 0이어야 합니다.")
        try:
            with Image.open(BytesIO(data)) as original:
                expected = "PNG" if kind in ("png", "image/png") else "JPEG"
                if original.format != expected:
                    raise ValueError("파일 내용과 지정한 이미지 형식이 일치하지 않습니다.")
                _check_size(*original.size)  # Check metadata before decoding pixels.
                if original.mode in ("RGBA", "LA") or "transparency" in original.info:
                    rgba = original.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    gray = np.array(Image.alpha_composite(background, rgba).convert("L"))
                else:
                    gray = np.array(original.convert("L"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("이미지를 읽을 수 없습니다. PNG/JPEG 파일을 확인하세요.") from exc
    else:
        raise ValueError("지원하는 파일 형식은 PNG, JPEG, PDF입니다.")

    gray = gray.reshape(gray.shape[:2])  # PDFium grayscale may have a channel axis.
    try:
        if not preserve_thin_lines:
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        angle = rotation % 360
        if angle:
            height, width = gray.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            new_width = math.ceil(width * cos + height * sin - 1e-9)
            new_height = math.ceil(width * sin + height * cos - 1e-9)
            _check_size(new_width, new_height)
            matrix[0, 2] += (new_width - width) / 2
            matrix[1, 2] += (new_height - height) / 2
            gray = cv2.warpAffine(gray, matrix, (new_width, new_height), borderValue=255)
    except cv2.error as exc:
        raise ValueError("OpenCV 전처리에 실패했습니다. 이미지 크기와 형식을 확인하세요.") from exc
    return gray


def extract_geometry(gray: np.ndarray) -> dict:
    """Return unverified line candidates, never interpreted electrical objects."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("도형 추출에는 uint8 형식의 2차원 흑백 이미지가 필요합니다.")
    height, width = gray.shape
    _check_size(width, height)
    try:
        edges = cv2.Canny(gray, 50, 150)
        # ponytail: fixed Hough thresholds are preview heuristics; tune on local labeled drawings.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=30, maxLineGap=5)
    except cv2.error as exc:
        raise ValueError("OpenCV 선분 추출에 실패했습니다. 입력 이미지를 확인하세요.") from exc
    candidates = [] if lines is None else lines.reshape(-1, 4).tolist()
    warnings = [
        "선분은 검증 전 후보입니다. 중복·교차·분기·배선 연결을 확인하지 않았습니다.",
        "YOLO/PaddleOCR 로컬 모델은 아직 연결되지 않았습니다. 문자·기호·원·원호·곡선은 자동 추출하지 않습니다.",
        "자동 기울기 보정과 도면 정렬은 미구현입니다. 필요하면 수동 회전을 적용하세요.",
    ]
    if len(candidates) > 5000:
        candidates = candidates[:5000]
        warnings.append("선분 후보가 5,000개를 넘어 일부만 표시됩니다. 영역을 줄여 다시 처리하세요.")
    return {
        "width": width,
        "height": height,
        "objects": [
            {"id": f"LINE_{index:04d}", "kind": "line",
             "geometry": {"start": [x1, y1], "end": [x2, y2]}}
            for index, (x1, y1, x2, y2) in enumerate(candidates, 1)
        ],
        "texts": [],
        "warnings": warnings,
    }


def self_check():
    """Runnable synthetic check; no company drawings or network required."""
    drawing = np.full((120, 160), 255, np.uint8)
    cv2.line(drawing, (10, 60), (150, 60), 0, 3)
    ok, encoded = cv2.imencode(".png", drawing)
    assert ok
    data = encoded.tobytes()
    gray = preprocess(data, "png")
    assert gray.shape == (120, 160) and extract_geometry(gray)["objects"]
    assert preprocess(data, "png", rotation=90).shape == (160, 120)
    with pdfium.PdfDocument.new() as document:
        document.new_page(80, 60)
        output = BytesIO()
        document.save(output)
    pdf = output.getvalue()
    assert preprocess(pdf, "pdf").shape == (120, 160)
    for args in [(b"", "png"), (b"broken", "pdf"), (data, "gif"),
                 (data, "png", 0, float("nan")), (pdf, "pdf", 1)]:
        try:
            preprocess(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input accepted: {args[1:]}")
    try:
        extract_geometry(np.zeros((10, 10), dtype=float))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid geometry input accepted")
    print("vision synthetic checks passed")


def draw_overlay(gray, objects=(), texts=()):
    """Local review image: geometry in red, symbol/OCR boxes in blue/green."""
    from PIL import ImageDraw
    canvas = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for obj in objects:
        geometry = obj.get("geometry", {})
        if obj.get("kind") in ("line", "wire") and "start" in geometry and "end" in geometry:
            draw.line([tuple(geometry["start"]), tuple(geometry["end"])], fill="red", width=2)
        elif obj.get("kind") == "polyline":
            points = [tuple(p) for p in geometry["points"]]
            if geometry.get("closed"):
                points.append(points[0])
            draw.line(points, fill="red", width=1)
        elif obj.get("kind") in ("circle", "arc"):
            x,y = geometry["center"];r=geometry["radius"]
            if obj["kind"] == "circle":
                draw.ellipse((x-r,y-r,x+r,y+r),outline="red",width=1)
            else:
                draw.arc((x-r,y-r,x+r,y+r),geometry["start_angle"],geometry["end_angle"],fill="red",width=1)
        if "bbox" in obj:
            x, y, w, h = obj["bbox"]
            draw.rectangle((x, y, x+w, y+h), outline="blue", width=2)
            draw.text((x, max(0, y-12)), str(obj.get("class_name", obj["kind"])), fill="blue")
    for text in texts:
        x, y, w, h = text["bbox"]
        draw.rectangle((x, y, x+w, y+h), outline="green", width=2)
    return canvas


if __name__ == "__main__":
    self_check()
