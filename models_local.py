"""Local YOLO / PaddleOCR adapters; imports and model loading are explicit.

Only trusted local .pt weights may be loaded: PyTorch checkpoints can contain
executable pickle data. No company image, OCR text or model is sent to a service.
The offline guard is process-wide while active; use one notebook task at a time.
"""
from contextlib import contextmanager
import hashlib
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Support native dependencies inherited from a Windows Conda runtime.
# This is NOT the NumPy crash fix: use the project-local wheel and check_runtime.py.
# Keep this handle alive; closing it removes the DLL search directory.
_dll_path = Path(sys.base_prefix) / "Library" / "bin"
_dll_handle = os.add_dll_directory(str(_dll_path)) if os.name == "nt" and _dll_path.is_dir() else None


KINDS = {"connector", "terminal", "ground", "splice", "symbol"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _local_path(value, *, directory=False):
    if not isinstance(value, (str, os.PathLike)) or "://" in str(value):
        raise ValueError("존재하는 로컬 경로만 사용할 수 있습니다.")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"로컬 {'폴더' if directory else '파일'}가 없습니다: {path}")
    return path


def _task(task):
    if task not in {"detect", "obb"}:
        raise ValueError("task는 'detect' 또는 'obb'여야 합니다.")


def _classes(names):
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("names의 클래스 번호는 0부터 연속된 정수여야 합니다.")
        names = [names[i] for i in range(len(names))]
    if (not isinstance(names, list) or not names or
            any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("중복 없는 클래스 이름 목록(names)이 필요합니다.")
    return names


def validate_yolo_dataset(data_yaml, task="obb"):
    """Validate local images/{train,val[,test]} and matching labels directories.

    `detect`: class cx cy width height. `obb`: class x1 y1 ... x4 y4.
    Coordinates must be normalized. Empty label files explicitly mark negatives.
    The SHA-256 check rejects identical image files across different splits.
    """
    import yaml

    _task(task)
    yaml_path = _local_path(data_yaml)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    allowed = {"path", "train", "val", "test", "names", "nc"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("dataset.yaml에는 path/train/val/test/names/nc만 허용합니다. download 코드는 금지합니다.")
    names = _classes(data.get("names"))
    if "nc" in data and data["nc"] != len(names):
        raise ValueError("nc와 names의 클래스 수가 다릅니다.")
    configured_root = data.get("path", ".")
    if not isinstance(configured_root, str) or "://" in configured_root:
        raise ValueError("dataset path는 로컬 폴더여야 합니다.")
    root = _local_path(yaml_path.parent / configured_root, directory=True)
    seen_hashes, stats = {}, {}
    for split in ("train", "val", "test"):
        if split not in data:
            if split != "test":
                raise ValueError(f"{split} 데이터 경로가 필요합니다.")
            continue
        value = data[split]
        if not isinstance(value, str) or "://" in value:
            raise ValueError("각 split은 images/train 등 로컬 폴더 하나여야 합니다.")
        image_dir = _local_path(root / value, directory=True)
        expected = (root / "images" / split).resolve()
        if image_dir != expected or not image_dir.is_relative_to(root):
            raise ValueError(f"지원하는 이미지 폴더 구조: {root / 'images' / split}")
        label_dir = _local_path(root / "labels" / split, directory=True)
        if not label_dir.is_relative_to(root):
            raise ValueError("라벨 폴더가 dataset root를 벗어났습니다.")
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"{split} 이미지가 없습니다.")
        expected_labels, count, negatives = set(), 0, 0
        for image in images:
            if not image.resolve().is_relative_to(root):
                raise ValueError("이미지 경로가 dataset root를 벗어났습니다.")
            with image.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            # ponytail: exact file hashes catch copied images, not re-encoded copies;
            # group source drawing/revision families before splitting the dataset.
            previous = seen_hashes.setdefault(digest, split)
            if previous != split:
                raise ValueError(f"학습/평가 이미지 중복: {previous}와 {split}: {image.name}")
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            label = _local_path(label)
            if not label.is_relative_to(label_dir):
                raise ValueError("라벨 경로가 labels 폴더를 벗어났습니다.")
            if label in expected_labels:
                raise ValueError("확장자만 다른 이미지가 같은 라벨 파일을 공유합니다.")
            expected_labels.add(label)
            rows = [row.split() for row in label.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
            negatives += not rows
            row_keys = set()
            for row in rows:
                if len(row) != (9 if task == "obb" else 5):
                    raise ValueError(f"{label.name}: {task} 라벨 열 수가 맞지 않습니다.")
                try:
                    class_id = int(row[0])
                    coords = [float(x) for x in row[1:]]
                except ValueError as exc:
                    raise ValueError(f"{label.name}: 클래스 번호/좌표를 읽을 수 없습니다.") from exc
                if not 0 <= class_id < len(names) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords):
                    raise ValueError(f"{label.name}: 클래스 범위 또는 [0,1] 좌표 범위를 벗어났습니다.")
                if task == "detect":
                    cx, cy, w, h = coords
                    if min(w, h) <= 0 or min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
                        raise ValueError(f"{label.name}: 잘못된 axis-aligned bbox입니다.")
                else:
                    pts = list(zip(coords[::2], coords[1::2]))
                    crosses = []
                    for i in range(4):
                        a, b, c = pts[i], pts[(i+1) % 4], pts[(i+2) % 4]
                        crosses.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
                    if not (all(x > 1e-10 for x in crosses) or all(x < -1e-10 for x in crosses)):
                        raise ValueError(f"{label.name}: OBB 꼭짓점 순서/면적이 잘못되었습니다.")
                key = (class_id, *coords)
                if key in row_keys:
                    raise ValueError(f"{label.name}: 같은 라벨 행이 중복됩니다.")
                row_keys.add(key)
                count += 1
        actual_labels = {p.resolve() for p in label_dir.rglob("*.txt")}
        if actual_labels != expected_labels:
            raise ValueError(f"{split}: 대응 이미지가 없는 라벨 파일이 있습니다.")
        if not count:
            raise ValueError(f"{split}: 양성 객체 라벨이 하나 이상 필요합니다.")
        stats[split] = {"images": len(images), "objects": count, "negative_images": negatives}
    return {"root": str(root), "task": task, "classes": names, "splits": stats}


@contextmanager
def _offline(paddle_cache=None):
    """Block package auto-downloads as well as HTTP calls during this operation."""
    def blocked(*args, **kwargs):
        raise PermissionError("로컬 모델 단계에서는 네트워크 연결을 허용하지 않습니다.")

    cache = Path(__file__).parent / ".model_cache"
    env = {"YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
           "YOLO_CONFIG_DIR": str(cache / "ultralytics"), "HF_HUB_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
           "PADDLE_PDX_CACHE_HOME": str(Path(paddle_cache) if paddle_cache else cache / "paddlex"),
           "WANDB_DISABLED": "true"}
    # A notebook is single-user and sequential. For a server, use a dedicated
    # network-isolated worker instead of process-wide socket patches.
    with patch.dict(os.environ, env), patch("socket.socket.connect", blocked), \
            patch("socket.socket.connect_ex", blocked), patch("socket.create_connection", blocked), \
            patch("socket.getaddrinfo", blocked):
        yield


def _yolo_runtime():
    from ultralytics import YOLO, settings

    # Runtime only: do not rewrite the user's global settings file.
    for key in ("sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune",
                "raytune", "tensorboard", "wandb", "vscode_msg", "openvino_msg"):
        if key in settings:
            dict.__setitem__(settings, key, False)
    return YOLO


def load_yolo(weights_path, task="obb"):
    """Load an already-downloaded trusted checkpoint; never resolve a model name."""
    _task(task)
    weights = _local_path(weights_path)
    if weights.suffix.lower() != ".pt":
        raise ValueError("직접 검토한 로컬 .pt 가중치만 사용할 수 있습니다.")
    with _offline():
        detector = _yolo_runtime()(str(weights))
    if detector.task != task:
        raise ValueError(f"가중치 task={detector.task!r}, 요청 task={task!r}가 다릅니다.")
    return detector


def train_yolo(data_yaml, weights_path, output_dir, *, task="obb", epochs=100,
               imgsz=1024, device="cpu", batch=4):
    """Return {'best_weights': Path, 'metrics': dict}; never train on a URL.

    Dataset and checkpoint must already be local. This is not a training-quality
    guarantee: a one-epoch smoke test verifies plumbing, not recognition accuracy.
    """
    report = validate_yolo_dataset(data_yaml, task)
    for value in (epochs, imgsz, batch):
        if type(value) is not int or value <= 0:
            raise ValueError("epochs/imgsz/batch는 양수 정수여야 합니다.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if weights_path is None:
        # Build the installed YOLO11 nano architecture with random weights.
        # This requires no pretrained checkpoint or network download.
        import shutil
        with _offline():
            YOLO = _yolo_runtime()
            import ultralytics
            suffix = "-obb" if task == "obb" else ""
            source = Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / f"yolo11{suffix}.yaml"
            architecture = output / f"yolo11n{suffix}.yaml"
            shutil.copyfile(source, architecture)
            detector = YOLO(str(architecture), task=task)
    else:
        detector = load_yolo(weights_path, task)
    # Write a normalized config: no executable download fields and absolute paths.
    import json
    dataset = output / "dataset.local.yaml"
    root = Path(report["root"])
    config = {"path": str(root), "names": report["classes"],
              **{split: str(root / "images" / split) for split in report["splits"]}}
    dataset.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with _offline():
        result = detector.train(data=str(dataset), project=str(output), name="train",
                                exist_ok=False, epochs=epochs, imgsz=imgsz, device=device,
                                batch=batch, workers=0, amp=False, plots=False,
                                save=True, val=True, cache=False)
    best = Path(detector.trainer.best).resolve()
    if not best.is_file() or not best.is_relative_to(output):
        raise RuntimeError("학습이 끝났으나 지정 출력 폴더에서 best.pt를 찾지 못했습니다.")
    metrics = {str(key): float(value) for key, value in result.results_dict.items()}
    return {"best_weights": best, "metrics": metrics}


def _image(image):
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("uint8 흑백 또는 BGR 3채널 이미지가 필요합니다.")
    if image.shape[0] * image.shape[1] > 20_000_000:
        raise ValueError("최대 2,000만 픽셀 이미지만 처리합니다.")
    return np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def detect_symbols(image, model, class_map=None, *, conf=0.25, imgsz=1024, device="cpu"):
    """Return pipeline objects, preserving arbitrary detector classes.

    class_map example: {'R': {'kind': 'symbol', 'block_name': 'RESISTOR'}}.
    No block is invented for an unmapped class. OBB rotation is box orientation,
    not verified connector pin direction; confirm it before inserting CAD blocks.
    """
    import numpy as np
    image = _image(image)
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("conf는 [0,1] 숫자여야 합니다.")
    if getattr(model, "task", None) not in {"detect", "obb"}:
        raise ValueError("detect 또는 obb 모델이 필요합니다.")
    class_map = class_map or {}
    if not isinstance(class_map, dict):
        raise ValueError("class_map은 클래스명→{kind,block_name} 매핑이어야 합니다.")
    height, width = image.shape[:2]
    with _offline():
        results = model.predict(source=image, conf=conf, imgsz=imgsz, device=device,
                                save=False, verbose=False)
    if len(results) != 1:
        raise ValueError("입력 이미지 하나당 검출 결과 하나가 필요합니다.")
    result = results[0]
    obb = getattr(result, "obb", None)
    boxes = obb if obb is not None else getattr(result, "boxes", None)
    if boxes is None:
        return []
    cls, scores = _array(boxes.cls), _array(boxes.conf)
    if obb is not None:
        polygons = _array(obb.xyxyxyxy)
        angles = _array(obb.xywhr)[:, 4] * 180 / math.pi
    else:
        xyxy = _array(boxes.xyxy)
        polygons = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in xyxy]).reshape(-1,4,2)
        angles = np.zeros(len(cls))
    if polygons.shape != (len(cls),4,2) or len(scores) != len(cls) or len(angles) != len(cls):
        raise ValueError("검출 결과 배열 크기가 맞지 않습니다.")
    objects = []
    for class_id, score, polygon, angle in zip(cls, scores, polygons, angles):
        if not np.isfinite(polygon).all() or not all(math.isfinite(float(x)) for x in (class_id, score, angle)):
            raise ValueError("검출 결과에 유한하지 않은 숫자가 있습니다.")
        if int(class_id) != class_id or not 0 <= score <= 1:
            raise ValueError("유효하지 않은 class id/confidence입니다.")
        if score < conf:
            continue
        name = result.names[int(class_id)]
        mapping = class_map.get(name, {})
        if not isinstance(name, str) or not isinstance(mapping, dict) or set(mapping) - {"kind", "block_name"}:
            raise ValueError("클래스 이름 또는 class_map 형식이 잘못되었습니다.")
        kind = mapping.get("kind", name.lower() if name.lower() in KINDS else "symbol")
        if kind not in KINDS:
            raise ValueError("기호 kind는 connector/terminal/ground/splice/symbol 중 하나여야 합니다.")
        x1,y1 = np.maximum(polygon.min(axis=0), [0,0])
        x2,y2 = np.minimum(polygon.max(axis=0), [width,height])
        if x2 <= x1 or y2 <= y1:
            continue
        geometry = {"insert": polygon.mean(axis=0).astype(float).tolist(), "angle": float(angle)}
        if mapping.get("block_name"):
            if not isinstance(mapping["block_name"], str):
                raise ValueError("block_name은 문자열이어야 합니다.")
            geometry["block_name"] = mapping["block_name"]
        objects.append({"id": f"SYMBOL_{len(objects)+1:04d}", "kind": kind,
                        "class_name": name, "confidence": float(score),
                        "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)],
                        "obb": polygon.astype(float).tolist(), "geometry": geometry,
                        "orientation_verified": False, "status": "needs_review"})
    return objects


def load_ocr(det_dir, rec_dir, *, det_name="PP-OCRv5_server_det",
             rec_name="korean_PP-OCRv5_mobile_rec", device="cpu"):
    """Load exported inference models, not raw training .pdparams checkpoints."""
    dirs = [_local_path(p, directory=True) for p in (det_dir, rec_dir)]
    for folder in dirs:
        if (not (folder / "inference.yml").is_file()
                or not (folder / "inference.pdiparams").is_file()
                or not any((folder / name).is_file() for name in ("inference.json", "inference.pdmodel"))):
            raise ValueError(f"추론용 export 파일(inference.yml/json 또는 pdmodel/pdiparams)이 필요합니다: {folder}")
    cache_root = dirs[0].parent.parent
    with _offline(paddle_cache=cache_root):
        from paddleocr import PaddleOCR
        return PaddleOCR(text_detection_model_name=det_name, text_detection_model_dir=str(dirs[0]),
                         text_recognition_model_name=rec_name, text_recognition_model_dir=str(dirs[1]),
                         use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=False, device=device,
                         enable_mkldnn=False)


def recognize_texts(image, ocr, *, min_confidence=0.8):
    """Return local OCR text candidates with no invented object relationships."""
    import numpy as np
    image = _image(image)
    if image.ndim == 2:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("OCR confidence는 [0,1] 숫자여야 합니다.")
    with _offline(paddle_cache=getattr(ocr, "_cad_model_root", None)):
        results = ocr.predict(input=image)
    if len(results) != 1:
        raise ValueError("입력 이미지 하나당 OCR 결과 하나가 필요합니다.")
    payload = results[0].json
    payload = payload.get("res", payload)
    words, scores, polygons = (payload[key] for key in ("rec_texts", "rec_scores", "rec_polys"))
    if not len(words) == len(scores) == len(polygons):
        raise ValueError("OCR 결과 배열 크기가 맞지 않습니다.")
    height, width = image.shape[:2]
    texts = []
    for word, score, polygon in zip(words, scores, polygons):
        polygon = _array(polygon)
        if (not isinstance(word, str) or len(word) > 2048 or not math.isfinite(score)
                or not 0 <= score <= 1 or polygon.shape != (4,2) or not np.isfinite(polygon).all()):
            raise ValueError("잘못된 OCR 문자/점수/좌표 결과입니다.")
        if score < min_confidence or not word.strip():
            continue
        x1,y1 = np.maximum(polygon.min(axis=0), [0,0])
        x2,y2 = np.minimum(polygon.max(axis=0), [width,height])
        if x2 <= x1 or y2 <= y1:
            continue
        texts.append({"id": f"TEXT_{len(texts)+1:04d}", "text": word,
                      "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)],
                      "confidence": float(score), "polygon": polygon.astype(float).tolist(),
                      "links": [], "status": "needs_review"})
    return texts


def self_check():
    """Synthetic, offline check: dataset boundaries and mock detector/OCR adapters."""
    import json
    import tempfile
    from types import SimpleNamespace
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            # Only metadata validation is tested here, not an image decoder.
            (root / "images" / split / "sample.png").write_bytes(split.encode())
            (root / "labels" / split / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n")
        config = root / "dataset.yaml"
        config.write_text(json.dumps({"path": str(root), "train": "images/train", "val": "images/val", "names": ["R"]}))
        assert validate_yolo_dataset(config, "detect")["splits"]["train"]["objects"] == 1
        for change in ("0 0.5 0.5 2 0.4", "0 nan 0.5 0.2 0.4", "1 0.5 0.5 0.2 0.4"):
            label = root / "labels/val/sample.txt"
            label.write_text(change)
            try:
                validate_yolo_dataset(config, "detect")
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid label accepted")
        label.write_text("0 0.5 0.5 0.2 0.4")
        (root / "images/val/sample.png").write_bytes(b"train")
        try:
            validate_yolo_dataset(config, "detect")
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-split duplicate accepted")
    image = np.zeros((60,100), dtype=np.uint8)
    boxes = SimpleNamespace(cls=np.array([0]), conf=np.array([0.9]), xyxy=np.array([[10,20,30,40]]))
    result = SimpleNamespace(obb=None, boxes=boxes, names={0:"R"})
    model = SimpleNamespace(task="detect", predict=lambda **kwargs: [result])
    objects = detect_symbols(image, model)
    assert objects[0]["kind"] == "symbol" and objects[0]["class_name"] == "R"
    assert "block_name" not in objects[0]["geometry"]
    assert detect_symbols(image, model, {"R":{"block_name":"RESISTOR"}})[0]["geometry"]["block_name"] == "RESISTOR"
    ocr = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(json={"res":{
        "rec_texts":["0.5SQ"], "rec_scores":[0.9], "rec_polys":[[[5,5],[30,5],[30,15],[5,15]]]}})])
    texts = recognize_texts(image, ocr)
    assert texts[0]["text"] == "0.5SQ" and texts[0]["links"] == []
    from pipeline import validate_structure
    validate_structure({"width":100, "height":60, "objects":objects, "texts":texts})
    print("models_local synthetic dataset and mock-inference checks passed (no live model inference)")


if __name__ == "__main__":
    self_check()
