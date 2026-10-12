"""Local PDF tiles, manual YOLO labels, inference and reviewed CAD replacement.

No company drawing is uploaded. Download pretrained weights separately before
training. PDF coordinates here use TOP LEFT origin, x right and y down.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import random
import re

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

SIZE = 640


def prepared_roi_files(folder):
    """Validate user-cropped files without cropping or resizing any image."""
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    for path in files:
        with Image.open(path) as image:
            if image.size != (SIZE, SIZE):
                raise ValueError(f"{path.name}: {image.size}, expected exactly 640x640. No automatic resizing.")
    return files


def load_prepared_roi(path, *, group=None):
    with Image.open(path) as source:
        if source.size != (SIZE, SIZE):
            raise ValueError("User-prepared ROI must be exactly 640x640")
        image = source.convert("RGB").copy()
    # No page position or render DPI is invented for an externally cropped ROI.
    meta = {"source_kind": "prepared_image", "source_image": str(Path(path).resolve()),
            "group": group, "size_px": [SIZE, SIZE], "valid_size_px": [SIZE, SIZE]}
    return image, meta


def import_prepared_labels(image_folder, label_folder, root, classes, *, groups=None):
    """Import paired user YOLO labels; never interpret a missing label as negative."""
    files = prepared_roi_files(image_folder)
    if not files:
        raise ValueError("No user-prepared 640x640 ROI images")
    pairs = []
    for path in files:
        label = Path(label_folder) / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label: {label}. An intentional negative needs an empty .txt file.")
        boxes = []
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO row in {label.name}")
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if not (0 <= cid < len(classes) and all(math.isfinite(v) for v in (cx,cy,w,h))
                    and 0 < w <= 1 and 0 < h <= 1
                    and cx-w/2 >= -1e-6 and cy-h/2 >= -1e-6
                    and cx+w/2 <= 1+1e-6 and cy+h/2 <= 1+1e-6):
                raise ValueError(f"Invalid class or normalized box: {label.name}")
            boxes.append({"class_id":cid,"xyxy":[max(0,(cx-w/2)*SIZE),max(0,(cy-h/2)*SIZE),
                                                 min(SIZE,(cx+w/2)*SIZE),min(SIZE,(cy+h/2)*SIZE)]})
        pairs.append((path, boxes))
    for path, boxes in pairs:
        image, meta = load_prepared_roi(path, group=(groups or {}).get(path.name))
        save_roi(root, image, meta, boxes, classes, reviewed=True)
    return len(pairs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pdf_info(pdf_path):
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        return [{"page": i + 1, "width_pt": p.get_width(), "height_pt": p.get_height()}
                for i, p in enumerate(doc)]


def preview(pdf_path, page=1, max_width=3200):
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        p = doc[page - 1]
        scale = min(1.0, max_width / p.get_width())
        bitmap = p.render(scale=scale)
        image = bitmap.to_pil().convert("RGB").copy()
        bitmap.close()
        return image, [p.get_width(), p.get_height()]


def render_roi(pdf_path, page, origin_px, dpi=150):
    """Render only one 640px tile; pad the page edge white, never resize it."""
    scale = dpi / 72.0
    if not 36 <= dpi <= 600:
        raise ValueError("DPI must be between 36 and 600")
    x, y = (int(round(v)) for v in origin_px)
    if x < 0 or y < 0:
        raise ValueError("ROI origin must be inside page")
    with pdfium.PdfDocument(str(pdf_path)) as doc:
        p = doc[page - 1]
        width, height = p.get_size()
        if x >= math.ceil(width * scale) or y >= math.ceil(height * scale):
            raise ValueError("ROI lies outside page")
        # PDFium's crop order: left, bottom, right, top, in PDF points.
        crop = (x / scale, max(0, height - (y + SIZE) / scale),
                max(0, width - (x + SIZE) / scale), y / scale)
        bitmap = p.render(scale=scale, crop=crop)
        rendered = bitmap.to_pil().convert("RGB").copy()
        bitmap.close()
        canvas = Image.new("RGB", (SIZE, SIZE), "white")
        canvas.paste(rendered.crop((0, 0, min(SIZE, rendered.width),
                                   min(SIZE, rendered.height))), (0, 0))
        meta = {"page": page, "dpi": dpi, "origin_px": [x, y],
                "size_px": [SIZE, SIZE], "page_size_pt": [width, height],
                "valid_size_px": [min(SIZE, math.ceil(width * scale) - x),
                                  min(SIZE, math.ceil(height * scale) - y)],
                "source_pdf": str(Path(pdf_path).resolve())}
        return canvas, meta


def save_roi(root, image, meta, boxes, classes, *, reviewed=False):
    """boxes=[{'class_id':0,'xyxy':[x1,y1,x2,y2]}, ...]."""
    if not reviewed:
        raise ValueError("Confirm all visible symbols have been labelled first")
    if image.size != (SIZE, SIZE):
        raise ValueError("ROI image must be exactly 640x640")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Class names must be nonempty and unique")
    width, height = meta["valid_size_px"]
    clean = []
    for box in boxes:
        cid = int(box["class_id"])
        x1, y1, x2, y2 = map(float, box["xyxy"])
        if not (0 <= cid < len(classes) and 0 <= x1 < x2 <= width
                and 0 <= y1 < y2 <= height):
            raise ValueError("Label class/box exceeds valid, unpadded image area")
        clean.append({"class_id": cid, "xyxy": [x1, y1, x2, y2]})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "annotations.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"classes": list(classes), "rois": []})
    if manifest["classes"] != list(classes):
        raise ValueError("Class order differs from saved dataset; keep existing order")
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    image_path = root / "raw_images" / f"roi_{key}.png"
    image_path.parent.mkdir(exist_ok=True)
    image.save(image_path)
    item = {**meta, "id": f"roi_{key}", "image": str(image_path.resolve()),
            "image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
            "boxes": clean, "reviewed": True}
    manifest["rois"] = [x for x in manifest["rois"] if x["id"] != item["id"]] + [item]
    write_json(manifest_path, manifest)
    return item


def _overlap(a, b):
    if a.get("source_kind") == "prepared_image" or b.get("source_kind") == "prepared_image":
        # Without PDF crop coordinates, only known groups / duplicates can be grouped.
        return (a.get("group") is not None and a.get("group") == b.get("group")) or a.get("image_hash") == b.get("image_hash")
    if (a["source_pdf"], a["page"]) != (b["source_pdf"], b["page"]):
        return a.get("image_hash") == b.get("image_hash")
    ax, ay = (v * 72 / a["dpi"] for v in a["origin_px"])
    bx, by = (v * 72 / b["dpi"] for v in b["origin_px"])
    aw, bw = SIZE * 72 / a["dpi"], SIZE * 72 / b["dpi"]
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + aw, by + bw)


def export_dataset(root, val_fraction=.2, seed=42):
    """Split overlapping ROI groups together to avoid train/val crop leakage."""
    import shutil
    root = Path(root)
    manifest = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    rois = manifest["rois"]
    if len(rois) < 2 or not 0 < val_fraction < 1:
        raise ValueError("At least two spatially separated ROI groups are required")
    parent = list(range(len(rois)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(rois):
        for j in range(i):
            if _overlap(a, rois[j]) or a["image_hash"] == rois[j]["image_hash"]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(rois)):
        groups.setdefault(find(i), []).append(i)
    group_list = list(groups.values())
    if len(group_list) < 2:
        raise ValueError("All ROIs overlap. Add separated ROIs for validation")
    random.Random(seed).shuffle(group_list)
    count = min(len(group_list) - 1, max(1, round(len(group_list) * val_fraction)))
    val_ids = {i for group in group_list[:count] for i in group}
    # New folder per export: do not leave old examples in a new split.
    output = root / "exports" / f"dataset_{len(list((root / 'exports').glob('dataset_*'))) + 1:03d}"
    stats = {}
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "labels" / split).mkdir(parents=True, exist_ok=False)
        selected = [r for i, r in enumerate(rois) if (i in val_ids) == (split == "val")]
        class_counts = Counter()
        for roi in selected:
            with Image.open(roi["image"]) as im:
                if im.size != (SIZE, SIZE):
                    raise ValueError("Stored image is not 640x640")
            shutil.copyfile(roi["image"], output / "images" / split / f"{roi['id']}.png")
            lines = []
            for box in roi["boxes"]:
                cid = box["class_id"]
                x1, y1, x2, y2 = box["xyxy"]
                lines.append(f"{cid} {(x1+x2)/1280:.8f} {(y1+y2)/1280:.8f} {(x2-x1)/640:.8f} {(y2-y1)/640:.8f}")
                class_counts[manifest["classes"][cid]] += 1
            (output / "labels" / split / f"{roi['id']}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats[split] = {"images": len(selected), "objects": dict(class_counts)}
        if not class_counts:
            raise ValueError(f"{split} has no labelled symbols; collect more separated positive ROIs")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val",
              "names": {i: name for i, name in enumerate(manifest["classes"])}}
    # JSON is valid YAML; Ultralytics reads it with yaml.safe_load.
    write_json(output / "data.yaml", config)
    write_json(output / "split_report.json", {"spatial_groups": len(group_list), "stats": stats})
    return output / "data.yaml", stats


def tile_origins(length, start=0, stop=None, overlap=160):
    if not 0 <= overlap < SIZE:
        raise ValueError("Overlap must be in [0,640)")
    stop = min(length, stop if stop is not None else length)
    start = max(0, start)
    last = max(start, stop - SIZE)
    return sorted(set(list(range(start, last + 1, SIZE - overlap)) + [last]))


def nms(detections, iou=.45):
    """Class-aware page-coordinate NMS after tile-local YOLO NMS."""
    remaining = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in remaining:
        a = d["bbox_px"]
        duplicate = False
        for k in kept:
            if d["class_id"] != k["class_id"] or d["page"] != k["page"]:
                continue
            b = k["bbox_px"]
            inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter
            if inter / max(area, 1e-9) > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    for i, d in enumerate(kept, 1):
        d["id"] = f"SYMBOL_{i:04d}"
    return kept


def infer_pdf(pdf_path, weights, classes, *, page=1, dpi=150, region_pt=None,
              overlap=160, conf=.25, iou=.45, device="cpu", on_progress=None):
    """Only local .pt weights; each inference tile has exactly 640x640 pixels."""
    if not Path(weights).is_file():
        raise FileNotFoundError("Train first, then select runs/.../weights/best.pt")
    from ultralytics import YOLO
    model = YOLO(str(weights), task="detect")
    names = [model.names[i] for i in range(len(model.names))]
    if list(classes) != names or model.task != "detect":
        raise ValueError("Checkpoint class order/task differs from this notebook")
    info = pdf_info(pdf_path)[page - 1]
    scale = dpi / 72
    width, height = math.ceil(info["width_pt"] * scale), math.ceil(info["height_pt"] * scale)
    rect = region_pt or [0, 0, info["width_pt"], info["height_pt"]]
    if not (0 <= rect[0] < rect[2] <= info["width_pt"] and 0 <= rect[1] < rect[3] <= info["height_pt"]):
        raise ValueError("Inference region must be inside PDF page")
    xs = tile_origins(width, int(rect[0]*scale), int(math.ceil(rect[2]*scale)), overlap)
    ys = tile_origins(height, int(rect[1]*scale), int(math.ceil(rect[3]*scale)), overlap)
    detections = []
    for index, (x, y) in enumerate(((x, y) for y in ys for x in xs), 1):
        image, meta = render_roi(pdf_path, page, [x, y], dpi)
        result = model.predict(image, imgsz=SIZE, conf=conf, iou=iou, device=device, verbose=False)[0]
        for box in result.boxes:
            a, b, c, d = map(float, box.xyxy[0].cpu().tolist())
            valid_w, valid_h = meta["valid_size_px"]
            if c > valid_w or d > valid_h:
                continue
            full = [a+x, b+y, c+x, d+y]
            centre = [(full[0]+full[2])/2/scale, (full[1]+full[3])/2/scale]
            if not (rect[0] <= centre[0] <= rect[2] and rect[1] <= centre[1] <= rect[3]):
                continue
            cid = int(box.cls.item())
            detections.append({"page": page, "class_id": cid, "class_name": names[cid],
                "confidence": float(box.conf.item()), "bbox_px": full,
                "bbox_pt_top": [v/scale for v in full], "dpi": dpi,
                "tile_edge": a < 3 or b < 3 or c > valid_w-3 or d > valid_h-3,
                "approved": False, "rotation_deg": 0.0})
        if on_progress:
            on_progress(index, len(xs)*len(ys))
    return {"source_pdf": str(Path(pdf_path).resolve()), "page": page,
            "page_size_pt": [info["width_pt"], info["height_pt"]],
            "region_pt": rect, "tile_count": len(xs)*len(ys),
            "coordinate_system": "pdf_points_top_left_y_down", "detections": nms(detections, iou)}


def detection_crops(pdf_path, report, padding=20):
    for d in report["detections"]:
        x1, y1, x2, y2 = d["bbox_px"]
        x, y = max(0, int(x1)-padding), max(0, int(y1)-padding)
        image, _ = render_roi(pdf_path, report["page"], [x, y], d["dpi"])
        crop = image.crop((0, 0, min(SIZE, math.ceil(x2-x+padding)), min(SIZE, math.ceil(y2-y+padding))))
        draw = ImageDraw.Draw(crop)
        draw.rectangle((x1-x, y1-y, x2-x, y2-y), outline="red", width=2)
        yield d, crop


def make_block_lisp(report, block_map, output, *, source_points=None,
                    mode="group_imported", import_layer="PDF_Geometry"):
    """Make opt-in per-symbol local replacement with two-point CAD calibration.

    No geometric symbol recognition is invented: user must provide reviewed
    class->BLOCK mappings, block width and anchors. Entire entities crossing the
    ROI are excluded. Fully contained wires still need the user's inspection.
    """
    approved = [d for d in report["detections"] if d.get("approved") is True]
    if not approved:
        raise ValueError("Approve verified detections first")
    if mode not in ("group_imported", "replace_library"):
        raise ValueError("Unknown CAD block mode")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", import_layer):
        raise ValueError("Use an ASCII import-layer name")
    width, height = report["page_size_pt"]
    points = source_points or [[0, height], [width, 0]]
    if (len(points) != 2 or any(len(p) != 2 for p in points)
            or any(not math.isfinite(float(v)) for p in points for v in p)
            or abs(points[1][0]-points[0][0]) < 1 or abs(points[1][1]-points[0][1]) < 1):
        raise ValueError("Choose two calibration points separated in both x and y")
    rows = []
    for d in approved:
        x1, y1, x2, y2 = map(float, d["bbox_pt_top"])
        if (not all(math.isfinite(v) for v in (x1, y1, x2, y2))
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError("Approved symbol bbox must be inside PDF page")
        if not re.fullmatch(r"SYMBOL_[0-9]+", d["id"]):
            raise ValueError("Invalid symbol ID")
        if mode == "group_imported":
            cid = int(d["class_id"])
            if cid < 0:
                raise ValueError("Class ID must be nonnegative")
            rows.append(f'  (sr:group "{d["id"]}" "ROI_C{cid}_{d["id"]}" "{import_layer}" '
                        f'{x1:.8f} {y1:.8f} {x2:.8f} {y2:.8f})')
            continue
        m = block_map.get(d["class_name"])
        if not m:
            raise ValueError(f"No BLOCK mapping for {d['class_name']}")
        name, layer = m["block_name"], m["import_layer"]
        # Narrow ASCII identifiers keep generated LISP strings unambiguous.
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in (name, layer)):
            raise ValueError("Use ASCII letters/digits/_/- for BLOCK and import-layer names")
        reference_width = float(m["reference_width"])
        anchor = m.get("anchor", [.5, .5])
        rotation = float(d.get("rotation_deg", 0))
        if (not math.isfinite(reference_width) or reference_width <= 0
                or len(anchor) != 2 or any(not 0 <= float(v) <= 1 for v in anchor)
                or not math.isfinite(rotation)):
            raise ValueError("Invalid block width/anchor/rotation")
        rotation %= 360
        if abs(rotation / 90 - round(rotation / 90)) > 1e-6:
            raise ValueError("Library mode supports verified 0/90/180/270 degree rotations")
        rows.append(f'  (sr:replace "{d["id"]}" "{name}" "{layer}" '
                    f'{x1:.8f} {y1:.8f} {x2:.8f} {y2:.8f} '
                    f'{reference_width:.8f} {float(anchor[0]):.8f} {float(anchor[1]):.8f} {rotation:.8f})')
    header = f'''; Generated locally. Mode: {mode}. First run PDFIMPORT in Model space.
; group_imported builds BLOCKs from the selected original geometry.
; replace_library requires existing named BLOCK definitions.
; Two source points in PDF TOP-LEFT coordinates: {json.dumps(points)}
; Confirm each highlighted selection; entities crossing the bbox are excluded.
(vl-load-com)
(defun sr:map (x y)
  (list (+ (car sr:ca) (* (- x {points[0][0]:.8f}) sr:sx))
        (+ (cadr sr:ca) (* (- y {points[0][1]:.8f}) sr:sy)) 0.0))
(defun sr:copyblock (bn ins ss / doc blk objs arr copied obj newobj)
  (setq doc (vla-get-ActiveDocument (vlax-get-acad-object)))
  (setq blk (vla-Add (vla-get-Blocks doc) (vlax-3d-point '(0.0 0.0 0.0)) bn))
  (setq objs nil)
  (repeat (sslength ss)
    (setq objs (cons (vlax-ename->vla-object (ssname ss (length objs))) objs)))
  (setq arr (vlax-make-safearray vlax-vbObject (cons 0 (1- (length objs)))))
  (vlax-safearray-fill arr objs)
  (setq copied (vla-CopyObjects doc (vlax-make-variant arr) blk))
  (foreach obj (vlax-safearray->list (vlax-variant-value copied))
    (vla-Move obj (vlax-3d-point ins) (vlax-3d-point '(0.0 0.0 0.0))))
  (setq newobj (vla-InsertBlock (vla-get-ModelSpace doc) (vlax-3d-point ins) bn 1.0 1.0 1.0 0.0))
  newobj)
(defun sr:group (id bn layer x1 y1 x2 y2 / ll ur ss ok ins newobj i)
  (setq ll (sr:map x1 y2) ur (sr:map x2 y1))
  (command "_.ZOOM" "_Window" (trans ll 0 1) (trans ur 0 1))
  (setq ss (ssget "_W" (trans ll 0 1) (trans ur 0 1)
    (list '(0 . "LINE,LWPOLYLINE,POLYLINE,ARC,CIRCLE,ELLIPSE,SPLINE,HATCH") (cons 8 layer))))
  (if (and ss (not (tblsearch "BLOCK" bn)))
    (progn
      (sssetfirst nil ss)
      (initget "Yes Skip")
      (setq ok (getkword (strcat "\\n" id " : group highlighted PDF geometry into BLOCK? [Yes/Skip] <Skip>: ")))
      (if (= ok "Yes")
        (progn
          (setq ins (sr:map (/ (+ x1 x2) 2.0) (/ (+ y1 y2) 2.0)))
          (setq newobj (vl-catch-all-apply 'sr:copyblock (list bn ins ss)))
          (if (not (vl-catch-all-error-p newobj))
            (progn (setq i 0) (repeat (sslength ss)
              (entdel (ssname ss i)) (setq i (1+ i))))
            (princ "\\nBlock creation failed; originals preserved. Inspect unused definition before retry."))))
      (sssetfirst nil nil))
    (princ (strcat "\\nNo selectable geometry or BLOCK name already exists: " id))))
(defun sr:replace (id bn layer x1 y1 x2 y2 ref ax ay angle / ll ur ss ok ins size i newobj)
  (if (not (tblsearch "BLOCK" bn))
    (princ (strcat "\\nMissing BLOCK, skipped: " bn))
    (progn
      (setq ll (sr:map x1 y2) ur (sr:map x2 y1))
      (command "_.ZOOM" "_Window" (trans ll 0 1) (trans ur 0 1))
      (setq ss (ssget "_W" (trans ll 0 1) (trans ur 0 1)
        (list '(0 . "LINE,LWPOLYLINE,POLYLINE,ARC,CIRCLE,ELLIPSE,SPLINE") (cons 8 layer))))
      (if ss
        (progn
          (sssetfirst nil ss)
          (initget "Yes Skip")
          (setq ok (getkword (strcat "\\n" id " -> " bn " : inspect highlighted geometry. Replace? [Yes/Skip] <Skip>: ")))
          (if (= ok "Yes")
            (progn
              (setq ins (sr:map (+ x1 (* (- x2 x1) ax)) (+ y1 (* (- y2 y1) ay))))
              (setq size (/ (if (or (= angle 90.0) (= angle 270.0))
                (* (- y2 y1) (- sr:sy)) (* (- x2 x1) sr:sx)) ref))
              (setq newobj (vl-catch-all-apply 'vla-InsertBlock
                (list (vla-get-ModelSpace (vla-get-ActiveDocument (vlax-get-acad-object)))
                  (vlax-3d-point ins) bn size size size (* pi (/ angle 180.0)))))
              (if (not (vl-catch-all-error-p newobj))
                (progn (setq i 0) (repeat (sslength ss)
                  (entdel (ssname ss i)) (setq i (1+ i))))
                (princ "\\nINSERT failed; original geometry preserved."))))
          (sssetfirst nil nil))
        (princ (strcat "\\nNo fully-contained geometry on import layer, skipped: " id))))))
(defun c:SYMBOL640 (/ sr:ca sr:cb sr:sx sr:sy sr:undo *error*)
  (defun *error* (msg)
    (sssetfirst nil nil)
    (if sr:undo (command-s "_.UNDO" "_End"))
    (if msg (princ (strcat "\\nStopped: " msg)))
    (princ))
  (if (/= (getvar "WORLDUCS") 1)
    (progn (princ "\\nSet UCS to World before running SYMBOL640.") (exit)))
  (setq sr:ca (getpoint "\\nPick CAD point matching source point A: "))
  (if sr:ca (setq sr:cb (getpoint "\\nPick CAD point matching source point B: ")))
  (if (and sr:ca sr:cb)
    (progn
      (setq sr:ca (trans sr:ca 1 0) sr:cb (trans sr:cb 1 0))
      (setq sr:sx (/ (- (car sr:cb) (car sr:ca)) {points[1][0]-points[0][0]:.8f}))
      (setq sr:sy (/ (- (cadr sr:cb) (cadr sr:ca)) {points[1][1]-points[0][1]:.8f}))
      (if (and (> sr:sx 0) (< sr:sy 0)
               (< (/ (abs (+ sr:sx sr:sy)) sr:sx) 0.02))
        (progn
          (command "_.UNDO" "_Begin")
          (setq sr:undo T)
'''
    trailer = '''          (command "_.UNDO" "_End")
          (setq sr:undo nil)
        )
        (princ "\\nCalibration rejected: use an unrotated PDFIMPORT with equal X/Y scale."))))
  (princ))
(princ "\\nLoaded. Run SYMBOL640. Save result as a NEW DWG after review.")
(princ)
'''
    Path(output).write_text(header + "\n".join(rows) + "\n" + trailer, encoding="ascii")
    return Path(output)
