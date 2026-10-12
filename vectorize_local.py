"""Local centerline vectorization adapted to electrical diagrams.

Uses the reviewed skeleton tracing / arc fitting subset of the supplied
drawing-to-dxf archive. Circle support testing follows Image2CAD's principle
of verifying fitted circles against image evidence. Never infers connectivity.
"""
import math
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from reference_vector.skeleton_graph import trace_skeleton_polylines
from reference_vector.geometry_model import VectorDrawing, PolylineDef
from reference_vector.vector_fit import apply_polyline_fittings


def ink_mask(gray):
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("uint8 흑백 이미지가 필요합니다.")
    if gray.size > 4_000_000:
        raise ValueError("중심선 비교 모드는 최대 400만 픽셀입니다. PDF/이미지를 줄이거나 영역별로 처리하세요.")
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 7) > 0


def text_mask(shape, texts):
    mask = np.zeros(shape, np.uint8)
    for text in texts:
        x, y, w, h = text["bbox"]
        # No expansion into adjacent wires or circular coil boundaries.
        x0, y0 = max(0, int(math.ceil(x))), max(0, int(math.ceil(y)))
        x1, y1 = min(shape[1], int(x+w)), min(shape[0], int(y+h))
        mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def _circle_candidates(gray, ink, texts):
    arr = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                          param1=100, param2=16, minRadius=5,
                          maxRadius=max(6, min(gray.shape)//4))
    if arr is None:
        return []
    # Verify circles against the full ink image. PaddleOCR boxes around labels
    # inside coils can overlap the ring itself; excluding those boxes here would
    # erase valid motor/coil circles. The containment check below still rejects
    # OCR glyphs such as an isolated 'O'.
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    distance = cv2.distanceTransform((~ink).astype(np.uint8),cv2.DIST_L2,3)
    angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
    candidates = []
    for x,y,r in arr[0]:
        if x-r < 0 or y-r < 0 or x+r >= gray.shape[1] or y+r >= gray.shape[0]:
            continue
        # Suppress an OCR character such as 'O', but keep rings surrounding labels.
        if any(x-r >= t['bbox'][0] and y-r >= t['bbox'][1] and
               x+r <= t['bbox'][0]+t['bbox'][2] and y+r <= t['bbox'][1]+t['bbox'][3]
               for t in texts):
            continue
        # Refine radius against original thin ink rather than the Hough blur width.
        trial_r=[float(r)+offset for offset in (-2,-1,0,1,2) if r+offset>=5]
        trial_r=[rr for rr in trial_r if x-rr>=0 and y-rr>=0 and x+rr<=gray.shape[1]-1 and y+rr<=gray.shape[0]-1]
        if not trial_r:
            continue
        r=min(trial_r,key=lambda rr:float(distance[np.rint(y+rr*np.sin(angles)).astype(int),np.rint(x+rr*np.cos(angles)).astype(int)].mean()))
        xs = np.rint(x+r*np.cos(angles)).astype(int)
        ys = np.rint(y+r*np.sin(angles)).astype(int)
        support = float(near[ys,xs].mean())
        if support < .9:
            continue
        if any(math.hypot(x-c[0],y-c[1]) < 3 and abs(r-c[2]) < 3 for c in candidates):
            continue
        candidates.append((float(x),float(y),float(r),support))
    return candidates[:300]


def extract_vectors(gray, texts=(), *, min_length=4.0, tolerance=.8):
    ink = ink_mask(gray)
    mask = text_mask(gray.shape, texts)
    circles = _circle_candidates(gray, ink, texts)
    clean = (ink & ~mask).astype(np.uint8)
    for x,y,r,_ in circles:
        cv2.circle(clean, (round(x),round(y)), round(r), 0, 3)
    clean = remove_small_objects(clean.astype(bool), min_size=3, connectivity=2)
    skeleton = skeletonize(clean)
    if skeleton.sum() > 180000:
        raise ValueError("중심선 픽셀이 너무 많습니다. 영역별로 나누어 처리하세요.")
    paths, closed = trace_skeleton_polylines(skeleton)
    polylines = []
    for points, loop in zip(paths, closed):
        pts = np.array(points, np.float32).reshape(-1,1,2)
        if cv2.arcLength(pts, loop) < min_length:
            continue
        simple = cv2.approxPolyDP(pts, tolerance, loop).reshape(-1,2).tolist()
        if len(simple) >= 2:
            polylines.append(PolylineDef(simple, loop))
    # The reference three-point fit can mistake a shallow rectangular path for
    # an enormous arc. Accept a fit only if its entire curve is supported by ink.
    near=cv2.dilate((ink & ~mask).astype(np.uint8),np.ones((3,3),np.uint8))>0
    def supported(cx,cy,r,start=0,span=360):
        angles=np.radians(np.linspace(start,start+span,max(40,int(r*span/40))))
        xs=np.rint(cx+r*np.cos(angles)).astype(int);ys=np.rint(cy+r*np.sin(angles)).astype(int)
        if min(xs)<0 or min(ys)<0 or max(xs)>=gray.shape[1] or max(ys)>=gray.shape[0]:
            return False
        return near[ys,xs].mean()>=.93
    fitted=VectorDrawing()
    for poly in polylines:
        proposal=apply_polyline_fittings(VectorDrawing(polylines=[poly]))
        valid=all(supported(c.cx,c.cy,c.r) for c in proposal.circles) and all(
            supported(a.cx,a.cy,a.r,a.start_angle_deg,(a.end_angle_deg-a.start_angle_deg)%360)
            for a in proposal.arcs)
        if valid:fitted.merge_inplace(proposal)
        else:fitted.polylines.append(poly)
    objects = []
    def add(kind, geometry, **extra):
        objects.append({"id":f"VEC_{len(objects)+1:05d}", "kind":kind,
                        "geometry":geometry,"status":"needs_review",**extra})
    for x,y,r,support in circles:
        add("circle", {"center":[x,y],"radius":r}, ring_support=support)
    for circle in fitted.circles:
        add("circle", {"center":[circle.cx,circle.cy],"radius":circle.r})
    for arc in fitted.arcs:
        add("arc", {"center":[arc.cx,arc.cy],"radius":arc.r,
                    "start_angle":arc.start_angle_deg,"end_angle":arc.end_angle_deg})
    for poly in fitted.polylines:
        if len(poly.points) == 2 and not poly.closed:
            add("line", {"start":list(poly.points[0]),"end":list(poly.points[1])})
        else:
            # Preserve corners. Do not smooth electrical contacts into splines.
            add("polyline", {"points":[list(p) for p in poly.points],"closed":poly.closed})
    if len(objects)>10000:
        raise ValueError("도형이 10000개를 초과했습니다. 영역을 나누세요.")
    return {"width":gray.shape[1],"height":gray.shape[0],"objects":objects,"texts":list(texts),
            "warnings":["중심선·원·원호는 도형 후보입니다. 교차선의 전기적 접속을 확정하지 않습니다.",
                        "기호는 윤곽만 벡터화합니다. 접촉기/타이머 분류·핀 매핑 모델을 학습한 결과가 아닙니다.",
                        "OCR로 잡힌 영역을 제거하므로 문자와 겹친 선/기호가 일부 누락될 수 있습니다."]}
