"""Rebuild the user-facing notebook. Does not process or train company data."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []
def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))

md('''# 벡터 PDF → 심볼 학습(640×640) → CAD BLOCK 교체

**이 노트북은 도면 비교 코드와 독립적으로 실행합니다.** VS Code에서 Python 3.11 가상환경을 커널로 선택하고 위에서 아래로 실행하세요.

1. 원본 PDF를 저해상도 미리보기로 표시합니다.
2. 심볼 위치를 클릭해 **정확히 640×640 픽셀** ROI를 저장합니다.
3. ROI 안의 모든 학습 대상 심볼에 박스와 클래스를 직접 지정합니다.
4. YOLO11 detect를 `imgsz=640`으로 학습합니다.
5. PDF를 같은 DPI의 640×640 겹침 타일로 검출하고 전체 도면 좌표로 합칩니다.
6. 검출 결과를 검토하고 BLOCK 생성 방식을 고릅니다.
7. CAD에서 PDFIMPORT를 실행한 뒤, 생성된 AutoLISP로 기존 심볼 객체를 BLOCK으로 묶거나 라이브러리 BLOCK으로 교체하고 새 DWG로 저장합니다.

**640 ROI 자체가 정답 라벨은 아닙니다.** ROI 안에서 심볼만 감싸는 박스가 라벨입니다. 배선·문자를 과하게 포함하지 마세요. 같은 BLOCK으로 바꿀 모양/핀 구성이 같은 심볼끼리 같은 클래스를 사용하세요.

PDF와 이미지의 렌더링·라벨링·학습·검출은 로컬입니다. OpenAI API와 OCR은 이 작업에 사용하지 않습니다. PDFIMPORT와 DWG 저장은 CAD에서 실행하며, Python만으로 DWG가 생성되는 것으로 표시하지 않습니다.

대상 PDF는 CATIA V5에서 생성된 가로로 긴 벡터 PDF(1페이지, 약 12500.79×2383.94 pt)입니다. 전체를 300 DPI로 렌더링하면 약 5억 픽셀이므로 고해상도 전체 이미지 대신 필요한 ROI만 렌더링합니다.''')

md('''## 0. 처음 한 번 설치

새 환경을 만들려면 VS Code 터미널에서 `py -3.11 -m venv .venv-symbol` 실행 후 해당 환경을 커널로 선택하세요. 아래 셀은 **현재 노트북 커널**에 설치합니다. 설치 후 VS Code의 **Restart Kernel**을 누르고 1번부터 실행하세요.

CUDA GPU를 쓸 경우 PyTorch는 [공식 설치 안내](https://pytorch.org/get-started/locally/)에서 드라이버에 맞는 명령으로 먼저 설치하세요. 기본 설정은 CPU이고 데이터는 외부로 전송하지 않습니다.''')
code('''# 처음 한 번만 실행. 기존 환경을 보존하려면 새 가상환경 커널에서 실행하세요.
# %pip install -r requirements.txt
# VS Code 작업 폴더가 프로젝트 상위라면 다음 경로를 사용하세요.
# %pip install -r symbol_roi640/requirements.txt
import sys
print("현재 커널:", sys.executable)''')

md('''## 1. 실행 설정

`CLASSES`를 먼저 정하세요. 예시 이름은 임의의 형태 구분명이며 아직 PDF 심볼을 자동 분류한 결과가 아닙니다. 이름/순서를 바꾸려면 새 `DATA_ROOT`를 사용하세요.

`DPI=150`이면 640픽셀은 PDF상 약 307.2 pt 범위입니다. 큰 심볼이 잘리면 DPI를 낮추고, 작은 심볼이 흐리면 올리세요. **학습 ROI와 검출은 같은 DPI**를 사용하세요. `detect`는 직사각형 박스이며 회전각·핀 접속을 자동 확정하지 않습니다.''')
code(r'''from pathlib import Path
import os, sys, json, math

HERE = Path.cwd()
PROJECT = HERE if (HERE / "roi_tools.py").is_file() else HERE / "symbol_roi640"
if not (PROJECT / "roi_tools.py").is_file():
    raise FileNotFoundError("symbol_roi640 폴더에서 실행하거나 VS Code 작업 폴더를 프로젝트로 열어주세요.")
sys.path.insert(0, str(PROJECT.resolve()))
from roi_tools import *

PDF_PATH = Path(r"C:\Users\Win11Pro\Desktop\1. DWG 객체 변환 도면 생성 샘플\DWG 객체 변환 샘플.pdf")
PAGE = 1
DPI = 150
CLASSES = ["symbol_A", "symbol_B"]  # 직접 정한 형태별 이름으로 수정
DATA_ROOT = PROJECT / "my_dataset"
OUTPUT = PROJECT / "output"
OUTPUT.mkdir(exist_ok=True)
DEVICE = "cpu"  # CUDA 환경이 준비됐으면 0
PRETRAINED = PROJECT / "weights" / "yolo11s.pt"

os.environ["YOLO_CONFIG_DIR"] = str(PROJECT / ".yolo_config")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
if not PDF_PATH.is_file():
    raise FileNotFoundError(PDF_PATH)
print(pdf_info(PDF_PATH))
print("ROI 크기:", SIZE, "x", SIZE, " / PDF 범위(pt):", SIZE * 72 / DPI)''')

md('''## 2. 전체 PDF 미리보기

여기는 위치를 고르는 낮은 해상도 보기입니다. 저장할 ROI는 원본 벡터 PDF에서 따로 렌더링합니다. 오른쪽 부품표까지 자동으로 심볼이라고 라벨링하지 마세요.''')
code('''from IPython.display import display, clear_output
page_preview, page_size_pt = preview(PDF_PATH, PAGE, max_width=3600)
display(page_preview)
print("PDF 좌표계: 좌상단 (0,0), X 오른쪽 / Y 아래. 크기(pt):", page_size_pt)''')

md('''## 3. 직접 ROI 선택 + 라벨링 UI

아래 셀 실행 후:

- 위 도면에서 심볼 중심을 클릭합니다. 확대/이동 도구를 쓴 후에는 그 도구를 끄고 클릭하세요.
- 아래에 640×640 ROI가 나옵니다. 클래스 선택 후 **심볼만 둘러싸도록 드래그**하세요. ROI 안의 대상 심볼을 모두 라벨링합니다.
- 실수한 박스는 `마지막 박스 취소`로 지웁니다.
- `라벨 검토 완료` 체크 후 `ROI·라벨 저장`을 누릅니다. 대상 심볼이 없는 ROI도 음성 샘플로 저장할 수 있습니다.
- 다른 위치를 클릭하면 다음 ROI입니다. 서로 떨어진 위치도 수집해야 train/val 분리가 가능합니다.

VS Code에 Python/Jupyter 확장이 필요합니다. `ipympl` 관련 오류가 나면 0번 설치와 커널 재시작을 확인하세요. UI 대신 4번 수동 좌표 방법도 쓸 수 있습니다.''')
code('''%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import ipywidgets as widgets

plt.close("all")
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 표시
plt.rcParams["axes.unicode_minus"] = False
fig, (ax_page, ax_roi) = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 3]})
ax_page.imshow(page_preview, extent=[0, page_size_pt[0], page_size_pt[1], 0])
ax_page.set_title("PDF 위치 선택: 심볼 중심 클릭 (좌표 단위 pt)")
ax_roi.set_title("아직 ROI를 선택하지 않았습니다")
class_pick = widgets.Dropdown(options=[(name, i) for i, name in enumerate(CLASSES)], description="클래스")
reviewed = widgets.Checkbox(value=False, description="이 ROI의 대상 심볼 전체 라벨 검토 완료")
undo_button = widgets.Button(description="마지막 박스 취소")
save_button = widgets.Button(description="ROI·라벨 저장", button_style="success")
message = widgets.Output()
state = {"image": None, "meta": None, "boxes": [], "artists": []}

def redraw_roi():
    for artist in state["artists"]:
        artist.remove()
    state["artists"] = []
    if state["image"] is not None:
        if ax_roi.images:
            ax_roi.images[0].set_data(state["image"])
        else:
            ax_roi.imshow(state["image"])
        for box in state["boxes"]:
            x1,y1,x2,y2 = box["xyxy"]
            rectangle=Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="red",linewidth=2)
            ax_roi.add_patch(rectangle)
            label=ax_roi.text(x1,y1,CLASSES[box["class_id"]],color="red",fontsize=9)
            state["artists"].extend([rectangle,label])
        ax_roi.set_xlim(0,640); ax_roi.set_ylim(640,0)
        ax_roi.set_title(f"640×640 ROI / 박스 {len(state['boxes'])} / 시작(px): {state['meta']['origin_px']}")
    fig.canvas.draw_idle()

def select_page(event):
    if event.inaxes is not ax_page or event.button != 1 or event.xdata is None:
        return
    if getattr(fig.canvas.toolbar, "mode", ""):
        return
    scale = DPI / 72
    x = max(0, round(event.xdata * scale - 320))
    y = max(0, round(event.ydata * scale - 320))
    image, meta = render_roi(PDF_PATH, PAGE, [x,y], DPI)
    state.update(image=image,meta=meta,boxes=[])
    reviewed.value = False
    redraw_roi()

def select_box(start, end):
    if state["image"] is None or start.xdata is None or end.xdata is None:
        return
    w,h = state["meta"]["valid_size_px"]
    x1,x2 = sorted([max(0,min(w,start.xdata)),max(0,min(w,end.xdata))])
    y1,y2 = sorted([max(0,min(h,start.ydata)),max(0,min(h,end.ydata))])
    if x2-x1 >= 2 and y2-y1 >= 2:
        state["boxes"].append({"class_id": class_pick.value,"xyxy":[x1,y1,x2,y2]})
        reviewed.value=False
        redraw_roi()

def undo_box(_):
    if state["boxes"]: state["boxes"].pop()
    reviewed.value=False
    redraw_roi()

def save_current(_):
    with message:
        clear_output(wait=True)
        try:
            if state["image"] is None: raise ValueError("위쪽 PDF에서 ROI 위치를 클릭하세요.")
            item=save_roi(DATA_ROOT,state["image"],state["meta"],state["boxes"],CLASSES,reviewed=reviewed.value)
            print("저장:", item["id"], "심볼", len(item["boxes"]), "개")
        except Exception as exc: print("저장 안 됨:", exc)

roi_selector=RectangleSelector(ax_roi,select_box,useblit=False,button=[1],minspanx=2,minspany=2,spancoords="pixels")
page_click_id=fig.canvas.mpl_connect("button_press_event",select_page)
undo_button.on_click(undo_box); save_button.on_click(save_current)
display(widgets.HBox([class_pick,undo_button,save_button]),reviewed,message)
plt.show()''')

md('''## 4. 선택 사항: 마우스 UI 대신 좌표로 저장

`MANUAL_ORIGIN_PX`는 DPI 기준 전체 페이지의 좌상단 픽셀 좌표입니다. `MANUAL_BOXES`는 640 ROI 내부 좌표입니다. 보기만 하고 저장하지 않으려면 `SAVE_MANUAL=False`로 둡니다.''')
code('''SAVE_MANUAL = False
MANUAL_ORIGIN_PX = [500, 500]  # 직접 수정
MANUAL_BOXES = [
    # {"class_id": 0, "xyxy": [100, 100, 250, 280]},
]
if SAVE_MANUAL:
    manual_image, manual_meta = render_roi(PDF_PATH, PAGE, MANUAL_ORIGIN_PX, DPI)
    display(manual_image)
    saved = save_roi(DATA_ROOT,manual_image,manual_meta,MANUAL_BOXES,CLASSES,reviewed=True)
    print(saved["id"])''')

md('''## 5. 데이터셋 분리 및 점검

서로 겹치는 ROI는 같은 묶음으로 유지해 train/val에 같은 심볼이 중복되는 것을 막습니다. 모든 ROI가 한 덩어리로 이어지면 분리를 중단하므로 떨어진 영역을 더 수집하세요. 같은 PDF에서 나눈 val 점수는 **다른 도면 일반화 성능이 아닙니다**. 클래스가 한쪽에 빠졌다면 해당 클래스의 다른 위치 샘플을 더 추가하세요.

각 라벨은 Ultralytics detect 형식 `class_id center_x center_y width height`이며 좌표는 0~1입니다. 증강은 학습셋에만 적용됩니다.''')
code('''DATA_YAML, split_stats = export_dataset(DATA_ROOT, val_fraction=.2, seed=42)
print("데이터셋:", DATA_YAML)
print(json.dumps(split_stats,ensure_ascii=False,indent=2))
print(DATA_YAML.read_text(encoding="utf-8"))''')

md('''## 6. 사전학습 YOLO11s 준비 (선택 다운로드)

**YOLO11s detect**를 기본으로 선택했습니다. 형태별 심볼 위치 검출용이며 선을 다시 벡터화하는 모델이 아닙니다. 회사 이미지가 아닌 공개 사전학습 가중치만 다운로드하는 선택 셀입니다. 이미 받은 `yolo11s.pt`를 `weights` 폴더에 놓아도 됩니다. 다운로드 셀 실행 후에는 학습/검출을 오프라인으로 돌립니다.

[Ultralytics detect 데이터 형식](https://docs.ultralytics.com/datasets/detect/) · [검출/학습 공식 문서](https://docs.ultralytics.com/tasks/detect/)''')
code('''DOWNLOAD_PRETRAINED = False  # 필요하면 True로 바꾸고 이 셀만 실행
if DOWNLOAD_PRETRAINED:
    from urllib.request import urlretrieve
    PRETRAINED.parent.mkdir(exist_ok=True)
    if not PRETRAINED.exists():
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",PRETRAINED)
    print("받은 가중치:",PRETRAINED)
else:
    print("사전학습 가중치 존재:",PRETRAINED.is_file(),PRETRAINED)''')

md('''## 7. 학습

ROI 라벨을 충분히 수집한 뒤 `RUN_TRAIN=True`로 바꾸세요. 우선 클래스당 여러 실제 위치·방향의 심볼을 수집하고, 배선만 있는 음성 ROI도 넣으세요. 한 페이지에서 심볼 개수가 부족하면 다른 도면이 필요합니다.

라벨 없는 crop, 라벨 잘린 crop, 같은 crop 복제는 학습 품질을 떨어뜨립니다. 회전/대칭이 BLOCK 방향과 연결되면 flip 증강은 끄세요. 아래는 `workers=0`, CPU, `amp=False`로 Windows에서 먼저 실행을 확인하는 설정입니다.''')
code('''RUN_TRAIN = False
EPOCHS = 100
BATCH = 4
if RUN_TRAIN:
    from ultralytics import YOLO, settings
    from unittest.mock import patch
    settings.update({key:False for key in ["sync","hub","wandb","mlflow","clearml","comet","neptune","dvc","tensorboard"]
                     if key in settings})
    if not PRETRAINED.is_file(): raise FileNotFoundError("6번에서 로컬 사전학습 가중치를 준비하세요.")
    if "DATA_YAML" not in globals(): raise ValueError("5번 데이터셋 분리부터 실행하세요.")
    def block_network(*args,**kwargs):
        raise PermissionError("학습 중 네트워크 호출을 차단했습니다.")
    # 회사 도면 학습 중 패키지/모델 자동 다운로드와 외부 통신 차단.
    with patch("socket.socket.connect",block_network), patch("socket.create_connection",block_network):
        model=YOLO(str(PRETRAINED),task="detect")
        training=model.train(
            data=str(DATA_YAML),imgsz=640,epochs=EPOCHS,batch=BATCH,
            device=DEVICE,workers=0,amp=False,cache=False,
            project=str(PROJECT/"runs"),name="symbols640",exist_ok=False,
            seed=42,patience=25,degrees=0,fliplr=0,flipud=0,
            mosaic=.3,close_mosaic=10,translate=.05,scale=.15,
            hsv_h=0,hsv_s=0,hsv_v=.05,
        )
    BEST_WEIGHTS=Path(model.trainer.best)
    print("학습 결과:",BEST_WEIGHTS)
else:
    print("ROI 라벨을 준비한 뒤 RUN_TRAIN=True로 바꿔 실행하세요.")''')

md('''## 8. 검증 점수와 오류 이미지

학습 이후 val의 precision/recall/mAP와 confusion matrix를 확인하세요. 필요한 클래스가 val에 없으면 그 클래스 검증은 불가능합니다. mAP가 높아도 같은 PDF 기준일 수 있으므로 별도 PDF를 최종 평가에 사용하세요.''')
code('''RUN_VAL = False
BEST_WEIGHTS = globals().get("BEST_WEIGHTS", PROJECT/"runs"/"symbols640"/"weights"/"best.pt")
if RUN_VAL:
    from ultralytics import YOLO, settings
    settings.update({key:False for key in ["sync","hub","wandb","mlflow","clearml","comet","neptune","dvc","tensorboard"]
                     if key in settings})
    validator=YOLO(str(BEST_WEIGHTS),task="detect")
    metrics=validator.val(data=str(DATA_YAML),imgsz=640,device=DEVICE,workers=0,plots=True)
    print(metrics.results_dict)
    print("검증 이미지:",metrics.save_dir)
else:
    print("학습 후 RUN_VAL=True로 변경하세요.")''')

md('''## 9. 전체 도면을 640 타일로 검출

PDF 전체를 작은 한 장으로 줄이지 않습니다. **150 DPI의 640 타일**을 160픽셀씩 겹쳐 검출하고, 동일 클래스의 중복 박스를 전체 좌표에서 합칩니다. 타일 경계에 걸린 심볼은 잘릴 수 있어 `tile_edge=True`로 별도 검토합니다.

처음에는 `REGION_PT=[왼쪽,위,오른쪽,아래]`로 배선 그림 영역만 지정하면 부품표 오검출과 시간을 줄일 수 있습니다. 아래 기본값은 `None`(페이지 전체)입니다. 이 PDF의 전면 탐색은 타일 수가 많습니다. 박스 안의 선·픽셀을 YOLO가 객체별로 분해하는 것은 아니며 실제 BLOCK 형상은 별도로 준비합니다.''')
code('''RUN_INFERENCE = False
REGION_PT = None  # 예: [0, 0, 5000, 2300]. 미리보기 좌표를 보고 직접 설정
CONFIDENCE = .25
if RUN_INFERENCE:
    from unittest.mock import patch
    from ultralytics import settings
    settings.update({key:False for key in ["sync","hub","wandb","mlflow","clearml","comet","neptune","dvc","tensorboard"]
                     if key in settings})
    progress=widgets.IntProgress(value=0,min=0,max=1,description="타일 검출")
    display(progress)
    def progress_cb(done,total):
        progress.max=total;progress.value=done
    def deny_network(*args,**kwargs):
        raise PermissionError("회사 PDF 검출 중 네트워크 호출을 차단했습니다.")
    with patch("socket.socket.connect",deny_network),patch("socket.create_connection",deny_network):
        DETECTION_REPORT=infer_pdf(
            PDF_PATH,BEST_WEIGHTS,CLASSES,page=PAGE,dpi=DPI,
            region_pt=REGION_PT,overlap=160,conf=CONFIDENCE,device=DEVICE,on_progress=progress_cb)
    write_json(OUTPUT/"detections.json",DETECTION_REPORT)
    print("검출:",len(DETECTION_REPORT["detections"]),"타일:",DETECTION_REPORT["tile_count"])
else:
    print("학습 완료 후 RUN_INFERENCE=True로 실행하세요.")''')

md('''## 10. 심볼별 확대 검토 → 승인

검출된 심볼을 원본 crop과 대조합니다. 이름이 같아도 다른 형태가 섞여 있으면 클래스나 라벨을 고쳐 재학습하세요. `approved=False`가 기본이며, 확인한 ID만 승인하세요. `rotation_deg`는 CAD BLOCK의 회전각(반시계, 도)이며 detect 모델이 추정한 값이 아닙니다.''')
code('''if (OUTPUT/"detections.json").is_file():
    DETECTION_REPORT=json.loads((OUTPUT/"detections.json").read_text(encoding="utf-8"))
    for detection,crop in detection_crops(PDF_PATH,DETECTION_REPORT):
        print(detection["id"],detection["class_name"],round(detection["confidence"],3),
              "타일 경계:",detection["tile_edge"],"PDF bbox:",detection["bbox_pt_top"])
        display(crop)
else:
    print("아직 학습된 모델의 실제 검출 결과가 없습니다.")''')
code('''APPROVED_IDS = []  # 예: ["SYMBOL_0001", "SYMBOL_0003"]. 확인한 것만 추가
ROTATIONS = {}      # 예: {"SYMBOL_0003": 90.0}. CAD BLOCK 기준 회전각
if "DETECTION_REPORT" in globals():
    for detection in DETECTION_REPORT["detections"]:
        detection["approved"]=detection["id"] in APPROVED_IDS
        detection["rotation_deg"]=ROTATIONS.get(detection["id"],0.0)
    write_json(OUTPUT/"detections.reviewed.json",DETECTION_REPORT)
    print("승인:",len(APPROVED_IDS))''')

md('''## 11. CAD BLOCK 생성 / 라이브러리 교체 스크립트

**기본 `group_imported` 모드:** YOLO 검출 영역 안의 PDFIMPORT 객체들을 복사해 새 BLOCK 정의를 만들고, 그 BLOCK을 같은 위치에 삽입한 후 원본 선택 객체를 제거합니다. 클래스/심볼 ID가 포함된 BLOCK 이름이 생성됩니다. 심볼의 실제 선·색상·모양을 활용하므로 미리 준비한 BLOCK 라이브러리가 필요 없습니다. 객체마다 BLOCK 정의가 하나씩 생깁니다. 같은 클래스의 서로 다른 도형을 강제로 같은 모양으로 바꾸지 않습니다.

**선택 `replace_library` 모드:** 검출 위치를 회사의 표준 BLOCK으로 교체합니다. 이 모드는 아래 둘 중 하나로 BLOCK을 미리 준비해야 합니다.

- CAD에서 기준 심볼의 실제 선/곡선을 선택하고 `BLOCK`으로 만듭니다.
- 회사의 기존 심볼 BLOCK 라이브러리를 DWG에 삽입해 정의를 불러옵니다.

라이브러리 모드의 `reference_width`는 **BLOCK 정의의 회전 전 가로 폭(CAD 단위)**이고, `anchor`는 검출 박스에서 BLOCK 기준점이 놓일 비율입니다. `[0.5,0.5]`는 중앙 기준점인 BLOCK만 해당합니다. 핀/케이블 접속점을 기준점으로 쓸 경우 그에 맞게 수정하세요. 회전은 검토한 0/90/180/270도만 지원합니다. 기본 묶기 모드에서는 원본 방향/축척을 유지합니다.

PDFIMPORT로 가져온 배선이 한 폴리라인으로 심볼 안팎을 이어가는 경우 그 객체 전체를 지우면 배선이 사라집니다. 생성된 LISP는 **박스 안에 완전히 포함된 객체만**, 지정한 import layer에서 선택하고, 화면으로 확인한 뒤 심볼별 `Yes`를 입력해야 교체합니다. 박스 안에 완전히 들어간 짧은 배선은 여전히 선택될 수 있습니다. 교체 후보를 확인하고 필요하면 `BREAK`로 경계를 나누거나 `Skip`하세요.

캘리브레이션은 원본 PDF의 두 점과 CAD에서 대응되는 두 점을 사용합니다. 아래 기본은 페이지의 좌하단/우상단입니다. 페이지 모서리가 CAD에서 잡히지 않으면 **그림 테두리의 같은 두 교점**을 PDF 미리보기 좌표로 지정하세요. 두 점은 X/Y 모두 떨어져 있어야 합니다. PDFIMPORT는 회전 0°, X/Y 같은 축척으로 가져옵니다.''')
code('''BLOCK_MODE = "group_imported"  # 기본: PDFIMPORT 객체를 그대로 BLOCK으로 묶기
# BLOCK_MODE = "replace_library"  # 선택: 표준 BLOCK 라이브러리로 교체
IMPORT_LAYER = "PDF_Geometry"  # CAD 실제 레이어 이름 확인 (ASCII 이름)
BLOCK_MAP = {
    # 이름/폭/기준점을 실제 준비한 BLOCK에 맞춰 작성하세요.
    # "symbol_A": {"block_name":"CONNECTOR_A","reference_width":10.0,
    #              "anchor":[.5,.5],"import_layer":"PDF_Geometry"},
}
SOURCE_POINTS_PT = [
    [0.0, page_size_pt[1]],  # A: PDF 페이지 좌하단
    [page_size_pt[0], 0.0],  # B: PDF 페이지 우상단
]
MAKE_LISP = False
if MAKE_LISP:
    reviewed_report=json.loads((OUTPUT/"detections.reviewed.json").read_text(encoding="utf-8"))
    lisp_path=make_block_lisp(reviewed_report,BLOCK_MAP,OUTPUT/"replace_symbols.lsp",
                              source_points=SOURCE_POINTS_PT,
                              mode=BLOCK_MODE,import_layer=IMPORT_LAYER)
    print("스크립트:",lisp_path)
    print("CAD에서 APPLOAD → SYMBOL640 → 기준점 A/B 선택 → 심볼별 교체 승인 → SAVEAS")
else:
    print("검출 승인/레이어 확인 후 MAKE_LISP=True로 실행하세요. 라이브러리 모드는 BLOCK_MAP도 필요합니다.")''')

md('''## 12. CAD에서 PDFIMPORT → BLOCK 교체 → 새 DWG 저장

1. AutoCAD/ZWCAD에서 새 도면을 엽니다. 원본 DWG가 있다면 원본 객체를 활용하는 편이 더 정확합니다.
2. `PDFIMPORT`로 이 PDF의 1페이지를 가져옵니다. 벡터 geometry를 가져오고 회전 0°, 균일 축척을 사용하세요. **Import as block은 끄세요.** 켠 경우 가져온 전체 BLOCK을 확인 후 `EXPLODE`해야 개별 선을 선택할 수 있습니다.
3. 기본 묶기 모드는 준비된 BLOCK이 필요 없습니다. 라이브러리 모드라면 심볼 BLOCK 정의를 불러오거나 대표 심볼을 선택해 `BLOCK`을 만듭니다. 이름/기준점/폭을 11번 매핑과 맞추세요.
4. CAD의 Model 탭과 World UCS에서 `APPLOAD`로 `output/replace_symbols.lsp`를 로드한 후 `SYMBOL640`을 실행합니다.
5. 원본 PDF의 A/B와 같은 CAD 두 점을 찍습니다. 임의의 테두리 모서리를 페이지 모서리로 대신 찍으면 좌표가 어긋납니다.
6. 각 강조된 심볼의 선택 객체와 BLOCK을 확인합니다. `Yes`는 교체, 기본 `Skip`은 유지입니다. 기존 선이 박스 밖으로 이어지면 선택되지 않아 일부 원본이 남을 수 있습니다.
7. 결과를 열어 배선·문자·핀·축척을 점검한 뒤 `SAVEAS`로 **새 이름의 DWG**를 저장합니다.

AutoCAD Windows의 AutoLISP/Visual LISP 경로를 대상으로 스크립트를 구성했습니다. ZWCAD 호환성은 해당 버전에서 확인해야 합니다. 실제 CAD 명령 실행/새 DWG 저장은 아직 여기서 테스트하지 않았습니다.

**기본 모드는 PDFIMPORT의 실제 심볼 객체를 BLOCK으로 묶습니다.** YOLO가 BLOCK의 선을 새로 그리는 것이 아닙니다. 라이브러리 모드에서는 별도로 준비한 정확한 BLOCK이 필요합니다. 두 모드 모두 학습 라벨과 정확한 캘리브레이션이 필요하며, 심볼 밖으로 이어지는 폴리라인이나 박스 안 배선을 자동으로 완벽하게 분리하지는 못합니다.

[Autodesk PDFIMPORT 공식 문서](https://help.autodesk.com/cloudhelp/2026/ENU/AutoCAD-Core/files/GUID-02D97F82-47EC-4EE8-9429-E9CF6198EF34.htm)

출력 위치: `my_dataset/raw_images`, `my_dataset/annotations.json`, `my_dataset/exports`, `runs`, `output/detections.json`, `output/detections.reviewed.json`, `output/replace_symbols.lsp`. 학습/검출을 실행하기 전에는 이 노트북이 결과를 만들어냈다고 표시하지 않습니다.''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
                     "language_info":{"name":"python","version":"3.11"}}
nbf.validate(notebook)
nbf.write(notebook,root/"PDF_symbol_ROI640_YOLO_BLOCK.ipynb")
print(root/"PDF_symbol_ROI640_YOLO_BLOCK.ipynb")
