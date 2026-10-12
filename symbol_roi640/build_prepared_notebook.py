"""Notebook input is the user's already-cropped 640x640 files."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
target = root / "PDF_symbol_ROI640_YOLO_BLOCK.ipynb"
previous = nbf.read(target, as_version=4)
tail_start = next(i for i, c in enumerate(previous.cells)
                  if c.cell_type == "markdown" and c.source.startswith("## 5. 데이터셋"))
tail = previous.cells[tail_start:]
cells = []
def md(value): cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value): cells.append(nbf.v4.new_code_cell(value.strip()))

md('''# 직접 준비한 640×640 ROI → 심볼 라벨 → YOLO 학습 → CAD BLOCK

**PDF에서 640×640 영역을 고르는 작업은 사용자가 직접 합니다.** 1-1번 셀에서 로컬 자르기 화면을 열고 도면을 확대해 직접 위치를 선택·저장하세요. 이미 잘라 둔 이미지가 있다면 바로 `prepared_rois` 폴더에 넣어도 됩니다. 자동으로 심볼 위치를 고르거나 임의로 사진을 저장하지 않습니다.

1. 직접 자른 PNG/JPG를 `prepared_rois` 폴더에 넣습니다.
2. 이미지 크기가 정확히 640×640인지 검사합니다.
3. 각 이미지 안의 대상 심볼에 박스/클래스를 표시하거나 이미 만든 YOLO 라벨을 가져옵니다.
4. YOLO11s detect를 `imgsz=640`으로 학습합니다.
5. 학습 후 원본 PDF에서 심볼을 검출하고 검토합니다.
6. CAD에서 PDFIMPORT 후 검토한 영역을 BLOCK으로 묶는 스크립트를 실행합니다.

**학습 이미지 640×640과 정답 박스는 별개입니다.** 한 ROI에 심볼이 여러 개라면 모두 표시하세요. 아직 심볼 학습을 실행하거나 전체 학습 데이터 이미지를 만들어 둔 상태가 아닙니다. OpenAI API는 사용하지 않습니다.''')
md('''## 0. 처음 한 번 설치

VS Code에서 Python 3.11 가상환경을 커널로 선택하세요. 아래 설치 명령의 `#`을 지워 한 번 실행하고 커널을 재시작하세요. 이미지 폴더만 있어서는 검출 학습이 되지 않으며 3번 또는 4번에서 정답 박스가 필요합니다.''')
code('''# 처음 한 번만 실행 후 커널 재시작
# %pip install -r requirements.txt
# 상위 프로젝트 폴더에서 실행한다면:
# %pip install -r symbol_roi640/requirements.txt
import sys
print("현재 커널:", sys.executable)''')
md('''## 1. 폴더·클래스 설정

`PREPARED_ROI_DIR`에 직접 자른 640×640 사진을 넣으세요. 기본 위치는 이 노트북 옆 `prepared_rois`입니다. 형태/핀 수가 다른 심볼은 다른 클래스로 정하세요. `CLASSES` 순서는 라벨 파일의 클래스 번호와 같아야 합니다.

원본 PDF 경로는 학습 입력이 아니라 **학습 후 전체 심볼 검출**에 사용합니다. `DPI` 역시 추후 PDF 검출 해상도입니다. 직접 만든 crop의 렌더링 DPI를 모르면 아래 값이 맞다고 가정할 수 없습니다. crop마다 같은 PDF 확대율을 쓰고, 검출 시 심볼 픽셀 크기가 학습 이미지와 비슷하도록 DPI를 조정하세요.''')
code(r'''from pathlib import Path
import os, sys, json, math
from IPython.display import display, clear_output

HERE=Path.cwd()
PROJECT=HERE if (HERE/"roi_tools.py").is_file() else HERE/"symbol_roi640"
if not (PROJECT/"roi_tools.py").is_file():
    raise FileNotFoundError("노트북과 roi_tools.py가 들어 있는 symbol_roi640 폴더에서 실행하세요.")
sys.path.insert(0,str(PROJECT.resolve()))
from roi_tools import *

PREPARED_ROI_DIR=PROJECT/"prepared_rois"  # 직접 자른 사진 폴더
PREPARED_LABEL_DIR=PROJECT/"prepared_labels"  # 이미 라벨링했다면 txt 파일 폴더
CLASSES=["symbol_A","symbol_B"]  # 실제 심볼 형태별 이름으로 수정
DATA_ROOT=PROJECT/"my_dataset_prepared"
OUTPUT=PROJECT/"output"
PRETRAINED=PROJECT/"weights"/"yolo11s.pt"
DEVICE="cpu"
PDF_PATH=Path(r"C:\Users\Win11Pro\Desktop\1. DWG 객체 변환 도면 생성 샘플\DWG 객체 변환 샘플.pdf")
PAGE=1
DPI=150  # 학습 후 원본 PDF 검출 시 조정

ROI_GROUPS={
    # 같은 심볼 위치를 중복해서 자른 사진들은 같은 그룹으로 지정하세요.
    # "roi_001.png":"same_location_A", "roi_002.png":"same_location_A",
}
for folder in (PREPARED_ROI_DIR,PREPARED_LABEL_DIR,OUTPUT): folder.mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"]=str(PROJECT/".yolo_config")
os.environ["YOLO_AUTOINSTALL"]="false"
os.environ["YOLO_OFFLINE"]="true"
if PDF_PATH.is_file():
    info=pdf_info(PDF_PATH)[PAGE-1]
    page_size_pt=[info["width_pt"],info["height_pt"]]
else:
    page_size_pt=None
print("직접 자른 이미지 넣을 곳:",PREPARED_ROI_DIR.resolve())''')
md('''## 1-1. PDF를 보면서 직접 640×640 자르기

아래 셀을 실행하고 링크를 열면 PDF를 직접 확대·이동하면서 ROI 위치를 고를 수 있습니다. 「640 ROI 선택」 상태에서 심볼 중심을 클릭하거나 박스를 드래그하세요. 오른쪽 실제 640×640 사진을 확인한 후 **저장 버튼**을 누릅니다. 저장 전에는 학습 이미지 파일이 생기지 않습니다.

파일은 `prepared_rois`에 PNG로 저장되고, 좌표/DPI는 같은 이름의 JSON에 기록됩니다. 이 화면은 OCR/YOLO 검출을 실행하지 않습니다. 원하는 사진을 저장한 뒤 노트북으로 돌아와 2번 셀을 다시 실행하세요. 해상도와 프레임이 맞지 않으면 저장 DPI를 바꾸고 다시 선택하세요.''')
code('''from crop_launcher import start_crop_ui
from IPython.display import HTML
CROP_URL=start_crop_ui(PDF_PATH,PREPARED_ROI_DIR,port=8510)
display(HTML(f'<a href="{CROP_URL}" target="_blank">PDF에서 직접 640×640 영역 자르기 화면 열기</a>'))
print("자른 사진 저장 폴더:",PREPARED_ROI_DIR.resolve())''')
md('''## 2. 직접 만든 640×640 사진 확인

정확히 640×640인 파일만 통과합니다. 크기가 다르면 파일명을 표시하고 중단합니다. 원본 사진은 변경하지 않습니다. 파일이 없으면 먼저 위 폴더에 사진을 넣고 이 셀을 다시 실행하세요.''')
code('''ROI_FILES=prepared_roi_files(PREPARED_ROI_DIR)
print("사용자가 준비한 ROI:",len(ROI_FILES),"장")
for image_path in ROI_FILES[:12]:
    print(image_path.name)
    image,_=load_prepared_roi(image_path)
    display(image)
if not ROI_FILES:
    print("아직 ROI 사진이 없습니다. prepared_rois 폴더에 직접 자른 사진을 넣으세요.")''')
md('''## 3. 사진 안의 심볼 라벨링

**이 화면에서 PDF를 자르는 것이 아닙니다.** 드롭다운으로 이미 준비한 사진을 선택하고, 클래스 선택 후 사진 안의 심볼 박스를 드래그합니다. 여러 심볼을 모두 표시하고 검토 완료 후 저장하세요. 다음 사진으로 이동하기 전에 저장하세요. 심볼이 없는 음성 샘플도 검토 완료 후 빈 라벨로 저장할 수 있습니다.

`%matplotlib widget` 오류가 나면 0번에서 `ipympl` 설치 후 커널을 재시작하세요. 이미 LabelImg 등에서 YOLO 라벨을 만들었다면 이 셀을 건너뛰고 4번을 사용하세요.''')
code('''%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import ipywidgets as widgets

if not ROI_FILES:
    raise ValueError("2번에서 확인할 640×640 이미지가 없습니다.")
plt.close("all")
plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False
fig,ax_roi=plt.subplots(figsize=(8,8))
image_pick=widgets.Dropdown(options=[(p.name,str(p)) for p in ROI_FILES],description="사진")
class_pick=widgets.Dropdown(options=[(name,i) for i,name in enumerate(CLASSES)],description="클래스")
reviewed=widgets.Checkbox(value=False,description="이 사진의 대상 심볼 전체 라벨 검토 완료")
undo_button=widgets.Button(description="마지막 박스 취소")
save_button=widgets.Button(description="라벨 저장",button_style="success")
message=widgets.Output()
state={"image":None,"meta":None,"boxes":[],"artists":[]}

def redraw():
    for artist in state["artists"]: artist.remove()
    state["artists"]=[]
    if ax_roi.images: ax_roi.images[0].set_data(state["image"])
    else: ax_roi.imshow(state["image"])
    for box in state["boxes"]:
        x1,y1,x2,y2=box["xyxy"]
        rectangle=Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="red",linewidth=2)
        ax_roi.add_patch(rectangle)
        label=ax_roi.text(x1,y1,CLASSES[box["class_id"]],color="red")
        state["artists"].extend([rectangle,label])
    ax_roi.set_xlim(0,640);ax_roi.set_ylim(640,0)
    ax_roi.set_title(f"{Path(image_pick.value).name} / 640×640 / 박스 {len(state['boxes'])}")
    fig.canvas.draw_idle()

def load_image(_=None):
    path=Path(image_pick.value)
    image,meta=load_prepared_roi(path,group=ROI_GROUPS.get(path.name))
    boxes=[]
    manifest_path=DATA_ROOT/"annotations.json"
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["classes"]!=CLASSES: raise ValueError("저장된 클래스 순서와 CLASSES가 다릅니다.")
        for roi in manifest["rois"]:
            if roi.get("source_image")==meta["source_image"]: boxes=roi["boxes"]
    state.update(image=image,meta=meta,boxes=boxes)
    reviewed.value=False
    redraw()

def select_box(start,end):
    if start.xdata is None or end.xdata is None: return
    x1,x2=sorted([max(0,min(640,start.xdata)),max(0,min(640,end.xdata))])
    y1,y2=sorted([max(0,min(640,start.ydata)),max(0,min(640,end.ydata))])
    if x2-x1>=2 and y2-y1>=2:
        state["boxes"].append({"class_id":class_pick.value,"xyxy":[x1,y1,x2,y2]})
        reviewed.value=False
        redraw()

def undo(_):
    if state["boxes"]: state["boxes"].pop()
    reviewed.value=False
    redraw()

def save(_):
    with message:
        clear_output(wait=True)
        try:
            item=save_roi(DATA_ROOT,state["image"],state["meta"],state["boxes"],CLASSES,reviewed=reviewed.value)
            print("저장:",Path(image_pick.value).name,"심볼",len(item["boxes"]),"개")
        except Exception as exc: print("저장 안 됨:",exc)

load_image()
selector=RectangleSelector(ax_roi,select_box,useblit=False,button=[1],minspanx=2,minspany=2,spancoords="pixels")
image_pick.observe(load_image,names="value")
undo_button.on_click(undo);save_button.on_click(save)
display(widgets.HBox([image_pick,class_pick]),widgets.HBox([undo_button,save_button]),reviewed,message)
plt.show()''')
md('''## 4. 선택 사항: 이미 만든 YOLO 라벨 가져오기

`prepared_rois/roi_001.png`와 `prepared_labels/roi_001.txt`처럼 이름을 맞추세요. txt는 `class_id cx cy width height`(0~1) 형식입니다. 대상 심볼이 없는 사진은 빈 txt가 필요합니다. 빠진 라벨을 음성 샘플로 임의 처리하지 않습니다. 3번에서 라벨링했다면 아래는 `False`로 둡니다.''')
code('''IMPORT_EXISTING_LABELS=False
if IMPORT_EXISTING_LABELS:
    count=import_prepared_labels(PREPARED_ROI_DIR,PREPARED_LABEL_DIR,DATA_ROOT,CLASSES,groups=ROI_GROUPS)
    print("사용자 이미지/라벨 가져오기:",count,"쌍")''')

tail[0].source='''## 5. 데이터셋 분리 및 점검

준비한 이미지와 라벨을 train/val로 분리합니다. 같은 이미지의 복제와 `ROI_GROUPS`에 지정한 같은 위치 그룹은 함께 배치합니다. **외부에서 자른 사진의 PDF 원점은 알 수 없어 겹치는 ROI를 자동 판단할 수 없습니다.** 같은 위치를 중복해서 자른 파일은 1번에서 같은 그룹으로 지정하세요.

정답 박스 없는 파일은 먼저 라벨링하세요. 같은 PDF의 val 점수는 다른 PDF 성능이 아닙니다. 클래스가 한쪽에 빠지면 해당 클래스의 다른 위치 샘플을 더 추가하세요.'''
notebook=nbf.v4.new_notebook(cells=cells+tail,metadata=previous.metadata)
nbf.validate(notebook)
nbf.write(notebook,target)
print(target)
