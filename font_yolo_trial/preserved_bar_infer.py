"""Keep existing OCR/YOLO output intact; expose learned | candidates separately.

Automatic replacement is disabled until mixed-font ambiguity passes validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results

from prepare import read_image,write_image
from preserved_reader import merge_predictions
from train import read_rows
from train_vertical_bar_branch import OUT as TRAINED
from vertical_bar_reader import VerticalBarReader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-height-ratio',type=float,default=1.15)
    args=parser.parse_args();torch.set_num_threads(4)
    reader=VerticalBarReader(TRAINED/'training/weights/best.pt',adapt_large=True,min_height_ratio=args.min_height_ratio)
    image=read_image(args.input)
    _,baseline,detail=reader.predict(image,args.input)
    original,_=merge_predictions(baseline.boxes.cpu().numpy().data.tolist(),detail['symbols'],detail['cleaned'])
    legacy=Results(orig_img=image,path=str(args.input),names=reader.names,boxes=original)
    proposals=[r['bar'] for r in detail['vertical_bar']['accepted']]
    candidates=Results(orig_img=image,path=str(args.input),names=reader.names,boxes=np.asarray(proposals,dtype=np.float32).reshape(-1,6))
    args.output.mkdir(parents=True,exist_ok=True)
    write_image(args.output/'legacy.png',legacy.plot(line_width=1,font_size=10))
    write_image(args.output/'bar_candidates.png',candidates.plot(line_width=1,font_size=10))
    report=dict(legacy_lines=read_rows(legacy),bar_candidate_lines=read_rows(candidates),
                automatic_replacement=False,llm_inference=False,answers_loaded=False,bar_candidates=proposals,
                note='Legacy output is unchanged. Candidate symbols may still be I/l/1 in a different font; do not export them blindly.')
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
