"""Check old synthetic alphanumeric examples without retraining or answer edits."""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from compare_punctuation import regression

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    results={}
    for name,path in [('original_62',ROOT/'alphanumeric/training/weights/best.pt'),
                      ('fine_tuned_68',ROOT/'fusion_iterations/spacing_training/training/weights/best.pt')]:
        results[name]=regression(YOLO(str(path)))
        print(name,results[name],flush=True)
    results['scope']='Same-template synthetic regression only; new model is not substituted for the original vector-letter reader.'
    (ROOT/'fusion_iterations/synthetic_regression.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
