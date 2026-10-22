"""Train only the | classifier rows; restore every original tensor exactly.

Frozen logits do not remove semantic ambiguity or cross-class NMS competition.
Keep the original inference branch authoritative when combining detections.
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from train import CharacterTrainer
from train_vertical_bar import OLD, OUT as DATA

OUT=Path(__file__).resolve().parent/'vertical_bar_frozen'


def new_class_gradient(gradient):
    masked=gradient.clone()
    masked[:67]=0
    return masked


def restore_protected(model, protected):
    state=model.state_dict()
    with torch.no_grad():
        for key, original in protected.items():
            target=state[key]
            value=original.to(device=target.device,dtype=target.dtype)
            if target.shape==value.shape:
                target.copy_(value)
            else:
                assert target.shape[0]==68 and value.shape[0]==67 and target.shape[1:]==value.shape[1:]
                target[:67].copy_(value)


def preserved(model, protected):
    state=model.state_dict();errors=[]
    for key, original in protected.items():
        current=state[key]
        if current.shape!=original.shape:current=current[:67]
        if not torch.equal(current.cpu(),original.cpu().to(current.dtype)):
            errors.append(key)
    return errors


class FrozenBarTrainer(CharacterTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        assert weights.names[37]=='l' and model.model[-1].nc==68
        self.protected={k:v.detach().cpu().clone() for k,v in weights.state_dict().items()}
        self.new_row_keys=[]
        with torch.no_grad():
            state=model.state_dict()
            for key,original in self.protected.items():
                target=state[key]
                if target.shape!=original.shape:
                    assert target.shape[0]==68 and original.shape[0]==67
                    target[67].copy_(target[37])
                    self.new_row_keys.append(key)
        assert len(self.new_row_keys)==12,self.new_row_keys
        assert preserved(model,self.protected)==[]
        return model

    def _setup_train(self):
        super()._setup_train()
        for name,p in self.model.named_parameters():
            p.requires_grad_(name in self.new_row_keys)
            if p.requires_grad:p.register_hook(new_class_gradient)
        assert any(p.requires_grad for p in self.model.parameters())
        report=dict(protected_state_tensors=len(self.protected),new_row_keys=self.new_row_keys,
                    effective_trainable_parameters=sum(p[67:].numel() for n,p in self.model.named_parameters() if n in self.new_row_keys),
                    batch_norm_eval=True,weight_decay=0,original_rows_restored_after_every_step=True)
        (self.save_dir/'frozen_training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

    def optimizer_step(self):
        super().optimizer_step()
        restore_protected(self.model,self.protected)
        if self.ema:restore_protected(self.ema.ema,self.protected)

    def save_model(self):
        assert preserved(self.model,self.protected)==[]
        assert preserved(self.ema.ema,self.protected)==[]
        return super().save_model()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args();torch.set_num_threads(4)
    if list((OUT/'training/weights').glob('*.pt')):raise FileExistsError('Preserve previous frozen trial')
    model=YOLO(str(OLD/'training/weights/best.pt'))
    model.train(trainer=FrozenBarTrainer,data=str(DATA/'data.yaml'),epochs=args.epochs,
        imgsz=384,batch=16,device=0,workers=0,project=str(OUT),name='training',exist_ok=True,seed=20261019,
        cls_remap=False,optimizer='AdamW',lr0=.003,lrf=.2,warmup_epochs=1,patience=12,
        weight_decay=0,amp=False,cache=False,plots=False,mosaic=0,mixup=0,cutmix=0,copy_paste=0,
        degrees=0,translate=0,scale=0,shear=0,perspective=0,fliplr=0,flipud=0,hsv_h=0,hsv_s=0,hsv_v=0)
    old=YOLO(str(OLD/'training/weights/best.pt')).model.state_dict()
    new=YOLO(str(OUT/'training/weights/best.pt')).model
    errors=preserved(new,old)
    report=dict(original_state_tensors=len(old),changed_original_tensors=errors,all_original_weights_preserved=not errors)
    (OUT/'preservation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not errors,errors
    print('FROZEN BAR TRAINING COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
