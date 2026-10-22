"""Add a trainable bar-only classifier branch while retaining every legacy tensor."""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from append_only_bar_head import AppendOnlyBarHead
from train import CharacterTrainer
from train_vertical_bar import OLD, OUT as DATA
from train_vertical_bar_frozen import restore_protected,preserved

OUT=Path(__file__).resolve().parent/'vertical_bar_branch'


class BranchBarTrainer(CharacterTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose)
        oldhead,newhead=weights.model[-1],model.model[-1]
        head_prefix=f'model.{len(model.model)-1}.'
        substitutions={}
        for attribute in ('cv3','one2one_cv3'):
            for i,branch in enumerate(getattr(oldhead,attribute)):
                getattr(newhead,attribute)[i]=AppendOnlyBarHead(branch)
                substitutions[head_prefix+f'{attribute}.{i}.']=head_prefix+f'{attribute}.{i}.legacy.'
        self.protected={}
        for key,value in weights.state_dict().items():
            destination=key
            for prefix,replacement in substitutions.items():
                if key.startswith(prefix):destination=key.replace(prefix,replacement,1);break
            self.protected[destination]=value.detach().cpu().clone()
        assert preserved(model,self.protected)==[]
        return model

    def _setup_train(self):
        super()._setup_train()
        for name,p in self.model.named_parameters():p.requires_grad_('.bar.' in name)
        report=dict(original_state_tensors=len(self.protected),trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    old_feature_layers_frozen=True,old_classifier_branches_frozen=True,old_batch_norm_frozen=True,new_class='|')
        (self.save_dir/'branch_training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    def _model_train(self):
        super()._model_train()
        for name,module in self.model.named_modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and '.bar.' not in name:module.eval()

    def optimizer_step(self):
        super().optimizer_step()
        restore_protected(self.model,self.protected)
        restore_protected(self.ema.ema,self.protected)

    def save_model(self):
        assert preserved(self.model,self.protected)==[]
        assert preserved(self.ema.ema,self.protected)==[]
        return super().save_model()


def main():
    torch.set_num_threads(4)
    if list((OUT/'training/weights').glob('*.pt')):raise FileExistsError('Preserve previous branch trial')
    model=YOLO(str(OLD/'training/weights/best.pt'))
    model.train(trainer=BranchBarTrainer,data=str(DATA/'data.yaml'),epochs=12,
        imgsz=384,batch=16,device=0,workers=0,project=str(OUT),name='training',exist_ok=True,seed=20261019,
        cls_remap=False,optimizer='AdamW',lr0=.0007,lrf=.2,warmup_epochs=1,patience=12,
        weight_decay=.0005,amp=False,cache=False,plots=False,mosaic=0,mixup=0,cutmix=0,copy_paste=0,
        degrees=0,translate=0,scale=0,shear=0,perspective=0,fliplr=0,flipud=0,hsv_h=0,hsv_s=0,hsv_v=0)
    checkpoint=YOLO(str(OUT/'training/weights/best.pt')).model
    expected=model.trainer.protected
    errors=preserved(checkpoint,expected)
    result=dict(original_state_tensors=len(expected),changed_original_tensors=errors,all_original_weights_preserved=not errors)
    (OUT/'preservation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert not errors,errors
    print('APPEND-ONLY BRANCH COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
