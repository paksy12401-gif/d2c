"""Decode the immutable legacy scores independently of the newly added class."""
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class LegacyOnlyPredictor(DetectionPredictor):
    def postprocess(self,preds,img,orig_imgs,**kwargs):
        prediction=preds[0] if isinstance(preds,(tuple,list)) else preds
        assert prediction.ndim==3 and prediction.shape[1]==72, prediction.shape
        prediction=prediction.clone()
        prediction[:,71,:]=0
        return super().postprocess(prediction,img,orig_imgs,**kwargs)
