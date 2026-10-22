"""Compare normalized stroke parts to supplied exemplars, without lexical correction."""
from pathlib import Path

import cv2
import numpy as np

from stroke_parts_probe import normalized_skeleton


def stroke_features(image):
    skeleton=normalized_skeleton(image)
    ink=(skeleton>0).astype(np.float32)
    yy,xx=np.indices(ink.shape,dtype=np.float32)
    def local(a):return cv2.boxFilter(a,-1,(9,9),normalize=False)
    count=np.maximum(local(ink),1.)
    mx,my=local(ink*xx)/count,local(ink*yy)/count
    vx,vy=local(ink*xx*xx)/count-mx*mx,local(ink*yy*yy)/count-my*my
    cov=local(ink*xx*yy)/count-mx*my
    angle=.5*np.arctan2(2*cov,vx-vy)
    channels=(np.rint(angle/np.pi*4).astype(int)%4).astype(np.int32)
    ys,xs=np.where(ink>0)
    maps=[]
    for channel in range(4):
        selected=(ink>0)&(channels==channel)
        maps.append(cv2.distanceTransform((~selected).astype(np.uint8),cv2.DIST_L2,3)
                    if selected.any() else np.full(ink.shape,40.,np.float32))
    maps=np.stack(maps)
    # Nearby directions tolerate raster stair steps, but do not equate diagonal/vertical strokes.
    distance=np.minimum.reduce([maps,np.roll(maps,1,axis=0)+2.5,
                                np.roll(maps,-1,axis=0)+2.5,np.roll(maps,2,axis=0)+6.])
    return dict(skeleton=skeleton,ys=ys,xs=xs,channels=channels[ys,xs],distance=distance)


def exemplar_images(directory):
    result={}
    for path in sorted(Path(directory).glob('*.png')):
        if int(path.stem.split('_')[0])>=62:continue
        rgba=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim!=3 or rgba.shape[2]!=4:raise ValueError(str(path))
        image=np.repeat((255-rgba[:,:,3])[:,:,None],3,axis=2)
        result[path.stem.split('_',1)[1]]=cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
    if len(result)!=62:raise ValueError('Expected 62 distinct exemplars')
    return result


class StrokeReader:
    def __init__(self,images):
        self.names=list(images)
        self.features=[stroke_features(images[name]) for name in self.names]
        self.maps=np.stack([f['distance'] for f in self.features])

    def predict(self,image,proposed=None):
        q=stroke_features(image)
        if not len(q['xs']):return dict(text='',distance=100.,margin=0.,proposed_distance=100.,alternatives=[])
        distances=self.maps[:,q['channels'],q['ys'],q['xs']]
        left,right=q['xs'].min(),q['xs'].max()
        edge=(q['xs']<=left+(right-left)*.07)|(q['xs']>=right-(right-left)*.07)
        # Touching letters share stems at crop edges. Expected template strokes remain mandatory.
        weights=np.where(edge,.1,1.)
        forward=(distances*weights).mean(axis=1)
        backward=np.array([q['distance'][f['channels'],f['ys'],f['xs']].mean()
                           for f in self.features])
        scores=(forward+backward)/2
        order=np.argsort(scores)
        first,second=map(int,order[:2])
        return dict(text=self.names[first],distance=float(scores[first]),
                    margin=float(scores[second]-scores[first]),
                    proposed_distance=float(scores[self.names.index(proposed)]) if proposed in self.names else None,
                    alternatives=[dict(text=self.names[int(i)],distance=float(scores[i])) for i in order[:5]])


def distorted(image,height,thickness):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ys,xs=np.where(gray<230)
    crop=gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    resized=cv2.resize(crop,(max(2,round(crop.shape[1]*height/crop.shape[0])),height))
    if thickness>1:resized=cv2.erode(resized,np.ones((thickness,thickness),np.uint8))
    resized=cv2.copyMakeBorder(resized,6,6,6,6,cv2.BORDER_CONSTANT,value=255)
    return cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)


def calibrate(reader,images):
    rows=[];limits={}
    for name,image in images.items():
        distances=[]
        for height in (24,40,56,80):
            for thickness in (1,2):
                value=reader.predict(distorted(image,height,thickness),proposed=name)
                rows.append(dict(reference=name,height=height,thickness=thickness,**value))
                distances.append(value['proposed_distance'])
        limits[name]=float(np.quantile(distances,.99)+1.)
    return limits,rows
