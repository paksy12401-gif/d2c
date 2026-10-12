"""Locate capture context in the original drawing, locally, for leakage checks."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
WORK=HERE/'capture_split_work_20261008'
SOURCE=HERE/'capture_box_review_20261008'

def canonical(a):
    a=a.astype(np.int16)
    bg=np.median(a.reshape(-1,3),axis=0)
    gray=np.max(abs(a-bg),axis=2)
    return np.clip(gray*3,0,255).astype(np.uint8)

def main():
    cv2.setNumThreads(4)
    pdf=np.asarray(Image.open(WORK/'pdf_drawing_144dpi.png').convert('RGB'))
    # Detection contrast: light ink has no less importance than filled text.
    pdf_gray=canonical(pdf)
    sift=cv2.SIFT_create(nfeatures=50000,contrastThreshold=.015,edgeThreshold=15)
    pk,pd=sift.detectAndCompute(pdf_gray,None)
    print('pdf_features',len(pk),flush=True)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=80))
    matcher.add([pd]);matcher.train()
    data=json.loads((SOURCE/'proposals.json').read_text(encoding='utf-8'))
    rows=[]
    for c in data['captures']:
        a=np.asarray(Image.open(SOURCE/'images'/c['filename']).convert('RGB'))
        gray=canonical(a)
        ck,cd=sift.detectAndCompute(gray,None)
        candidates=[]
        if cd is not None:
            for pair in matcher.knnMatch(cd,k=2):
                if len(pair)==2 and pair[0].distance<.72*pair[1].distance:candidates.append(pair[0])
        result={'capture_id':c['capture_id'],'matches':len(candidates),'inliers':0,'located':False}
        if len(candidates)>=8:
            src=np.float32([ck[m.queryIdx].pt for m in candidates])
            dst=np.float32([pk[m.trainIdx].pt for m in candidates])
            matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                ransacReprojThreshold=3,maxIters=5000,confidence=.999,refineIters=20)
            if matrix is not None:
                matrix=matrix/2  # Save locations in original PDF point units.
                count=int(inliers.sum());scale=float(np.hypot(*matrix[:,0]));angle=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
                source_inliers=src[inliers.ravel()>0]
                span=np.ptp(source_inliers,axis=0) if len(source_inliers) else [0,0]
                corners=np.array([[0,0],[c['width'],0],[c['width'],c['height']],[0,c['height']]],dtype=float)
                transformed=corners@matrix[:,:2].T+matrix[:,2]
                ratio=count/len(candidates)
                located=count>=12 and ratio>=.35 and .04<scale<10 and abs(angle)<2 and max(span)>60
                result.update(inliers=count,inlier_ratio=round(ratio,3),scale=scale,angle=angle,
                    located=bool(located),matrix=matrix.tolist(),source_inlier_span=list(map(float,span)),
                    pdf_rect_xyxy=[float(transformed[:,0].min()),float(transformed[:,1].min()),
                                   float(transformed[:,0].max()),float(transformed[:,1].max())])
        rows.append(result)
        print(c['capture_id'],result['located'],result['inliers'],flush=True)
    (WORK/'capture_locations_144.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    overview=Image.fromarray(pdf).copy();draw=ImageDraw.Draw(overview)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for row in rows:
        if not row['located']:continue
        box=[v*2 for v in row['pdf_rect_xyxy']];draw.rectangle(box,outline='magenta',width=2);draw.text(tuple(box[:2]),row['capture_id'],fill='magenta',font=font)
    overview.save(WORK/'location_overview_144.png')
    print('located',sum(r['located'] for r in rows),'of',len(rows),flush=True)

if __name__=='__main__':main()
