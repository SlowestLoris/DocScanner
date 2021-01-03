from transformimage import transform
from brighten import brighten
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="image path")
args=vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image,height=500)
cv2.imshow("Orig",image)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
for i in range(8):
    gray = brighten(gray)
edged=cv2.Canny(gray,75,200)

cv2.imshow("Edged",edged)

ctrs = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
ctrs = imutils.grab_contours(ctrs)
ctrs = sorted(ctrs,key=cv2.contourArea,reverse=True)[:3]
for c in ctrs:
    peri=cv2.arcLength(c,False)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx)==4:
        screenCnt=approx
        break

cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)

warped = transform(orig,screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T=threshold_local(warped,11,offset=10,method="gaussian")
warped=(warped>T).astype("uint8")*255
cv2.imshow("Scanned",imutils.resize(warped,height=500))

cv2.waitKey()
cv2.destroyAllWindows()