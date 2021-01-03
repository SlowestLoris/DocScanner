import numpy as np
import cv2

def brighten(image):
    bright=np.zeros(image.shape,image.dtype)
    alpha=1.1
    beta=-20
    bright=cv2.convertScaleAbs(image,alpha=alpha,beta=beta)
    return bright