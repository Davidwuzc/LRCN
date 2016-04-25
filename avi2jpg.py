# coding: UTF-8
import skvideo.io
import time
import numpy as np
import cv2
import sys
import os
import urllib
import urllib2
from BeautifulSoup import BeautifulSoup
import argparse

query = 'ApplyEyeMakeup'

newVideoPath = './frames/'+query
if not os.path.exists(newVideoPath):
    os.makedirs(newVideoPath)

#videoPath = "./UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
print videoPath
#cap = skvideo.io.VideoCapture('/Users/Hiroki/Code/AI/LRCN/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
#cap = skvideo.io.VideoCapture('UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
#cap = cv2.VideoCapture('UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
cap = cv2.VideoCapture('v_BenchPress_g01_c01.avi')

print cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame',frame)
    else:
        print 'False'
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
