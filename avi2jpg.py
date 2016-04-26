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
import math

query = 'ApplyEyeMakeup'

#newVideoPath = './frames/'+query
newVideoPath = './frames/./UCF-101/YoYo/v_YoYo_g25_c05.avi.mp4'
if not os.path.exists(newVideoPath):
    os.makedirs(newVideoPath)

#videoPath = "./UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
#cap = skvideo.io.VideoCapture('/Users/Hiroki/Code/AI/LRCN/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
#cap = skvideo.io.VideoCapture('UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
#cap = cv2.VideoCapture('UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
cap = cv2.VideoCapture(newVideoPath)
print 'cap:', cap
totalFrame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
frame10 = math.floor(totalFrame / 10)
print 'totalFrame:', totalFrame
print 'frame10', frame10
count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if count % frame10 == 0:
            cv2.imwrite('frame'+str(count)+'.jpg', frame)
            print 'image saved!'
        cv2.imshow('frame',frame)
        print 'count:',count
        count = count + 1
    else:
        print 'False'
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
