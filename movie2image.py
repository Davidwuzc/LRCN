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

PATH = './mp4/UCF-101/'

def get_dir_list(path):
    tmp = os.listdir(path)
    return tmp

list = get_dir_list(PATH)
for index,files in enumerate(list):
    videoPATH = PATH + files
    print 'index:',index
    print 'videoPATH:', videoPATH
    os.mkdir('./images/'+videoPATH)
    videos = get_dir_list(videoPATH)
    for index2, video in enumerate(videos):
        oneVideoPath = videoPATH+'/'+video
        os.mkdir('./images/'+oneVideoPath)
        print oneVideoPath

        cap = cv2.VideoCapture(oneVideoPath)
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
                    cv2.imwrite('./images/'+oneVideoPath+'/'+'frame'+str(count)+'.jpg', frame)
                print 'count:',count
                count = count + 1
            else:
                print 'False'
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
