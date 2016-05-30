# coding: UTF-8
import time
import numpy as np
import cv2
import sys
import os
import urllib
import urllib2
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
    os.mkdir('./images_seq/'+videoPATH)
    videos = get_dir_list(videoPATH)
    for index2, video in enumerate(videos):
        oneVideoPath = videoPATH+'/'+video
        os.mkdir('./images_seq/'+oneVideoPath)
        print oneVideoPath

        cap = cv2.VideoCapture(oneVideoPath)
        print 'cap:', cap
        totalFrame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        frame16 = math.floor(totalFrame / 16)
        for i in range(0,int(frame16)):
            print 'i',i
            os.mkdir('./images_seq/'+oneVideoPath+'/'+str(i))
        print 'totalFrame:', totalFrame
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                clip = math.floor(count / 16)

                if clip > frame16:
                    print 'frame16'
                    break

                cv2.imwrite('./images_seq/'+oneVideoPath+'/'+str(int(clip))+'/'+'frame'+str(count)+'.jpg', frame)
                print 'count:',count
                print 'clip:', clip
                count = count + 1
            else:
                print 'False'
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
