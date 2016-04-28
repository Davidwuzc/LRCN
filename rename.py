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

PATH = './images/mp4/UCF-101/'

def get_dir_list(path):
    tmp = os.listdir(path)
    return tmp

list = get_dir_list(PATH)
for index,files in enumerate(list):
    videoPATH = PATH + files
    print 'index:',index
    print 'videoPATH:', videoPATH
    videos = get_dir_list(videoPATH)
    print videos

    for index, video in enumerate(videos):
        print video
        print video.split('.')[0]
        print videoPATH+'/'+video
        os.rename(videoPATH+'/'+video, videoPATH+'/'+video.split('.')[0])
