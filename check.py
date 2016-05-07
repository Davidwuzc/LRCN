# -*- coding: utf-8 -*-
from animeface import AnimeFaceDataset
import chainer.serializers as s
import os
import cv2 as cv
import sys
import numpy as ap
from PIL import Image
import chainer
from chainer import cuda
import chainer.functions as F
from CNN import CNN
from chainer.functions import caffe
import chainer.serializers as s
import cPickle as pickle
import chainer.links as L
from chainer import cuda, Variable, FunctionSet, optimizers
from numpy.random import *
import six
from LSTM import LRCN
import numpy as np
import visualizer as vs

# CNNによって特徴量を取り出したデータセットを作る
cnn = CNN(data=[], target=[], gpu=-1, n_outputs=0)
cnn.load_model()
feature = cnn.feature()

print 'len(feature)', len(feature)
print 'len(feature[0])', len(feature[0])
print 'len(feature[0][0])', len(feature[0][0])
print 'len(feature[0][0][0])', len(feature[0][0][0])

for i, motion in enumerate(feature):
    vs.showPlot3(motion[0], str(i))
    print 'print ',i   
