# -*- coding: utf-8 -*-
from animeface import AnimeFaceDataset_Seq
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
from LRCN_single import *
import numpy as np

motions = 5


dataset = AnimeFaceDataset_Seq()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.get_n_types_target()

# モデルの準備
lrcn = LRCN_single(data, target, n_outputs=n_outputs, gpu=0)
print 'load model...'
#lrcn.load_model()
print 'train and test...'
lrcn.train_and_test(n_epoch=100, batch=50)
lrcn.dump_model('_after')

