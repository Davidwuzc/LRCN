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

# データセットを作る
dataset = AnimeFaceDataset()
dataset.load_data_target()
data = dataset.data
data = np.asarray(data)

target = dataset.target
print 'target: ',target
n_outputs = dataset.get_n_types_target()
print 'n_outputs: ',n_outputs

# CNNによって特徴量を取り出したデータセットを作る
cnn = CNN(data=data, target=target, gpu=-1, n_outputs=n_outputs)
cnn.load_model()
feature = cnn.feature()

# LSTMによって動作の識別
dim = len(feature)
print 'dim',dim
# モデルの準備
print 'length', len(feature[0][0][0])
print 'length', len(feature[0][0])
print 'length', len(feature[0])
lrcn = LRCN(feature, target, len(feature[0][0][0]), dim, gpu=-1)
lrcn.train_and_test()
lrcn.dump_model()
