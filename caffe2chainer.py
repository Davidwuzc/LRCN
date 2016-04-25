from chainer.functions import caffe
import chainer.serializers as s
import cPickle as pickle
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

dataset = AnimeFaceDataset()
dataset.load_data_target()
data = dataset.data
target = dataset.target
n_outputs = dataset.get_n_types_target()

cnn = CNN(data=data,
          target=target,
          gpu=-1,
          n_outputs=n_outputs)

#cnn.train_and_test(n_epoch=20,batchsize=100)

cnn.dump_model('_chainer_fc6')
#alex = caffe.CaffeFunction("bvlc_alexnet.caffemodel")
#pickle.dump(alex, open('alex','wb'))

#print 'Converting caffe model to chainer model was completed!'
