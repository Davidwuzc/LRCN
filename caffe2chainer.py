from chainer.functions import caffe
import chainer.serializers as s
import cPickle as pickle
from animeface import AnimeFaceDataset
import chainer.serializers as s
import os
import cv2 as cv
import sys
import numpy as np
from PIL import Image
import chainer
from chainer import cuda
import chainer.functions as F
from CNN_2 import CNN


from chainer.functions import caffe
import chainer.serializers as s
import cPickle as pickle

alex = caffe.CaffeFunction("bvlc_alexnet.caffemodel")
pickle.dump(alex, open('alex','wb'))

print 'Converting caffe model to chainer model was completed!'

cnn = CNN(data= np.zeros(5),
          target=np.zeros(5),
          gpu=-1,
          n_outputs=0)

#cnn.train_and_test(n_epoch=20,batchsize=100)

cnn.dump_model('_chainer_fc6')
