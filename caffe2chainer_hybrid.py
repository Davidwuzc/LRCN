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
from LRCN import *


from chainer.functions import caffe
import chainer.serializers as s
import cPickle as pickle


def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name

#alex = caffe.CaffeFunction("bvlc_alexnet.caffemodel")
#pickle.dump(alex, open('alex','wb'))

#cnn = CNN(data= np.zeros(5),
cnn = LRCN_Hybrid(data= np.zeros(5),
          target=np.zeros(5),
          gpu=0,
          n_outputs=5)
cnn.dump_model('')
alex = pickle.load(open('alex','rb'))
hybrid = pickle.load(open('HybridModel'))
copy_model(alex, hybrid)
print 'finish load model'
cnn.dump_model('Planted')
print 'Converting caffe model to chainer model was completed!'
