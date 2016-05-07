# -*- coding: utf-8 -*-
from animeface import AnimeFaceDataset
import chainer.serializers as s
import os
import cv2 as cv
import sys
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
import numpy as np


n_epoch = 10000
gpu = 0
# データセットを作る
#dataset = AnimeFaceDataset()
#dataset.load_data_target()
#data = dataset.data
#data = np.asarray(data)

#target = dataset.target
#n_outputs = dataset.get_n_types_target()
n_outputs = 2
# CNNによって特徴量を取り出したデータセットを作る
cnn = CNN(data=[], target=[], gpu=0, n_outputs=n_outputs)
cnn.load_model()
feature = cnn.feature()
dim = len(feature)
length = len(feature[0][0][0])

class LSTM(chainer.Chain):
    def __init__(self, length, n_outputs, n_units=256, train=True):
        super(LSTM, self).__init__(
            l0=L.Linear(length, n_units),
            l1=L.LSTM(n_units, n_units),
            #l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units,n_outputs)
        )

    def __call__(self, x):
        h = self.l0(x)
        h = self.l1(F.dropout(F.relu(h), train=True))
        #h = self.l2(F.dropout(h, train=train))
        h = self.l3(F.dropout(F.relu(h), train=True))
        return h

    def __forward(self, x, train=True):
        h = self.l0(x)
        h = self.l1(F.dropout(F.relu(h), train=train))
        #h = self.l2(F.dropout(h, train=train))
        h = self.l3(F.dropout(F.relu(h), train=train))
        return h

    def forward(self, x_data, y_data, train=True, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
            print 'to gpu'
        x, t = Variable(x_data), Variable(y_data)

        y = self.__forward(x, train=train)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def reset_state(self):
        self.l1.reset_state()
        #self.l2.reset_state()

    def predict(self, x_data, gpu=-1, train=False):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)

        y = self.__forward(x, train=train)

        return F.softmax(y).data

lstm = LSTM(length, dim)
model = L.Classifier(lstm)
optimizer = optimizers.Adam()
optimizer.setup(lstm)

epoch = 1
win = 0

for seq in range(n_epoch):
	lstm.reset_state()
	model.zerograds()
	sum_train_loss = 0
	sum_train_accuracy = 0
	randomMotion = randint(dim)
	sequence = feature[randomMotion][randint(len(feature[randomMotion]))]
	for i, image in enumerate(sequence):
		#x = cuda.cupy.array(image[cuda.cupy.newaxis, :])
		x = np.asarray([image], dtype=np.float32)
		t = np.asarray([randomMotion], dtype=np.int32)
		#if gpu >= 0:
		#	x = cuda.to_gpu(x)
		#	t = cuda.to_gpu(t)
		loss = model(Variable(x), Variable(t))
		loss.backward()
		optimizer.update()

		sum_train_loss += float(cuda.to_cpu(loss.data))
        #sum_train_accuracy += float(cuda.to_cpu(acc.data))
	print '=================================='
	print 'epoch:  ',epoch
	print 'train mean loss=',format(sum_train_loss/len(sequence))
	print '==================================='
	
	if epoch % 5 == 0:
		lstm.reset_state()
		randomMotion = randint(dim)
		sequence = feature[randomMotion][randint(len(feature[randomMotion]))]
		payload = np.zeros(dim)
		for i, image in enumerate(sequence):
			x = np.asarray([image], dtype=np.float32)
			data = lstm(Variable(x)).data
			payload += data[0]/len(sequence)
		if randomMotion == np.argmax(payload):
			win += 1
		print 'Answer:', randomMotion, ' Pred:', np.argmax(payload), ',',np.max(payload)*100,'%'
		print 'softmax', payload
		print 'Total winning ratio: ', win,'/',epoch/5
		print '=================================='
	epoch += 1