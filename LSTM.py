#! -*- coding: utf-8 -*-

import chainer
import chainer.links as L

import time
import six.moves.cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import alex
from numpy.random import *



class LSTM(chainer.Chain):
    def __init__(self, length, n_outputs, n_units=256, train=True):
        super(LSTM, self).__init__(
            l0=L.Linear(length, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units,n_outputs)
        )

    def __forward(self, x):
        h0 = self.l0(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return h3

    def forward(self, x_data, y_data, train=True, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
        x, t = Variable(x_data), Variable(y_data)

        y = self.__forward(x)

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def predict(self, x_data, gpu=-1):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)

        y = self.__forward(x)

        return F.softmax(y).data

class LRCN:
    def __init__(self, data, target, length, n_outputs, gpu=-1):

        self.model = LSTM(length, n_outputs)
        self.model_name = 'LSTM_Model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.length = length
        self.dim = n_outputs

        self.x_feature = data
        self.y_feature = target

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def predict(self):
        return self.model.predict(self.x_test, self.y_test, gpu=self.gpu)

    def train_and_test(self, n_epoch=200, batchsize=100):
        epoch = 1
        for seq in range(n_epoch):
            sum_train_accuracy = 0
            sum_train_loss = 0
            randomMotion = randint(self.dim)
            sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
            
            for i, image in enumerate(sequence):

                x = image[np.newaxis, :]

                t = np.asarray([randomMotion], dtype=np.int32)

                loss, acc = self.model.forward(x, t, gpu=self.gpu)
                loss.backward()
                self.optimizer.update
                self.model.zerograds()

                sum_train_loss += float(cuda.to_cpu(loss.data))
                sum_train_accuracy += float(cuda.to_cpu(acc.data))

            if epoch%10 == 0:
                print 'train mean loss={}, accuracy={}'.format(sum_train_loss/len(sequence), sum_train_accuracy/len(sequence))
            self.model.reset_state()

            # evaluation
            if epoch%10 == 0:
                print 'epoch:  ',epoch
                sum_test_accuracy = 0
                sum_test_loss = 0
                randomMotion = randint(self.dim)
                sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
                for i, image in enumerate(sequence):

                    x = image[np.newaxis, :]
                    t = np.asarray([randomMotion], dtype=np.int32)
                    loss, acc = self.model.forward(x, t, gpu=self.gpu)
                    sum_test_loss += float(cuda.to_cpu(loss.data))
                    sum_test_accuracy += float(cuda.to_cpu(acc.data))
                print '=================================='
                print 'test mean loss={}, accuracy={}'.format(sum_test_loss/len(sequence), sum_test_accuracy/len(sequence))
                print '=================================='
            
            self.model.reset_state()

            # prediction
            if epoch%10 ==0:
                randomMotion = randint(self.dim)
                sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
                prob = np.asarray([[0. for y in range(self.dim)]])
                for i, image in enumerate(sequence):
                    x = image[np.newaxis, :]
                    result = cuda.to_cpu(self.model.predict(x, gpu=self.gpu))
                    prob = prob[0] + result[0]/len(sequence-1)
                print 'Answer: ', randomMotion
                print 'prob: ', prob

            epoch += 1



    def dump_model(self):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name, 'wb'), -1)

    def load_model(self):
        self.model = pickle.load(open(self.model_name,'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        self.optimizer.setup(self.model)

    def test(self,gpu=0):
        sum_test_loss = 0
        sum_test_accuracy = 0
        x_batch=self.x_test
        y_batch=self.y_test
        loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
        real_batchsize=len(x_batch)
        sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
        sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize
        print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
        print(real_batchsize)
        print(sum_test_accuracy)
