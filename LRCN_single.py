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
from numpy.random import *

class Alex_Single(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self, length, n_outputs, n_units=1024, train=True):
        super(Alex_Single, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096,n_units),
            #l8=L.LSTM(n_units, n_units),
            fc9=L.Linear(n_units,n_outputs)
        )
        self.train = True

    def __forward(self, x, train=True):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        #h = F.relu(self.l8(h))
        h = self.fc9(h)
        return h

    def forward(self, x_data, y_data, train=True, gpu=-1):

        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
        x, t = Variable(x_data), Variable(y_data)
        y = self.__forward(x, train=train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def reset_state(self):
        print 'reset_state'
        #self.l8.reset_state()

    def predict(self, x_data, gpu=-1, train=False):
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)

        y = self.__forward(x, train=train)

        return F.softmax(y).data

class LRCN_single:
    def __init__(self, data, target, n_outputs=5, length=4096, gpu=-1):

        self.model = Alex_Single(length, n_outputs)
        self.model_name = 'SingleModelPlanted'
        self.dump_name = 'SingleModel'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.length = length
        self.dim = n_outputs

        self.x_feature = data
        self.y_feature = target

        self.optimizer = optimizers.RMSpropGraves()
        self.optimizer.setup(self.model)

    def predict(self):
        return self.model.predict(self.x_test, gpu=self.gpu)

    def train_and_test(self, n_epoch=200, batch=100):
        epoch = 1
        checkRatio = 20
        win = 0
        for seq in range(n_epoch):
            #self.model.reset_state()
            sum_train_accuracy = 0
            sum_train_loss = 0

            randomMotion1 = randint(len(self.x_feature))
            randomMotion2 = randint(len(self.x_feature[randomMotion1]))
            randomMotion3 = randint(len(self.x_feature[randomMotion1][randomMotion2]))
            sequence = self.x_feature[randomMotion1][randomMotion2][randomMotion3]
            x = []
            t = []
            for i in range(batch):
                randomMotion1 = randint(len(self.x_feature))
                randomMotion2 = randint(len(self.x_feature[randomMotion1]))
                randomMotion3 = randint(len(self.x_feature[randomMotion1][randomMotion2]))
                randomMotion4 = randint(len(self.x_feature[randomMotion1][randomMotion2][randomMotion3]))
                sequence = self.x_feature[randomMotion1][randomMotion2][randomMotion3]

                x.append(sequence[randomMotion4])
                t.append(randomMotion1)

            x = np.asarray(x, dtype=np.float32)
            t = np.asarray(t, dtype=np.int32)

            self.optimizer.zero_grads()
            loss, acc = self.model.forward(x, t, gpu=self.gpu)
            loss.backward()
            self.optimizer.update()
            print '=================='
            print epoch
            print loss.data

            sum_train_loss += float(cuda.to_cpu(loss.data))
            sum_train_accuracy += float(cuda.to_cpu(acc.data))
            
            if epoch%checkRatio == 0:
                print '=================================='
                print 'epoch:  ',epoch
                print 'train mean loss={}, accuracy={}'.format(sum_train_loss/len(sequence), sum_train_accuracy/len(sequence))

                # evaluation
                #self.model.reset_state()
                sum_test_accuracy = 0
                sum_test_loss = 0
                randomMotion1 = randint(len(self.x_feature))
                randomMotion2 = randint(len(self.x_feature[randomMotion1]))
                randomMotion3 = randint(len(self.x_feature[randomMotion1][randomMotion2]))
                sequence = self.x_feature[randomMotion1][randomMotion2][randomMotion3]
           
                for i, image in enumerate(sequence):
                    x = np.asarray(image[np.newaxis, :], dtype=np.float32)
                    t = np.asarray([randomMotion1], dtype=np.int32)
                    loss, acc = self.model.forward(x, t, gpu=self.gpu, train=True)

                    sum_test_loss += float(cuda.to_cpu(loss.data))
                    sum_test_accuracy += float(cuda.to_cpu(acc.data))
                print 'test mean loss={}, accuracy={}'.format(sum_test_loss/len(sequence), sum_test_accuracy/len(sequence))
            
            # prediction
            if epoch%checkRatio ==0:
                #self.model.reset_state()
                randomMotion1 = randint(len(self.x_feature))
                randomMotion2 = randint(len(self.x_feature[randomMotion1]))
                randomMotion3 = randint(len(self.x_feature[randomMotion1][randomMotion2]))
                randomMotion4 = randint(len(self.x_feature[randomMotion1][randomMotion2][randomMotion3]))
                sequence = self.x_feature[randomMotion1][randomMotion2][randomMotion3]

                payload = np.zeros(self.dim)
                for i, image in enumerate(sequence):
                    x = np.asarray(image[np.newaxis, :], dtype=np.float32)
                    result = cuda.to_cpu(self.model.predict(x, gpu=self.gpu, train=False))

                    payload += result[0]/len(sequence)
                if randomMotion1 == np.argmax(payload):
                    win += 1
                print 'Answer:', randomMotion1, ' Pred:', np.argmax(payload), ',',np.max(payload)*100,'%'
                print 'softmax', payload
                print 'Total winning ratio: ', win,'/',epoch/checkRatio
                print '=================================='

            epoch += 1



    def dump_model(self,name):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.dump_name+name, 'wb'), -1)

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

class LSTM(chainer.Chain):
    def __init__(self, length, n_outputs, n_units=256, train=True):
        super(LSTM, self).__init__(
            l0=L.Linear(length, n_units),
            l1=L.LSTM(n_units, n_units),
            #l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units,n_outputs)
        )

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

    def train_and_test(self, n_epoch=200):
        epoch = 1
        win = 0
        for seq in range(n_epoch):
            self.model.reset_state()
            sum_train_accuracy = 0
            sum_train_loss = 0
            randomMotion = randint(self.dim)
            sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
            for i, image in enumerate(sequence):
                x = image[np.newaxis, :]
                t = np.asarray([randomMotion], dtype=np.int32)
                
                loss, acc = self.model.forward(x, t, gpu=self.gpu)
                print acc.data
                self.optimizer.zero_grads()
                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(cuda.to_cpu(loss.data))
                sum_train_accuracy += float(cuda.to_cpu(acc.data))

            if epoch%100 == 0:
                print '=================================='
                print 'epoch:  ',epoch
                print 'train mean loss={}, accuracy={}'.format(sum_train_loss/len(sequence), sum_train_accuracy/len(sequence))

            # evaluation
                self.model.reset_state()
                sum_test_accuracy = 0
                sum_test_loss = 0
                randomMotion = randint(self.dim)
                sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
                for i, image in enumerate(sequence):

                    x = image[np.newaxis, :]
                    t = np.asarray([randomMotion], dtype=np.int32)
                    loss, acc = self.model.forward(x, t, gpu=self.gpu, train=True)
                    sum_test_loss += float(cuda.to_cpu(loss.data))
                    sum_test_accuracy += float(cuda.to_cpu(acc.data))
                print 'test mean loss={}, accuracy={}'.format(sum_test_loss/len(sequence), sum_test_accuracy/len(sequence))
            

            # prediction
            if epoch%100 ==0:
                self.model.reset_state()
                randomMotion = randint(self.dim)
                sequence = self.x_feature[randomMotion][randint(len(self.x_feature[randomMotion]))]
                #prob = np.asarray([[0. for y in range(self.dim)]])
                payload = np.zeros(self.dim)
                for i, image in enumerate(sequence):
                    x = image[np.newaxis, :]
                    result = cuda.to_cpu(self.model.predict(x, gpu=self.gpu, train=True))
                    #prob = prob[0] + result[0]/len(sequence)
                    payload += result[0]/len(sequence)
                if randomMotion == np.argmax(payload):
                    win += 1
                print 'Answer:', randomMotion, ' Pred:', np.argmax(payload), ',',np.max(payload)*100,'%'
                print 'softmax', payload
                print 'Total winning ratio: ', win,'/',epoch/100
                print '=================================='

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
