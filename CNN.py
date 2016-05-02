#! -*- coding: utf-8 -*-

import time
import six.moves.cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import alex
import os

class CNN:
    def __init__(self, data, target, n_outputs, gpu=-1):

        self.model = alex.Alex(n_outputs)
        self.model_name = 'alex_chainer_fc6'
        self.feature_dataset = 'feature_dataset'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.x_feature = data
        self.y_feature = target

        # lossが発散したので学習率を変更できるように
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def predict(self):

        return self.model.predict(self.x_test, self.y_test, gpu=self.gpu)

    def feature(self):
        if os.path.exists(self.feature_dataset):
            print 'os.path.exists'
            featureImage = pickle.load(open(self.feature_dataset,'rb'))
            return featureImage
        else:

            print len(self.x_feature)
            #　全部やるのは時間的に面倒なので、５つだけでやる
            #featureImage = [[] for y in range(len(self.x_feature))]
            featureImage = [[] for y in range(5)]
            payload = [[]]
            for i, motion in enumerate(self.x_feature):
                print 'motion NO.',i
                if i >= 5:
                    continue
                for j, image in enumerate(motion):
                    if len(image)==0:
			print 'skip this images'
                        continue
		    print 'payloading...'
                    payload = np.array(image, np.float32)

                    featureImage[i].append(self.model.feature(payload, gpu=self.gpu).data)

            pickle.dump(featureImage, open(self.feature_dataset, 'wb'), -1)
            return featureImage

    def train_and_test(self, n_epoch=20, batchsize=100):

        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch', epoch

            perm = np.random.permutation(self.n_train)
            sum_train_accuracy = 0
            sum_train_loss = 0

            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.x_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)

            # evaluation
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.x_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                real_batchsize = len(x_batch)

                loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)

            epoch += 1

    def dump_model(self,name):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name + name, 'wb'), -1)

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
