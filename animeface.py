#! -*- coding: utf-8 -*-

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv


class AnimeFaceDataset_Seq:
    def __init__(self):
        self.data_dir_path = u"../lrcn/images_seq/mp4/UCF-101/"
        #self.data_dir_path = u"./images_org/"

        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u'dataset_seq'
        self.image_size = 224
        self.index2name = []

    def get_dir_list(self, path):
        tmp = os.listdir(path)
        print tmp
        if tmp is None:
            return None

        result = sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])
        print 'result: ', result
        return result

    def get_dir_list2(self, path):
        tmp = os.listdir(path)
        print 'tmp: ', tmp
        return tmp

    def get_class_id(self, fname):
        dir_list = self.get_dir_list(self.data_dir_path)
        dir_name = filter(lambda x: x in fname, dir_list)
        print ''
        print 'get_class_id', dir_list.index(dir_name[0])
        return dir_list.index(dir_name[0])

    def load_data_target(self):
        dir_list2 = []
        if os.path.exists(self.dump_name):
            print 'os.path.exists'
            self.load_dataset()
        if self.target is None:
            dir_list = self.get_dir_list(self.data_dir_path)

            ret = {}
            self.target = []
            target_name = []
            ##self.data = [[] for y in range(len(dir_list))]
            self.data = [[] for y in range(5)]
            for i, dir_name in enumerate(dir_list):
                if i >=5:
                    continue
                dir_list2 = os.listdir(self.data_dir_path+dir_name)
                print 'dir_name', dir_name
                print 'dir_list2', dir_list2

                for j, dir_name2 in enumerate(dir_list2):
                    if dir_name2 == '.DS_Store':
                        continue
                    if j >= 30:
                        print 'more than 30 j'
                        continue
                    file_list = os.listdir(self.data_dir_path+dir_name+'/'+dir_name2)
                    self.data[i].append([])
                    for k, file_name in enumerate(file_list):
                        if k >= 3:
                            continue

                        self.data[i][j].append([])
                        file_list_seq = os.listdir(self.data_dir_path+dir_name+'/'+dir_name2+'/'+file_name)

                        payload = []
                        file_list_seq2 = []
                        for word in file_list_seq:
                            word = word.replace('frame','')
                            word = word.replace('.jpg','')
                            payload.append(int(word))
                        payload = sorted(payload)
                        for l, word in enumerate(payload):
                            file_list_seq2.append('frame'+str(payload[l])+'.jpg')

                        for file_seq in file_list_seq2:
                            root, ext = os.path.splitext(file_seq)

                            if ext == u'.jpg':
                                abs_name = self.data_dir_path+dir_name+'/'+dir_name2+'/'+file_name+'/'+file_seq
                                print 'i, j,k', i,':',j,':',k

                                self.target.append(i)
                                target_name.append(str(dir_name))
                                image = cv.imread(abs_name)
                                image = cv.resize(image, (self.image_size, self.image_size))
                                image = image.transpose(2,0,1)
                                image = image/255.
                                self.data[i][j][k].append(image)

                                print len(self.data)

        #self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        self.dump_dataset()

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name), open(self.dump_name, 'wb'), -1)

    def load_dataset(self):
        print 'dumpname',self.dump_name
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, 'rb'))

    def delete_dataset(self):
        print 'delete ', self.dump_name
        os.remove(self.dump_name)
