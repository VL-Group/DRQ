# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
from gensim.models import KeyedVectors

GOOGLE_PATH = './GoogleNews-vectors-negative300.bin'
SAVE_PATH = '/hhd12306/zhuxiaosu/DSQ/data/coco/coco_wordvec.txt'
VEC_LENGTH = 300

CIFAR_LABEL = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'light',
               'hydrant', 'sign', 'parking', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'ball', 'kite', 'bat', 'glove', 'skateboard', 'surfboard', 'tennis', 'bottle', 'wine', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog',
               'pizza', 'donut', 'cake', 'chair', 'couch', 'plant', 'bed', 'table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'book', 'clock', 'vase', 'scissors', 'teddy', 'drier', 'toothbrush']


class WordVec(object):
    """docstring for WordVec."""

    def __init__(self, labels):
        self.labels = labels
        self.vectors = None

    def LoadGoogle(self):
        if path.exists(GOOGLE_PATH):
            self.model = KeyedVectors.load_word2vec_format(
                GOOGLE_PATH, binary=True)
            print("load completed")
        else:
            print("can't find model")

    def EvaluateWithLabel(self, label):
        return self.model[label]

    def GetWordVec(self):
        self.vectors = np.zeros((len(self.labels), VEC_LENGTH))

        i = 0
        for l in self.labels:
            self.vectors[i] = self.model[l]
            i += 1

        return self.vectors

    def SaveWordVec(self):
        if self.vectors is None:
            self.GetWordVec()
        np.savetxt(SAVE_PATH, self.vectors)
        print("================ Saved ================")


if __name__ == '__main__':
    w = WordVec(CIFAR_LABEL)
    w.LoadGoogle()
    w.SaveWordVec()
