#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function, division

import random
import sys

import cv2
import numpy as np

py3 = sys.version_info >= (3, 4)

NUM_CLASSES = 21
DATABASE_PATH = './data/nus21/database.txt'
TRAIN_PATH = './data/nus21/train.txt'
TEST_PATH = './data/nus21/query.txt'


def resizeX(X, w, h):
    N = X.shape[0]
    # Resize img to 256 * 256
    resized = np.zeros((N, h, w, 3))
    for i in range(N):
        resized[i] = cv2.resize(X[i], (w, h), interpolation=cv2.INTER_LANCZOS4)
    return resized


# normalize [0~255] to [-1, 1]]
def normalize(inp):
    inp /= 255.0
    inp = 2 * inp - 1.0
    return inp


class NPair_NUS(object):
    """docstring for NUS."""

    def __init__(self, mode, resizeWidth, resizeHeight, batchSize):
        if mode != "train":
            raise AttributeError("Argument of mode is invalid.")
        self._mode = mode
        self._width = resizeWidth
        self._height = resizeHeight
        self.Progress = 0
        self._batchSize = batchSize
        self.readPath()

    def readPath(self):
        self.lines = open(TRAIN_PATH, 'r').readlines()
        print("total lines: %d" % len(self.lines))

        self.DataNum = len(self.lines)
        self.ClassNum = NUM_CLASSES
        self.n_samples = self.DataNum
        self._counts = self.n_samples

        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.GroupByLabel()

    def GroupByLabel(self):
        groups = dict()
        index = 0
        for line in self.lines:
            s = line.strip().split()
            _, labels = s[0], s[1:]
            # combine to a string
            label_str = "".join(labels)
            if label_str not in groups:
                groups[label_str] = list()
            groups[label_str].append(index)
            index += 1
        # make sure all group has even value
        for key in groups.keys():
            if len(groups[key]) % 2 == 1:
                groups[key].append(random.choice(groups[key]))
        self.Group = groups

    def Read(self, index):
        if self._status:
            return resizeX(self.X[index], self._width, self._height), self.Y[index]
        else:
            ret_img = []
            ret_label = []
            for i in index:
                if i >= self.DataNum:
                    break
                try:
                    if not self._load[i]:
                        self._img[i] = cv2.resize(cv2.imread(
                            self.lines[i].strip().split()[0]), (256, 256))
                        self._label[i] = [
                            int(j) for j in self.lines[i].strip().split()[1:]]
                        self._load[i] = 1
                        self._load_num += 1
                    ret_img.append(self._img[i])
                    ret_label.append(self._label[i])
                except:
                    print('cannot open', self.lines[i])
                # else:
                # print(self.lines[i])

            if self._load_num == self.n_samples:
                self._status = 1
                self.X = np.array(self._img)
                self.Y = np.array(self._label)
                print('All images read')
                print("X:")
                print(self.X.shape)
                print("Y:")
                print(self.Y.shape)
            return resizeX(np.asarray(ret_img), self._width, self._height), np.asarray(ret_label)

    def NextBatch(self):
        anchors = []
        positives = []
        a = self.GroupHasValue
        # pick all group non empty
        random.shuffle(a)
        for c in a:
            # randomly pick anchor and positive in each group
            anchor = random.choice(self.Group[c])
            # remove selected from list
            self.Group[c].remove(anchor)
            # positive
            positive = random.choice(self.Group[c])
            # remove selected from list
            self.Group[c].remove(positive)
            anchors.append(anchor)
            positives.append(positive)

        anchorX, anchorY = self.Read(anchors)
        positiveX, _ = self.Read(positives)
        return anchorX, positiveX, anchorY

    @property
    def EpochComplete(self):
        if len(self.GroupHasValue) < self._batchSize:
            self.Group = None
            self.GroupByLabel()
            return True
        return False

    @property
    def GroupHasValue(self):
        result = []
        for key, value in self.Group.items():
            if len(value) > 0:
                result.append(key)
        random.shuffle(result)
        return result[:self._batchSize]
