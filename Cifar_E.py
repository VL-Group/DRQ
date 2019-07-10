# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import pickle as p
import cv2

import sys

py3 = sys.version_info >= (3, 4)

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10

TRAINX = "./cifar_eccv/train_x.npy"
TRAINY = "./cifar_eccv/train_y.npy"
TESTX = "./cifar_eccv/test_x.npy"
TESTY = "./cifar_eccv/test_y.npy"
DATABASEX = "./cifar_eccv/data_x.npy"
DATABASEY = "./cifar_eccv/data_y.npy"

LABEL_NAME = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", "frog", "horse", "ship", "truck"]


def LoadCifarFile(filename):
    with open(filename, 'rb') as f:
        if py3:
            datadict = p.load(f, encoding='latin-1')
        else:
            datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


class CIFAR(object):
    """docstring for CIFAR."""

    def __init__(self):
        self.TrainName = ["data_batch_1", "data_batch_2",
                          "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
        self.DataFolder = "./cifar-10-batches-py"

    def CalcMean(self):
        imgs = list()
        labels = list()

        filenames = [path.join(self.DataFolder, name)
                     for name in self.TrainName]
        for f in filenames:
            x, y = LoadCifarFile(f)
            imgs.append(x)
            labels.append(y)

        imgs = np.array(imgs)
        labels = np.array(labels)

        # [60000, 32, 32, 3]
        X = imgs.reshape(
            (-1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        Y = labels.reshape((-1))

        r = np.mean(X[:, :, :, 0])
        g = np.mean(X[:, :, :, 1])
        b = np.mean(X[:, :, :, 2])

        print(r)
        print(g)
        print(b)

    def ReadCifar(self):
        imgs = list()
        labels = list()

        filenames = [path.join(self.DataFolder, name)
                     for name in self.TrainName]
        for f in filenames:
            x, y = LoadCifarFile(f)
            imgs.append(x)
            labels.append(y)

        imgs = np.array(imgs)
        labels = np.array(labels)

        X = imgs.reshape(
            (-1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        Y = labels.reshape((-1))

        idx = np.random.permutation(X.shape[0])

        X = X[idx]
        Y = Y[idx]

        a = np.zeros((10, 6000, 32, 32, 3), dtype=int)

        count = [0] * 10

        for i in range(X.shape[0]):
            a[Y[i], count[Y[i]]] = X[i]
            count[Y[i]] += 1

        print(a.shape)

        self._queryX = np.zeros((10000, 32, 32, 3), dtype=int)
        self._queryY = np.zeros(10000, dtype=int)
        self._trainX = np.zeros((50000, 32, 32, 3), dtype=int)
        self._trainY = np.zeros(50000, dtype=int)
        self._dataX = np.zeros((50000, 32, 32, 3), dtype=int)
        self._dataY = np.zeros(50000, dtype=int)

        for i in range(a.shape[0]):
            index = np.random.permutation(a.shape[1])
            self._queryX[1000 * i:1000 * (i + 1)] = a[i, index[:1000]]
            self._trainX[5000 * i:5000 * (i + 1)] = a[i, index[1000:]]

            self._queryY[1000 * i:1000 * (i + 1)] = i
            self._trainY[5000 * i:5000 * (i + 1)] = i

        # self.Check(self._queryX, self._queryY)
        # self.Check(self._trainX, self._trainY)
        # self.Check(self._dataX, self._dataY)

        print(self._queryX.shape)
        print(self._trainX.shape)
        print(self._dataX.shape)

        np.save(TESTX, self._queryX)
        np.save(TESTY, self._queryY)
        np.save(TRAINX, self._trainX)
        np.save(TRAINY, self._trainY)
        np.save(DATABASEX, self._trainX)
        np.save(DATABASEY, self._trainY)

        print("done")

    def Check(self, X, y):
        rnd_idx = np.random.permutation(X.shape[0])
        rnd_idx[15] = X.shape[0] - 1

        i = 0

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 4)

        for ax in axes.ravel():
            ax.imshow(X[rnd_idx[i]].astype(int))
            ax.set_title(LABEL_NAME[y[rnd_idx[i]]])
            i += 1
        plt.show()
