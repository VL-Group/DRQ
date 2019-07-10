#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import pickle as p
import cv2
from Utils import ProgressBar
import sys
from Reader import INRIATXT, FLICKRTXT, ReadFlickrandSave, ReadINRIAandSave
import json
from Utils import RandInt
import math

py3 = sys.version_info >= (3, 4)

# INRIA holiday Dataset description
# The Holidays dataset is a set of images which mainly contains some of our personal holidays photos.
# The remaining ones were taken on purpose to test the robustness to various attacks:
# rotations, viewpoint and illumination changes, blurring, etc.
# The dataset includes a very large variety of scene types (natural, man-made, water and fire effects, etc)
# and images are in high resolution. The dataset contains 500 image groups,
# each of which represents a distinct scene or object.
# The first image of each group is the query image and the correct retrieval results are the other images of the group.


NUM_CLASSES = 501


class Inria(object):
    """docstring for Inria."""

    def __init__(self, mode, resizeWidth, resizeHeight, smallWidth, smallHeight):
        print(mode)
        if (mode != "database" and mode != "train" and mode != "test" and mode != "all"):
            raise AttributeError("Argument of mode is invalid.")
        self._mode = mode
        self._width = resizeWidth
        self._height = resizeHeight
        self._smallWidth = smallWidth
        self._smallHeight = smallHeight
        self.preparePaths()
        self._counts = self.X.shape[0]

    def preparePaths(self):
        if not path.exists(INRIATXT):
            ReadINRIAandSave()
        if not path.exists(FLICKRTXT):
            ReadFlickrandSave()

        with open(INRIATXT, 'r') as fp:
            inria = json.load(fp)
        with open(FLICKRTXT, 'r') as fp:
            flickr = json.load(fp)

        flickr_id = [500] * len(flickr)

        if self._mode == "all":
            self.X = np.array(inria['querys'] + inria['database'] + flickr)
            self.Y = np.array(inria['query_id'] + inria['ids'] + flickr_id)
            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return
        if self._mode == "database":
            self.X = np.array(inria['database'] + flickr)
            self.Y = np.array(inria['ids'] + flickr_id)

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)
            return
        if self._mode == "train":
            self.X = np.array(inria['database'] + flickr)
            self.Y = np.array(inria['ids'] + flickr_id)

            np.random.seed(541)

            choice = np.random.permutation(self.X.shape[0])

            self.X = self.X[choice[:5000]]
            self.Y = self.Y[choice[:5000]]

            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)

        else:
            self.X = np.array(inria['querys'])
            self.Y = np.array(inria['query_id'])
            self.DataNum = self.X.shape[0]
            self.ClassNum = NUM_CLASSES
            self.n_samples = self.DataNum

            self.Onehot()
            print("Label shape:", self.Y.shape)
            print("Data shape:", self.X.shape)

    def Onehot(self):
        # one-hot encoding
        y = np.zeros((self.X.shape[0], NUM_CLASSES), dtype=int)
        y[range(self.X.shape[0]), self.Y] = 1
        self.Y = y

    @staticmethod
    def resizeX(image, width, height):
        # Resize img, scale with same ratio to the smalle scale, then crop to w, h
        # for example, a 1024*768 image resize to 640*640, if scale down to 640*480,
        # it will be cropped, so firstly scale down to 853.33333 * 640, then crop to 640*640
        h, w, c = image.shape
        hRatio = height / h
        wRatio = width / w
        scale = max(hRatio, wRatio)
        # print(hRatio, wRatio)
        reshaped = cv2.resize(image, (int(round(h * scale)), int(round(w * scale))))
        nh, nw, c = reshaped.shape
        # print(reshaped.shape)
        hRemain = RandInt(nh - height)
        wRemain = RandInt(nw - width)
        # print(hRemain, hRemain + height)
        return reshaped[hRemain:(hRemain + height), wRemain:(wRemain + width)]

    # normalize [0~255] to [-1, 1]]
    def normalize(self, inp):
        return inp / 127.5 - 1.0

    def Get(self, index):
        paths = self.X[index]
        results = np.zeros((index.shape[0], self._height, self._width, 3))
        resultsSmall = np.zeros((index.shape[0], self._smallHeight, self._smallWidth, 3))
        j = 0
        for i in paths:
            x = cv2.imread(i)
            results[j] = self.resizeX(x, self._width, self._height)
            resultsSmall[j] = self.resizeX(x, self._smallWidth, self._smallHeight)

        return results, self.normalize(resultsSmall), self.Y[index]

    @property
    def SamplesCount(self):
        return self._counts
