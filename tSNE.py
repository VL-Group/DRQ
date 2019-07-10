#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import os
from os import path
import numpy as np
import tensorflow as tf
from DSQ import DSQ, IMAGE_WIDTH, IMAGE_HEIGHT
from Dataset import Dataset
import sklearn.manifold
from Utils import mAP

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("Dataset", "NUS", "The preferred dataset")
tf.app.flags.DEFINE_string("Mode", "train", "train or evaluate")
tf.app.flags.DEFINE_integer("BitLength", 32, "The quantization code length")
tf.app.flags.DEFINE_integer("ClassNum", 21, "The classification class number")
tf.app.flags.DEFINE_integer("K", 256, "The centroids number")
tf.app.flags.DEFINE_integer(
    "PrintEvery", 50, "How many batches after one print")
tf.app.flags.DEFINE_float("LearningRate", 0.01, "Init learning rate")
tf.app.flags.DEFINE_integer("Epoch", 64, "How many epoches")
tf.app.flags.DEFINE_integer("BatchSize", 16, "Batch size")
tf.app.flags.DEFINE_string("Device", "0", "Device ID")
tf.app.flags.DEFINE_boolean("UseGPU", True, "Using CUDA for acceleration")
tf.app.flags.DEFINE_boolean("SaveModel", True, "Options to save in every epoch")
tf.app.flags.DEFINE_integer("R", -1, "Recall@R, -1 for all")
tf.app.flags.DEFINE_float("Lambda", 0.1, "lambda")
tf.app.flags.DEFINE_float("Tau", 1, "tau")
tf.app.flags.DEFINE_float("Mu", 1, "Mu")
tf.app.flags.DEFINE_float("Nu", 0.1, "Nu")


os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.Device)


def main(_):
    model = DSQ(FLAGS)
    a = "/device:GPU:0" if FLAGS.UseGPU else "/cpu:0"
    print("Using device:", a, "<-", FLAGS.Device)
    with tf.device(a):
        vecs = Dataset('Cifar', 'database', 256, 256, 256)
        database, codebook, scale = model.GetFeature(vecs.data)

        idx = np.random.permutation(database.label.shape[0])[:5000]
        Y = database.label[idx]
        codes = database.codes[idx]
        np.save('tSNE_Y', np.argmax(Y, axis=1))
        for i in range(4):
            X = mAP.Quantize_RQ(codes, codebook, i+1, scale)
            print(X.shape, Y.shape)
            embedded = sklearn.manifold.TSNE(perplexity=50, init='pca', method='exact').fit_transform(X)
            np.save('tSNE_X_level_' + str(i+1), embedded)
            print(i, 'saved')
        embedded = sklearn.manifold.TSNE(perplexity=50, init='pca', method='exact').fit_transform(database.output[idx])
        np.save('tSNE_X_level_raw', embedded)
        print('raw saved')


if __name__ == '__main__':
    tf.app.run()
