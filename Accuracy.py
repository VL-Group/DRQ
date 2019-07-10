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

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("Dataset", "NUS", "The preferred dataset")
tf.app.flags.DEFINE_string("Mode", "eval", "train or evaluate")
tf.app.flags.DEFINE_integer("BitLength", 32, "The quantization code length")
tf.app.flags.DEFINE_integer("ClassNum", 21, "The classification class number")
tf.app.flags.DEFINE_integer("K", 256, "The centroids number")
tf.app.flags.DEFINE_integer(
    "PrintEvery", 50, "How many batches after one print")
tf.app.flags.DEFINE_float("LearningRate", 1e-4, "Init learning rate")
tf.app.flags.DEFINE_integer("Epoch", 64, "How many epoches")
tf.app.flags.DEFINE_integer("BatchSize", 256, "Batch size")
tf.app.flags.DEFINE_string("Device", "0", "Device ID")
tf.app.flags.DEFINE_boolean("UseGPU", True, "Batch size")
tf.app.flags.DEFINE_boolean("SaveModel", True, "Options to save in every epoch")
tf.app.flags.DEFINE_integer("R", 5000, "Recall@R, -1 for all")

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.Device)


def main(_):
    model = DSQ(FLAGS)
    a = "/device:GPU:0" if FLAGS.UseGPU else "/cpu:0"
    print("Using device:", a, "<-", FLAGS.Device)
    with tf.device(a):
        queryX, queryY, db = Dataset.PreparetoEval(FLAGS.Dataset, IMAGE_WIDTH, IMAGE_HEIGHT)
        model.EvalClassification(queryX, queryY)

        


if __name__ == '__main__':
    tf.app.run()

