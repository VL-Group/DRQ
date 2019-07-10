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
from Dataset import Dataset
import json
import cv2

'''
import tensorflow as tf
from DSQ import DSQ, IMAGE_WIDTH, IMAGE_HEIGHT

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("Dataset", "NUS", "The preferred dataset")
tf.app.flags.DEFINE_string("Mode", "eval", "train or evaluate")
tf.app.flags.DEFINE_integer("BitLength", 32, "The quantization code length")
tf.app.flags.DEFINE_integer("ClassNum", 21, "The classification class number")
tf.app.flags.DEFINE_integer("K", 256, "The centroids number")
tf.app.flags.DEFINE_integer("PrintEvery", 50, "How many batches after one print")
tf.app.flags.DEFINE_float("LearningRate", 1e-4, "Init learning rate")
tf.app.flags.DEFINE_integer("Epoch", 64, "How many epoches")
tf.app.flags.DEFINE_integer("BatchSize", 256, "Batch size")
tf.app.flags.DEFINE_string("Device", "0", "Device ID")
tf.app.flags.DEFINE_boolean("UseGPU", True, "Batch size")
tf.app.flags.DEFINE_boolean("SaveModel", True, "Options to save in every epoch")
tf.app.flags.DEFINE_integer("R", 5000, "Recall@R, -1 for all")
tf.app.flags.DEFINE_float("Lambda", 0.1, "lambda")
tf.app.flags.DEFINE_float("Tau", 1, "tau")
tf.app.flags.DEFINE_float("Mu", 1, "Mu")
tf.app.flags.DEFINE_float("Nu", 0.1, "Nu")

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.Device)

'''
def main():
    '''
    model = DSQ(FLAGS)
    a = "/device:GPU:0" if FLAGS.UseGPU else "/cpu:0"
    print("Using device:", a, "<-", FLAGS.Device)
    with tf.device(a):
        queryX, queryY, db = Dataset.PreparetoEval(FLAGS.Dataset, IMAGE_WIDTH, IMAGE_HEIGHT)
        query, database = model.GetRetrievalMat(queryX, queryY, db)
    '''
    q_data = Dataset('NUS', 'query', 256, 256, 256)

    data = Dataset('NUS', 'database', 256, 256, 256)

    ret = np.load('retrievalMat_NUS.npy')
    ids = np.load('ids.npy')
    # dvsq_ret = np.load('DVSQ_retrieval_mat.npy')
    # dvsq_ids = np.load("DVSQ_ids.npy")


    top100 = np.sum(ret[:, :100], axis=1)

    good = np.argwhere(top100 > 80).reshape(-1)
    good = np.random.permutation(good)[:20]

    good_paths = q_data.data.ShowPath(good)
    good_results = ids[good, :25]
    result_paths = []
    for i in range(good_results.shape[0]):
        result_paths.append(data.data.ShowPath(good_results[i]))
    
    TARGET_SIZE = 80

    canvas = np.zeros([len(good_paths)*TARGET_SIZE, (26)*TARGET_SIZE+20, 3], np.uint8)
    print(canvas.shape)

    i = 0
    # draw first row, query
    for p in good_paths:
        p = p.split()[0]
        im = cv2.imread(p)
        h, w = im.shape[0], im.shape[1]
        small = h if w > h else w
        im = cv2.resize(im[(h-small)//2:(h+small)//2, (w-small)//2:(w+small)//2], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
        canvas[i*TARGET_SIZE:(i+1)*TARGET_SIZE, 0:TARGET_SIZE] = im
        i += 1
    
    # draw every result
    offset = TARGET_SIZE + 20
    for row in range(len(good_paths)):
        for col in range(len(result_paths[row])):
            p = result_paths[row][col]
            p = p.split()[0]
            im = cv2.imread(p)
            h, w = im.shape[0], im.shape[1]
            small = h if w > h else w
            im = cv2.resize(im[(h-small)//2:(h+small)//2, (w-small)//2:(w+small)//2], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
            canvas[row*TARGET_SIZE:(row+1)*TARGET_SIZE, col*TARGET_SIZE+offset:(col+1)*TARGET_SIZE+offset] = im

    cv2.imwrite('./retrieval.png', canvas)

    return

    b_left = top3 > 1
    b_right = top3 > dvsq_top3
    b_and = b_left * b_right
    better = np.argwhere(b_and).reshape(-1)
    better = np.random.permutation(better)

    bad = np.argwhere(top3 == 0).reshape(-1)
    bad = np.random.permutation(bad)

    ''' good '''
    sample_query = q_data.data.ShowPath(good[:5])
    a = ids[good[:5], :10]
    b = dvsq_ids[good[:5], :10]
    ours = []
    dvsq = []
    for i in range(a.shape[0]):
        ours.append(data.data.ShowPath(a[i]))
        dvsq.append(data.data.ShowPath(b[i]))
    result = {}
    result['query'] = sample_query
    result['ours'] = ours
    result['dvsq'] = dvsq

    with open('good.json', 'w') as fp:
        json.dump(result, fp)

    ''' better '''
    sample_query = q_data.data.ShowPath(better[:5])
    a = ids[better[:5], :10]
    b = dvsq_ids[better[:5], :10]
    ours = []
    dvsq = []
    for i in range(a.shape[0]):
        ours.append(data.data.ShowPath(a[i]))
        dvsq.append(data.data.ShowPath(b[i]))
    result = {}
    result['query'] = sample_query
    result['ours'] = ours
    result['dvsq'] = dvsq

    with open('better.json', 'w') as fp:
        json.dump(result, fp)

    ''' bad '''
    sample_query = q_data.data.ShowPath(bad[:5])
    a = ids[bad[:5], :10]
    b = dvsq_ids[bad[:5], :10]
    ours = []
    dvsq = []
    for i in range(a.shape[0]):
        ours.append(data.data.ShowPath(a[i]))
        dvsq.append(data.data.ShowPath(b[i]))
    result = {}
    result['query'] = sample_query
    result['ours'] = ours
    result['dvsq'] = dvsq

    with open('bad.json', 'w') as fp:
        json.dump(result, fp)

'''
    sample_query = q_data.data.ShowPath(highest[:5])
    q = query.output[highest[:5]]
    d = -np.dot(q, database.output.T)
    ids = np.argsort(d, 1)[:, :10]

    sample_database = []
    for i in ids:
        sample_database.append(data.data.ShowPath(i))

    result = {}

    result['query'] = sample_query
    result['database'] = sample_database

    with open('highest.txt', 'w') as fp:
        json.dump(result, fp)



    lowest = np.argwhere(top20==0).reshape(-1)
    
    lowest = np.random.permutation(lowest)

    sample_query = q_data.data.ShowPath(lowest[:5])
    q = query.output[lowest[:5]]
    d = -np.dot(q, database.output.T)
    ids = np.argsort(d, 1)[:, :10]

    sample_database = []
    for i in ids:
        sample_database.append(data.data.ShowPath(i))

    result = {}

    result['query'] = sample_query
    result['database'] = sample_database

    with open('lowest.txt', 'w') as fp:
        json.dump(result, fp)

'''


if __name__ == '__main__':
    main()
