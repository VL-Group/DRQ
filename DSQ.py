# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function, division
import os
import numpy as np
import tensorflow as tf
from Utils import PrintWithTime, BarFormat, CountVariables, Object, mAP
from Dataset import Dataset
from Encoder_VGG import Encoder_VGG
from Encoder_Alex import Encoder_Alex
import time
import shutil
from tqdm import trange, tqdm
import math

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CNN_STR = ["Alex", "VGG"]

CNN_TYPE = 0

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


class DSQ(object):
    def __init__(self, FLAG):
        self.SESSION_SAVE_PATH = "./models/{0}/{1}.ckpt"
        # used for prediction (classification)
        self._classNum = FLAG.ClassNum
        # center matrix C: [M * K * D]
        # D = U, U is the embedding layer output dimension
        self._k = FLAG.K
        # from code length get sub space count
        assert self._k != 0 and (self._k & (self._k - 1)) == 0
        perLength = int(np.asscalar(np.log2(self._k)))
        self._stackLevel = FLAG.BitLength // perLength

        PrintWithTime("Init with config:")
        print("                # Stack Levels :", self._stackLevel)
        print("                # Class Num  :", self._classNum)
        print("                # Centers K  :", self._k)

        # other settings for learning
        self._initLR = FLAG.LearningRate
        self._epoch = FLAG.Epoch
        self._batchSize = FLAG.BatchSize
        self._saveModel = FLAG.SaveModel
        self._recallatR = FLAG.R
        self._multiLabel = str.upper(FLAG.Dataset) == "NUS" or str.upper(FLAG.Dataset) == 'COCO'

        self._lambda = FLAG.Lambda
        self._tau = FLAG.Tau
        self._mu = FLAG.Mu
        self._nu = FLAG.Nu

        self._initMargin = 1.0
        self._targetMargin = 16.0
        self._windowSize = 100
        self._threshold = 0.3
        self._factor = 1.1

        if not os.path.exists('./models'):
            os.mkdir('./models')

        self.SESSION_SAVE_PATH = self.SESSION_SAVE_PATH.format(str.upper(FLAG.Dataset),
                                                               'Alex' if CNN_TYPE == 0 else 'VGG')

        # other settings for printing
        self._printEvery = FLAG.PrintEvery

        assert (FLAG.Mode == 'train' or FLAG.Mode == 'eval')

        self._train = FLAG.Mode == 'train'

        if self._train:
            # dataset
            self.Dataset = Dataset(str.upper(FLAG.Dataset), FLAG.Mode, FLAG.BatchSize,
                                   IMAGE_WIDTH, IMAGE_HEIGHT)
        self.DatasetName = FLAG.Dataset
        # tensorflow configs
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._config = config

        self.NetPQ = Encoder_Alex(str.upper(FLAG.Dataset), self._batchSize, self._classNum, self._lambda,
                                  self._stackLevel, self._k,
                                  self._multiLabel, self._train) if CNN_TYPE == 0 else Encoder_VGG(
            self._batchSize,
            self._classNum,
            self._stackLevel,
            self._k, self._train)

        self._name = "lr_{0}_epoch_{1}_batch_{2}_M_{3}_K_{4}_{5}_{6}".format(self._initLR, self._epoch, self._batchSize, self._stackLevel,
                                                                   self._k, str.upper(FLAG.Dataset),
                                                                   'Alex' if CNN_TYPE == 0 else 'VGG')

    def Inference(self):
        self.NetPQ.Inference(self.Input, self.LabelHot)

    def ApplyLoss(self):
        lr = tf.train.exponential_decay(self._initLR, global_step=self._embeddingStep, decay_steps=10000,
                                        decay_rate=0.9)
        codebook1lr = tf.train.exponential_decay(1e-4, global_step=self._codebook1Step, decay_steps=10000,
                                                 decay_rate=0.9)
        codebook2lr = tf.train.exponential_decay(1e-4, global_step=self._codebook2Step, decay_steps=10000,
                                                 decay_rate=0.9)

        print("Total var num:", CountVariables(tf.trainable_variables()))

        # Note that these are updated respectively
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        g_1 = opt.compute_gradients(self.NetPQ.Clustering,
                                    self.NetPQ.train_layers + self.NetPQ.cluster_layer)
        fc8_fcgrad, _ = g_1[-4]
        fc8_fbgrad, _ = g_1[-3]
        fc9_fcgrad, _ = g_1[-2]
        fc9_fbgrad, _ = g_1[-1]

        g_2 = opt.compute_gradients(self.NetPQ.Distinction,
                                    self.NetPQ.train_layers + self.NetPQ.distinction_layer)
        fc81_fcgrad, _ = g_2[-4]
        fc81_fbgrad, _ = g_2[-3]
        fc91_fcgrad, _ = g_2[-2]
        fc91_fbgrad, _ = g_2[-1]


        if CNN_TYPE == 0:
            self.TrainEncoder_FINE_TUNE = opt.apply_gradients(
                [((g_1[0][0] + g_2[0][0]) / 2, self.NetPQ.train_layers[0]),
                 (g_1[1][0] + g_2[1][0],
                  self.NetPQ.train_layers[1]),
                 ((g_1[2][0] + g_2[2][0]) / 2,
                  self.NetPQ.train_layers[2]),
                 (g_1[3][0] + g_2[3][0],
                  self.NetPQ.train_layers[3]),
                 ((g_1[4][0] + g_2[4][0]) / 2,
                  self.NetPQ.train_layers[4]),
                 (g_1[5][0] + g_2[5][0],
                  self.NetPQ.train_layers[5]),
                 ((g_1[6][0] + g_2[6][0]) / 2,
                  self.NetPQ.train_layers[6]),
                 (g_1[7][0] + g_2[7][0],
                  self.NetPQ.train_layers[7]),
                 ((g_1[8][0] + g_2[8][0]) / 2,
                  self.NetPQ.train_layers[8]),
                 (g_1[9][0] + g_2[9][0],
                  self.NetPQ.train_layers[9]),
                 ((g_1[10][0] + g_2[10][0]) / 2,
                  self.NetPQ.train_layers[10]),
                 (g_1[11][0] + g_2[11][0],
                  self.NetPQ.train_layers[11]),
                 ((g_1[12][0] + g_2[12][0]) / 2,
                  self.NetPQ.train_layers[12]),
                 (g_1[13][0] + g_2[13][0],
                  self.NetPQ.train_layers[13]),
                 ((fc8_fcgrad + fc81_fcgrad) * 5,
                  self.NetPQ.cluster_layer[0]),
                 ((fc8_fbgrad + fc81_fbgrad) * 10,
                  self.NetPQ.cluster_layer[1]),
                 ((fc9_fcgrad + fc91_fcgrad) * 5,
                  self.NetPQ.cluster_layer[2]),
                 ((fc9_fbgrad + fc91_fbgrad) * 10,
                  self.NetPQ.cluster_layer[3]),],
                global_step=self._embeddingStep)
        else:
            raise Exception()


        # Stage 1 Codebook Learning
        self.TrainCodebook_1 = tf.train.AdamOptimizer(learning_rate=codebook1lr).minimize(self._tau * (
                self.NetPQ._8SoftDistortion + self._mu * self.NetPQ._8HardDistortion + self._nu * self.NetPQ._8JointCenter),
                                                                                          global_step=self._codebook1Step,
                                                                                          var_list=[
                                                                                              self.NetPQ.Codebook])

        # Stage 2 Codebook Learning
        self.TrainCodebook_2 = tf.train.AdamOptimizer(learning_rate=codebook2lr).minimize(self._tau * (
                self.NetPQ.SoftDistortion + self._mu * self.NetPQ.HardDistortion + self._nu * self.NetPQ.JointCenter),
                                                                                          global_step=self._codebook2Step,
                                                                                          var_list=[
                                                                                                       self.NetPQ.Codebook] + self.NetPQ.CodebookScale)

    def InitVariables(self):
        self.Input = tf.placeholder(tf.float32, shape=[self._batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="Input")
        self.LabelHot = tf.placeholder(tf.int32, shape=[self._batchSize, self._classNum], name="Label")

        self._embeddingStep = tf.Variable(0, trainable=False, name="EmbeddingStep")
        self._codebook1Step = tf.Variable(0, trainable=False, name="Codebook1Step")
        self._codebook2Step = tf.Variable(0, trainable=False, name="Codebook2Step")

        self.Inference()
        self.ApplyLoss()

        PrintWithTime(BarFormat("Variables Inited"))

    def AddSummary(self, graph):
        # tf.summary.scalar('Semantic Loss', self.NetPQ.triplet_loss)
        # tf.summary.scalar('Classification Loss', self.NetPQ.classify)
        tf.summary.scalar('Soft Distortion', self.NetPQ.SoftDistortion)
        tf.summary.scalar('Hard Distortion', self.NetPQ.HardDistortion)
        tf.summary.scalar('JCL', self.NetPQ.JointCenter)
        tf.summary.scalar('Clustering (semantic)', self.NetPQ.Clustering)
        tf.summary.scalar('Distinction (cross entropy)', self.NetPQ.Distinction)
        tf.summary.histogram('Codebook', self.NetPQ.Codebook)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by
        # default)
        self._summary = tf.summary.merge_all()
        if os.path.exists('/tmp/' + self._name):
            shutil.rmtree('/tmp/' + self._name, ignore_errors=True)
            print('Cleared tensorboard history')
        print('Tensorboard path:')
        print('/tmp/' + self._name)
        self._writer = tf.summary.FileWriter('/tmp/' + self._name, graph=graph)

    def Train(self):

        movingMean = np.ones([self._windowSize])
        margin = self._initMargin
        getToTarget = True

        PrintWithTime(BarFormat("Training Start"))

        start = time.time()

        with tf.Session(config=self._config) as sess:
            sess.run(tf.global_variables_initializer())
            if self._saveModel:
                # Create a saver
                self._saver = tf.train.Saver()
            self.AddSummary(sess.graph)

            pbar = tqdm(total=100, ncols=50, bar_format='{percentage:3.0f}%|{bar}|{postfix}')

            j = 0
            increment = 0
            """ Pre-train stage """
            tqdm.write(BarFormat("Pre-train Stage"))
            for i in range(self._epoch // 2):
                if self._saveModel:
                    self._saver.save(sess, self.SESSION_SAVE_PATH)
                while not self.Dataset.EpochComplete:
                    j += 1
                    image, label = self.Dataset.NextBatch()
                    assert image.shape[0] == self._batchSize
                    [_, triplet, summary] = sess.run(
                        [self.TrainEncoder_FINE_TUNE, self.NetPQ.Distinction, self._summary],
                        {self.Input: image, self.LabelHot: label})

                    if not getToTarget:
                        movingMean[:-1] = movingMean[1:]
                        movingMean[-1] = triplet
                        if np.mean(movingMean) < self._threshold:
                            tqdm.write('raise margin from {0} to {1}'.format(margin, self._factor * margin))
                            margin *= self._factor
                            if margin > self._targetMargin:
                                margin = self._targetMargin
                                getToTarget = True
                            a = tf.assign(self.NetPQ.Margin, margin)
                            sess.run(a)
                            movingMean[:] = margin

                    self._writer.add_summary(summary, global_step=j)
                    if j % self._printEvery == 0:
                        # Can't simply run with (self.NetPQ.JointCenter +
                        # self.NetPQ.Distortion + self.NetPQ.QHard +
                        # self.NetPQ.QSoft)
                        # This will cause graph re-creation and variables
                        # re-allocation
                        pbar.postfix = "Epoch {0} Step {1} loss={2:.2f}".format(i, j, np.mean(
                            sess.run(self.NetPQ.loss, {self.Input: image, self.LabelHot: label})))
                        percent = (100 * (i + self.Dataset.Progress)) / self._epoch
                        pbar.update(percent - increment)
                        increment = percent

            """ Codebook learning stage 1 """
            tqdm.write(BarFormat("Codebook Learning Stage"))
            for i in range(self._epoch // 2 + 1, 3 * self._epoch // 4):
                if self._saveModel:
                    self._saver.save(sess, self.SESSION_SAVE_PATH)
                while not self.Dataset.EpochComplete:
                    j += 1
                    image, label = self.Dataset.NextBatch()
                    assert image.shape[0] == self._batchSize
                    [_, _, triplet, summary] = sess.run([self.TrainEncoder_FINE_TUNE, self.TrainCodebook_1, self.NetPQ.Distinction, self._summary],
                                               {self.Input: image, self.LabelHot: label})
                    self._writer.add_summary(summary, global_step=j)

                    if not getToTarget:
                        movingMean[:-1] = movingMean[1:]
                        movingMean[-1] = triplet
                        if np.mean(movingMean) < self._threshold:
                            tqdm.write('raise margin from {0} to {1}'.format(margin, self._factor * margin))
                            margin *= self._factor
                            if margin > self._targetMargin:
                                margin = self._targetMargin
                                getToTarget = True
                            a = tf.assign(self.NetPQ.Margin, margin)
                            sess.run(a)
                            movingMean[:] = margin

                    if j % self._printEvery == 0:
                        pbar.postfix = "Epoch {0} Step {1} loss={2:.2f}".format(i, j, np.mean(
                            sess.run(self.NetPQ.loss, {self.Input: image, self.LabelHot: label})))
                        percent = (100 * (i + self.Dataset.Progress)) / self._epoch
                        pbar.update(percent - increment)
                        increment = percent

            """ Codebook learning stage 2 """
            for i in range(3 * self._epoch // 4 + 1, self._epoch):
                if self._saveModel:
                    self._saver.save(sess, self.SESSION_SAVE_PATH)
                while not self.Dataset.EpochComplete:
                    j += 1
                    image, label = self.Dataset.NextBatch()
                    assert image.shape[0] == self._batchSize
                    if j % 2 == 0:
                        [_, _, triplet, summary] = sess.run([self.TrainEncoder_FINE_TUNE, self.TrainCodebook_1, self.NetPQ.Distinction, self._summary],
                                                   {self.Input: image, self.LabelHot: label})
                    else:
                        [_, _, triplet, summary] = sess.run([self.TrainEncoder_FINE_TUNE, self.TrainCodebook_2, self.NetPQ.Distinction, self._summary],
                                                   {self.Input: image, self.LabelHot: label})

                    if not getToTarget:
                        movingMean[:-1] = movingMean[1:]
                        movingMean[-1] = triplet
                        if np.mean(movingMean) < self._threshold:
                            tqdm.write('raise margin from {0} to {1}'.format(margin, self._factor * margin))
                            margin *= self._factor
                            if margin > self._targetMargin:
                                margin = self._targetMargin
                                getToTarget = True
                            a = tf.assign(self.NetPQ.Margin, margin)
                            sess.run(a)
                            movingMean[:] = margin

                    self._writer.add_summary(summary, global_step=j)

                    if j % self._printEvery == 0:
                        pbar.postfix = "Epoch {0} Step {1} loss={2:.2f}".format(i, j, np.mean(
                            sess.run(self.NetPQ.loss, {self.Input: image, self.LabelHot: label})))
                        percent = (100 * (i + self.Dataset.Progress)) / self._epoch
                        pbar.update(percent - increment)
                        increment = percent
            end = time.time()
            pbar.close()
            print('%d seconds for %d epochs, %d batches and %d samples' % (
                end - start, self._epoch, j, j * self._batchSize))
            PrintWithTime(BarFormat("Train Finished"))

    def Evaluate(self, queryX, queryY, dataset):
        print(self._recallatR if self._recallatR > 0 else 'all')
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)

                scale = sess.run(self.NetPQ.CodebookScale)
                print(scale)

                query = Object()
                database = Object()
                query.label = queryY

                Nq = queryX.shape[0]

                dim = self.NetPQ.X.get_shape().as_list()[1]

                query.output = np.zeros([Nq, dim], np.float32)
                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.X, {self.Input: inp})
                    query.output[i * self._batchSize:(i * self._batchSize) + num] = out[:num]

                Nb = dataset.DataNum
                database_feature = np.zeros([Nb, dim], dtype=np.float32)
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                database.codes = np.zeros([Nb, self._stackLevel], np.int32)

                start = time.time()
                print('Encoding database')
                total_db = (Nb // self._batchSize) + 1

                with trange(total_db, ncols=50) as t:
                    for i in t:
                        idx = np.arange(start=i * self._batchSize,
                                        stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                        inp, label = dataset.Get(idx)
                        num = inp.shape[0]
                        database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        out, hardCode = sess.run([self.NetPQ.X, self.NetPQ.HardCode], {self.Input: inp})
                        hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                        database.codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode,
                                                                                                   np.int32).T[
                                                                                          :num]
                        database_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]

                end = time.time()
                print('Encoding Complete')
                print('Time:', end - start)
                print('Average time for single sample:')
                print((end - start) / Nb)
                database.output = database_feature

                del dataset

                # np.save('database_codes_DSQ', codes)
                # np.save('database_codes', database.codes)
                # np.save('query_feature', query.output)
                codebook = sess.run(self.NetPQ.Codebook)
                # np.save('codebook', codebook)

                result = mAP(codebook, scale, self._recallatR if self._recallatR > 0 else database.codes.shape[0],
                             database)

                r = result.AQD_mAP(query)
                print(r)
                return r

    def GetFeature(self, queryX, queryY, dataset):
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)

                query = Object()
                database = Object()
                query.label = queryY

                Nq = queryX.shape[0]

                dim = self.NetPQ.X.get_shape().as_list()[1]

                query_feature = np.zeros([Nq, dim], np.float32)
                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.X, {self.Input: inp})
                    query_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                query.output = query_feature

                Nb = dataset.DataNum
                database_feature = np.zeros([Nb, dim], dtype=np.float32)
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                codes = np.zeros([Nb, self._stackLevel], np.int32)

                start = time.time()
                print('Encoding database')
                total_db = (Nb // self._batchSize) + 1
                with trange(total_db, ncols=50) as t:
                    for i in t:
                        idx = np.arange(start=i * self._batchSize,
                                        stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                        inp, label = dataset.Get(idx)
                        num = inp.shape[0]
                        database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        out, hardCode = sess.run([self.NetPQ.X, self.NetPQ.HardCode], {self.Input: inp})
                        hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                        codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[:num]
                        database_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]

                end = time.time()
                print('Encoding Complete')
                print('Time:', end - start)
                print('Average time for single sample:')
                print((end - start) / Nb)
                database.output = database_feature
                scale = sess.run(self.NetPQ.CodebookScale)
                # [N, M]
                database.codes = codes
                return query, database, scale

    def CheckTime(self, queryX):
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)

                inp = queryX[:self._batchSize]

                start = time.time()
                for _ in range(1000):
                    _ = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                end = time.time()
                print('total time', end - start)
                print('avg time', (end - start) / (1000 * self._batchSize))

    def EvalClassification(self, queryX, queryY):
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)
                Nq = queryX.shape[0]
                dim = self._classNum

                if self.DatasetName == 'NUS':
                    result = -1 * np.ones([Nq, dim], np.int)

                    for i in range((Nq // self._batchSize) + 1):
                        inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                        num = inp.shape[0]
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        out = sess.run(self.NetPQ.cls, {self.Input: inp})
                        for j in range(num):
                            result[i * self._batchSize + j, np.argsort(out[j])[::-1][:2]] = 1

                    checked = np.sum(np.equal(result, queryY), axis=1) > 0
                    accuracy = np.mean(checked)
                    print(accuracy)
                    return

                result = np.zeros([Nq], np.int)

                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.cls, {self.Input: inp})
                    result[i * self._batchSize:(i * self._batchSize) + num] = np.argmax(out[:num], axis=1)

                accuracy = np.mean(np.equal(result, np.argmax(queryY, axis=1)))
                print(accuracy)

    def GetRetrievalMat(self, queryX, queryY, dataset):
        self.R = self._recallatR if self._recallatR > 0 else dataset.DataNum
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)

                query = Object()
                database = Object()
                query.label = queryY

                Nq = queryX.shape[0]

                dim = self.NetPQ.X.get_shape().as_list()[1]

                query_feature = np.zeros([Nq, dim], np.float32)
                for i in range((Nq // self._batchSize) + 1):
                    inp = queryX[i * self._batchSize:(i + 1) * self._batchSize]
                    num = inp.shape[0]
                    if inp.shape[0] != self._batchSize:
                        placeholder = np.zeros(
                            [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                        inp = np.concatenate((inp, placeholder))

                    out = sess.run(self.NetPQ.X, {self.Input: inp})
                    query_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                query.output = query_feature

                Nb = dataset.DataNum
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                codes = np.zeros([Nb, self._stackLevel], np.int32)

                total_db = (Nb // self._batchSize) + 1
                with trange(total_db, ncols=50) as t:
                    for i in t:
                        idx = np.arange(start=i * self._batchSize,
                                        stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                        inp, label = dataset.Get(idx)
                        num = inp.shape[0]
                        database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        _, _ = sess.run([self.NetPQ.X, self.NetPQ.HardCode], {self.Input: inp})
                        hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                        codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[:num]

                # [N, M]
                database.codes = codes
                codebook = sess.run(self.NetPQ.Codebook)
                scale = sess.run(self.NetPQ.CodebookScale)
                # np.save('database_codes_DSQ', codes)
            db = mAP.Quantize_RQ(database.codes, codebook, 4, scale).T

            del dataset
            id_all = np.zeros([query.output.shape[0], self.R], np.int)
            retrieval_mat = np.zeros([query.output.shape[0], self.R], np.bool)
            for j in range(query.output.shape[0] // 50 + 1):
                q = query.output[j * 50:(j + 1) * 50]
                d = -np.dot(q, db)
                ids = np.argsort(d, 1)
                for i in range(d.shape[0]):
                    label = query.label[j * 50 + i, :]
                    label[label == 0] = -1
                    idx = ids[i, :]
                    imatch = np.sum(database.label[idx[0: self.R], :] == label, 1) > 0
                    id_all[j * 50 + i] = idx[:self.R]
                    retrieval_mat[j * 50 + i] = imatch[:self.R]
            np.save('retrievalMat_' + self.DatasetName, retrieval_mat)
            np.save('ids_' + self.DatasetName, id_all)
            return retrieval_mat, id_all

    def GetFeature(self, dataset):
        if os.path.exists(self.SESSION_SAVE_PATH + '.meta'):
            with tf.Session(config=self._config) as sess:
                self.InitVariables()
                self._saver = tf.train.Saver()
                self._saver.restore(sess, self.SESSION_SAVE_PATH)
                PrintWithTime("Restored model from " + self.SESSION_SAVE_PATH)
                database = Object()

                dim = self.NetPQ.X.get_shape().as_list()[1]

                Nb = dataset.DataNum
                database_feature = np.zeros([Nb, dim], dtype=np.float32)
                database.label = np.zeros([Nb, self._classNum], dtype=np.int16)

                codes = np.zeros([Nb, self._stackLevel], np.int32)

                total_db = (Nb // self._batchSize) + 1
                with trange(total_db, ncols=50) as t:
                    for i in t:
                        idx = np.arange(start=i * self._batchSize,
                                        stop=np.minimum(Nb, (i + 1) * self._batchSize), step=1)
                        inp, label = dataset.Get(idx)
                        num = inp.shape[0]
                        database.label[i * self._batchSize:(i * self._batchSize + num)] = label
                        if inp.shape[0] != self._batchSize:
                            placeholder = np.zeros(
                                [self._batchSize - inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]])
                            inp = np.concatenate((inp, placeholder))

                        out, hardCode = sess.run([self.NetPQ.X, self.NetPQ.HardCode], {self.Input: inp})
                        hardCode = sess.run(self.NetPQ.HardCode, {self.Input: inp})
                        codes[i * self._batchSize:(i * self._batchSize) + num] = np.array(hardCode, np.int32).T[:num]
                        database_feature[i * self._batchSize:(i * self._batchSize) + num] = out[:num]
                database.output = database_feature

                # [N, M]
                database.codes = codes
                codebook = sess.run(self.NetPQ.Codebook)
                scale = sess.run(self.NetPQ.CodebookScale)
            return database, codebook, scale

    def Save(self):
        with tf.Session(config=self._config) as sess:
            # Save the session
            save_path = self._saver.save(sess, self.SESSION_SAVE_PATH)
            PrintWithTime(BarFormat("Model saved"))
            PrintWithTime("Path: " + save_path)
