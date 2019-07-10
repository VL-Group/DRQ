# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import pickle as p
from operator import add

with open('./nus21/database.txt', 'r') as fp:
    all = fp.readlines()

with open('./nus21/query.txt', 'r') as fp:
    r = fp.readlines()

all = all + r

a = [i.strip().split()[1:] for i in all]

label = []

for l in a:
    label.append([int(j) for j in l])

flag = [0] * len(label)

index = np.random.permutation(len(label))

current = 0

count = 0

query = []

for i in range(21):
    while True:
        if label[index[current]][i] == 1:
            flag[index[current]] = 1
            query.append(all[index[current]])
            count += 1
        current += 1
        if count >= 100:
            count = 0
            break

remain = []

current = 0
for i in flag:
    if i == 0:
        remain.append(all[current])
    current += 1

with open('./nus_eccv/query.txt', 'w') as fp:
    fp.writelines(query)

with open('./nus_eccv/database.txt','w') as fp:
    fp.writelines(remain)


a = [i.strip().split()[1:] for i in remain]

label = []

for l in a:
    label.append([int(j) for j in l])

train = []

for i in range(20, -1, -1):
    count = 0
    while True:
        for j in range(len(label)):
            if label[j][i] == 1:
                label.remove(label[j])
                train.append(remain[j])
                remain.remove(remain[j])
                count += 1
                break
        if count >= 500:
            break

with open('./nus_eccv/train.txt','w') as fp:
    fp.writelines(train)