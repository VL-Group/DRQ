import numpy as np
import os
from pycocotools.coco import COCO
import os.path
import json
import random
import uuid


def getLabel(label):
    result = ''
    for l in label:
        result += str(l) + ' '
    return result


seed = uuid.uuid1().int
random.seed(seed)

train_json = '/hhd12306/zhuxiaosu/cocoapi/PythonAPI/instances_train2014.json'
val_json = '/hhd12306/zhuxiaosu/cocoapi/PythonAPI/instances_val2014.json'
train_prefix = '/hhd12306/chendaiyuan/Data/coco/coco_official/train2014/'
val_prefix = '/hhd12306/chendaiyuan/Data/coco/coco_official/val2014/'

coco = COCO(train_json)
val = COCO(val_json)
cats = coco.getCatIds()

all_dict = dict()

j = 0
for cat in cats:
    imgs = coco.getImgIds(catIds=[cat])
    files = coco.loadImgs(imgs)
    i = 0
    for img in files:
        fname = os.path.join(train_prefix, img['file_name'])
        if not os.path.isfile(fname):
            raise FileNotFoundError('{} not exists'.format(img))
        if imgs[i] not in all_dict:
            all_dict[imgs[i]] = {'path': fname, 'label': [0] * len(cats)}
        all_dict[imgs[i]]['label'][j] = 1
        i += 1
    j += 1

cats = val.getCatIds()
j = 0
for cat in cats:
    imgs = val.getImgIds(catIds=[cat])
    files = val.loadImgs(imgs)
    i = 0
    for img in files:
        fname = os.path.join(val_prefix, img['file_name'])
        if not os.path.isfile(fname):
            raise FileNotFoundError('{} not exists'.format(img))
        if imgs[i] not in all_dict:
            all_dict[imgs[i]] = {'path': fname, 'label': [0] * len(cats)}
        all_dict[imgs[i]]['label'][j] = 1
        i += 1
    j += 1

with open('/hhd12306/zhuxiaosu/DSQ_NUS/data/coco/all_imgs.json', 'w') as fp:
    json.dump(all_dict, fp)

all_key = list(all_dict.keys())

random.shuffle(all_key)

query_dict = list()
database_dict = list()

i = 0
for key in all_key:
    if i < 5000:
        query_dict.append(all_dict[key]['path'] + ' ' + getLabel(all_dict[key]['label']) + '\r\n')
    else:
        database_dict.append(all_dict[key]['path'] + ' ' + getLabel(all_dict[key]['label']) + '\r\n')
    i += 1

with open('/hhd12306/zhuxiaosu/DSQ_NUS/data/coco/query.txt', 'w') as fp:
    fp.writelines(query_dict)
with open('/hhd12306/zhuxiaosu/DSQ_NUS/data/coco/database.txt', 'w') as fp:
    fp.writelines(database_dict)

train_dict = list()
random.shuffle(database_dict)
i = 0
for key in database_dict:
    if i < 10000:
        train_dict.append(key)
    else:
        break
    i += 1

with open('/hhd12306/zhuxiaosu/DSQ_NUS/data/coco/train.txt', 'w') as fp:
    fp.writelines(train_dict)
