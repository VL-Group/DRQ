#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function, division
import os
from os import path
import numpy as np
import json
from Utils import CreateFile

INRIAPATH = './data/holiday/jpg/'

FLICKRPATH = './data/flickr/mirflickr/'

INRIATXT = './data/read/inria.json'
FLICKRTXT = './data/read/flickr.json'


def ScanINRIA():
    if not path.exists(INRIAPATH):
        raise FileNotFoundError("Please download INRIA holiday first.")
    files = os.listdir(INRIAPATH)
    return [path.join(INRIAPATH, f) for f in files]


def ScanFlickr():
    if not path.exists(INRIAPATH):
        raise FileNotFoundError("Please download INRIA holiday first.")
    files = os.listdir(INRIAPATH)
    return [path.join(INRIAPATH, f) for f in files]


def ReadINRIAandSave():
    paths = ScanINRIA()
    querys = []
    queryid = []
    ids = []
    databases = []
    for p in paths:
        if '00.jpg' in p:
            querys.append(p)
            queryid.append(int(p[-9:-6]))
        else:
            ids.append(int(p[-9:-6]))
            databases.append(p)
    inria = dict()
    inria['querys'] = querys
    inria['query_id'] = queryid
    inria['ids'] = ids
    inria['database'] = databases

    CreateFile(INRIATXT)

    with open(INRIATXT, 'w') as fp:
        json.dump(inria, fp)


def ReadFlickrandSave():
    paths = ScanFlickr()
    CreateFile(FLICKRTXT)
    with open(FLICKRTXT, 'w') as fp:
        json.dump(paths, fp)
