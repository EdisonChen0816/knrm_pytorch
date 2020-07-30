# -*- coding:utf-8


import os
import codecs

import yaml


def loadyaml(path):
    '''
     Read the config file with yaml
    :param path: the config file path
    :return: bidict
    '''
    doc = []
    if os.path.exists(path):
        with codecs.open(path, 'r') as yf:
            doc = yaml.safe_load(yf)
    return doc
