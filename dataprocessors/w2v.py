# -*- coding:utf-8 -*-


import os

from gensim.models import KeyedVectors
import numpy as np


class Embedding:

    def __init__(self, vec_path='', cache_vec_path='', logger=None):
        self.vec_path = vec_path
        self.cache_vec_path = cache_vec_path
        self.logger = logger
        self._load_vec()
        self._load_cache_vec()

    def _load_vec(self):
        if os.path.exists(self.vec_path):
            self.w2v = KeyedVectors.load(self.vec_path)
            self.logger.info("Step One: load w2v file successful!")
        else:
            # self.logger.warning("Step X: the w2v file not exists ! ")
            raise FileNotFoundError("The w2v file is not exists ! ")

    def _load_cache_vec(self):
        if os.path.exists(self.cache_vec_path):
            # self.cache_vec = zload(self.cache_vec_path)
            pass
        else:
            self.cache_vec = {}

    def new_vector(self, v):
        return v if v is not None else np.random.randn(self.w2v.vector_size)

    def __getitem__(self, word):
        v = None
        if word is None:
            return None
        if word in self.cache_vec:
            return self.cache_vec[word]
        if word in self.w2v:
            v = self.w2v[word]
        v = self.new_vector(v)
        self.cache_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.w2v

    @property
    def vector_size(self):
        return self.w2v.vector_size

    @property
    def get_vectors(self):
        pad_v = self.__getitem__('<pad>')
        unk_v = self.__getitem__('UNK')
        vectors = self.w2v.vectors
        vectors = np.concatenate((pad_v.reshape(1, -1), vectors, unk_v.reshape(1, -1)), axis=0)
        return vectors


if __name__ == '__main__':
    emb = Embedding('/Users/chuan/Project/ifchange/ssc-bot/data/embedding/w2v.model')
    print(f"vector_size: {emb.vector_size}")
