# encoding=utf-8
import codecs
import os
import re
import jieba


class Segment_base:

    def __init__(self, stopwords_path=None):
        self.stopwords = {}
        self.__load_stopwords(stopwords_path)

    def __load_stopwords(self, stopwords_path=None):
        if stopwords_path is not None and os.path.exists(stopwords_path):
            with codecs.open(stopwords_path, 'r', encoding='utf-8') as f:
                for i in f.readlines():
                    self.stopwords[i.strip()] = ''
        print("stopwords loaded, the length: %d" % len(self.stopwords))

    def __call__(self, sentence, prefilter=False, ifremove=True, iflower=True):
        return self.seg(sentence, prefilter, ifremove, iflower)

    def __contains__(self, item):
        """
        description: 用于判断item 是否在停用词里面
        :param item: string
        :return: bool
        """
        return item in self.stopwords


class Segment_jieba(Segment_base):
    """
        Description:
            tokenize str into a list of word

        Input:
            - sentence: str, use jieba.cut to tokenize the sentence
            - ifremvoe: bool, check if remove stopwords from sentence, default is True
            - stopword_path: str, specify the path stopword
            - user_dict:
    """
    def __init__(self, stopwords_path=None, user_dict=None):
        Segment_base.__init__(self, stopwords_path)
        self.__load_userdict(user_dict)
        self.nlp = jieba

    def __load_userdict(self, user_dict):
        if user_dict is not None and os.path.exists(user_dict):
            jieba.load_userdict(user_dict)

    def seg(self, sentence, prefilter=False, ifremove=True, iflower=True):
        tokens = []
        if iflower:
            sentence = sentence.lower()
        # python3
        if prefilter:
            sentence = re.sub(r"[\w\d]+", " ", sentence, flags=re.ASCII)
        # pytho2
        # sentence = re.sub(r"[\w\d]+", " ", sentence)
        for x in self.nlp.cut(sentence, cut_all=False):
            x = x.strip()
            if ifremove and x in self.stopwords:
                continue
            if len(x) < 1:
                continue
            tokens.append(x)
        return {"tokens": tokens}


class Segment_hanlp(Segment_base):

    def __init__(self, stopwords_path):
        from pyhanlp import HanLP
        Segment_base.__init__(self, stopwords_path)
        self.nlp = HanLP

    def seg(self, sentence, prefilter=False, ifremove=True, iflower=True):
        tokens = []
        if iflower:
            sentence = sentence.lower()
        # python3
        if prefilter:
            sentence = re.sub(r"[\w\d]+", " ", sentence, flags=re.ASCII)
        # pytho2
        # sentence = re.sub(r"[\w\d]+", " ", sentence)
        for x in self.nlp.segment(sentence, cut_all=False):
            x = x.word.strip()
            if ifremove and x in self.stopwords:
                continue
            if len(x) < 1:
                continue
            tokens.append(x)
        return {"tokens": tokens}


if __name__ == '__main__':
    pass
