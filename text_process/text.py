# -*- coding: utf-8 -*-
# @Time    : 18-9-26 上午10:23
# @Author  : duyongan
# @FileName: text.py
# @Software: PyCharm
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk import pos_tag
import re

class Text():
    def __init__(self, tokens):
        self.tokens = tokens
        self.ignored_words = stopwords.words('english')

    def collocations(self, duanyu_num=20, window_size=2):
        finder = BigramCollocationFinder.from_words(self.tokens, window_size)
        finder.apply_freq_filter(2)
        finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in self.ignored_words)
        bigram_measures = BigramAssocMeasures()
        self._collocations = finder.nbest(bigram_measures.likelihood_ratio, duanyu_num)
        cizus=[w1 + ' ' + w2 for w1, w2 in self._collocations]
        tag_word = pos_tag(self.tokens)
        tag_word_map = dict(tag_word)
        cizu_NN = []
        for cizu in cizus:
            flag = True
            for word in cizu.split():
                if tag_word_map[word] not in ['NN', 'NNS', 'NNP', 'NNPS'] \
                        or word.find('.') != -1 or word in self.ignored_words:
                    flag = False
            if flag:
                cizu_NN.append(re.sub('\.|\?|!|…', '', cizu))
        text = list(tag_word)
        text_n_list = [re.sub('\.|\?|!|…', '', word_[0]) for word_ in text if
                       len(word_[0]) > 4 and word_[0] not in self.ignored_words and
                       word_[1] in ['NN','NNS','NNP','NNPS'] and word_[0].find('.')==-1]
        text_n_list=text_n_list+cizu_NN
        return text_n_list