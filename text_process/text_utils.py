# -*- coding: utf-8 -*-
# @Time    : 18-9-28 下午1:47
# @Author  : duyongan
# @FileName: text_utils.py
# @Software: PyCharm
import re
from simple_pickle import utils
from jieba import posseg
from text_process.text import Text
import nltk
import os
import numpy as np

def text2sencents_zh(text):
    text = re.sub('\u3000|\r|\t|\xa0', '', text)
    text = re.sub('？”|！”|。”', '”', text)
    sentences = re.split("([。！？……])", text)
    sentences.append('')
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    last_sentences=[]
    for sentence in sentences:
        last_sentences+=[senten.replace('\n','').strip() for senten in sentence.split('\n\n')
                         if senten.replace('\n','').strip()]
    return last_sentences

def text2sencents_en(text):
    text = re.sub('\u3000|\r|\t|\xa0|\n', '', text)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return sentences


def sort_keys_weights(keys,weights,return_tuple=False):
    keys_weights = dict(zip(keys, weights))
    keys_weights = sorted(keys_weights.items(), key=lambda k: k[1], reverse=True)
    if return_tuple:
        return keys_weights
    keys = [term[0] for term in keys_weights]
    return keys

def text_process_zh_single(text):
    here = os.path.dirname(__file__)
    text = re.sub('\u3000|\r|\t|\xa0|\n', '', text)
    stopwords=utils.read_pickle(here+'/stopwords')
    text=posseg.lcut(text)
    text_n_list = [word_.word for word_ in text if
                   len(word_.word) > 1 and word_.word not in stopwords and
                   word_.flag in ['n','v','ns','nt','nr','ni','nl','nz',
                                  'nrf','nsf','nrj','nr1','nr2']]
    return text_n_list

def text_process_zh_not_single(text):
    here = os.path.dirname(__file__)
    stopwords = utils.read_pickle(here+'/stopwords')
    words = [tuple_ for tuple_ in list(posseg.cut(text))
             if list(tuple_)[0].strip() and len(list(tuple_)[0].strip())>1]
    words2 = []
    temp = ''
    enstart = False
    for i in range(len(words)):
        if words[i].flag in ['n','ns','nt','nr','ni','nl',
                             'nz','nrf','nsf','nrj','nr1',
                             'nr2'] and len(temp) <3 and not enstart:
            if words[i].word not in stopwords:
                temp = temp + words[i].word
            if i == len(words) - 1:
                if temp.strip() != '':
                    words2.append(temp)
        else:
            if temp.strip() != '' and not enstart:
                words2.append(temp)
                temp = ''
    return words2

def text_process_en(text):
    text = re.sub('\u3000|\r|\t|\xa0|\n', '', text)
    text = text.replace(',', ' ')
    text_list = text.split()
    texter = Text(text_list)
    text_n_list = texter.collocations()
    return text_n_list

def range_easy(a_object):
    return range(len(a_object))

def duplicate(a_list):
    return list(set(a_list))

def getKeywords_zh_single(text,num=5):
    here = os.path.dirname(__file__)
    idf_map = utils.read_pickle(here + '/idf_map')
    moshengci_weight = max(idf_map.values())
    text_n_list=text_process_zh_single(text)
    keywords_set = duplicate(text_n_list)
    keywords_count = [text_n_list.count(keyword) for keyword in keywords_set]
    keywords_weight = []
    for i, keyword in enumerate(keywords_set):
        keyword_count = keywords_count[i]
        len_keyword = len(keyword)
        try:
            idf_map[keyword]
        except:
            idf_map[keyword] = moshengci_weight
        keywords_weight.append(len_keyword * np.sqrt(keyword_count) * idf_map[keyword])
    return sort_keys_weights(keywords_set,keywords_weight)[:num]

def getKeywords_zh_not_single(text,num=5):
    here = os.path.dirname(__file__)
    idf_map = utils.read_pickle(here + '/idf_map')
    moshengci_weight = max(idf_map.values())
    text_n_list=text_process_zh_not_single(text)
    keywords_set = duplicate(text_n_list)
    keywords_count = [text_n_list.count(keyword) for keyword in keywords_set]
    keywords_weight = []
    for i, keyword in enumerate(keywords_set):
        keyword_count = keywords_count[i]
        len_keyword = len(keyword)
        try:
            idf_map[keyword]
        except:
            idf_map[keyword] = moshengci_weight
        keywords_weight.append(len_keyword * np.sqrt(keyword_count) * idf_map[keyword])
    return sort_keys_weights(keywords_set,keywords_weight)[:num]

def getKeywords_en(text,num=5):
    here = os.path.dirname(__file__)
    idf_map = utils.read_pickle(here + '/idf_map')
    moshengci_weight = max(idf_map.values())
    text_n_list=text_process_en(text)
    keywords_set = duplicate(text_n_list)
    keywords_count = [text_n_list.count(keyword) for keyword in keywords_set]
    keywords_weight = []
    for i, keyword in enumerate(keywords_set):
        keyword_count = keywords_count[i]
        len_keyword = len(keyword)
        try:
            idf_map[keyword]
        except:
            idf_map[keyword] = moshengci_weight
        keywords_weight.append(len_keyword * np.sqrt(keyword_count) * idf_map[keyword])
    return sort_keys_weights(keywords_set,keywords_weight)[:num]

def compare_two_txt(text1,text2):
    words1 = text_process_zh_single(text1)
    words2 = text_process_zh_single(text2)
    same_len=len([val for val in words1 if val in words2])
    return (same_len/len(words1)+same_len/len(words2))/2

def cos(i,j):
    return np.nan_to_num(np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j)))

class compare_bot:
    def __init__(self):
        self.__here = os.path.dirname(__file__)
        self.__single_word2vec = utils.read_pickle(self.__here + '/single_word2vec')
    def compare_two_txt_accuracy(self,text1,text2):
        words1 = getKeywords_zh_single(text1, 20)
        words2 = getKeywords_zh_single(text2, 20)
        vec1 = np.sum([self.__single_word2vec[w] for word in words1 for w in list(word)], axis=0) / 20
        vec2 = np.sum([self.__single_word2vec[w] for word in words2 for w in list(word)], axis=0) / 20
        return cos(vec1, vec2)

def compare_two_txt_accuracy(text1,text2):
    words1 = getKeywords_zh_single(text1,20)
    words2 = getKeywords_zh_single(text2,20)
    here = os.path.dirname(__file__)
    single_word2vec = utils.read_pickle(here + '/single_word2vec')
    vec1 = []
    for word in words1:
        for w in list(word):
            try:
                vec1.append(single_word2vec[w])
            except:
                pass
    vec1=np.sum(vec1,axis=0)/len(vec1)
    vec2 = []
    for word in words2:
        for w in list(word):
            try:
                vec2.append(single_word2vec[w])
            except:
                pass
    vec2=np.sum(vec2,axis=0)/len(vec2)
    return cos(vec1,vec2)

def getAbstract_zh(title,text,num=3):
    # compare_botor=compare_bot()
    sentences=text2sencents_zh(text)
    vecs_sim = []
    for sentence in sentences:
        vecs_sim.append(compare_two_txt(title, sentence))
        # vecs_sim.append(compare_botor.compare_two_txt_accuracy(title, sentence))
    abstract=sort_keys_weights(sentences,vecs_sim)[:num]
    index_num=[sentences.index(sentence) for sentence in abstract]
    abstract = sort_keys_weights(abstract, index_num)
    return ''.join(abstract)

def getAbstract_en(title,text,num=3):
    sentences=text_process_en(text)
    vecs_sim = []
    for sentence in sentences:
        vecs_sim.append(compare_two_txt(title, sentence))
    abstract=sort_keys_weights(sentences,vecs_sim)[:num]
    index_num=[sentences.index(sentence) for sentence in abstract]
    abstract = sort_keys_weights(abstract, index_num)
    return ''.join(abstract)



# title='人工智能发展未来可期北京股商安徽分公司谈多方面需正规协同合作'
# text="""
# 人工智能(AI)给现代社会带来积极的变革也许可以分为几个阶段：从可以拯救生命的自动驾驶汽车，到发现癌症治疗方法的数据分析程序，再到专门为无法说话的人设计的发声机器等等，人工智能将成为人类历史上最具革命性的创新之一。但是，要实现这一美好的愿景，仍有很长的一段路要走，而且这个过程需要持续的投入。人类社会遇到了史无前例的重大转型阶段，我们并没有一张现成的蓝图来为我们指引方向，但有一点是十分明确的：人工智能的挑战并不是一家公司、一个行业或是一个国家仅凭一己之力就能解决的，要想完全实现人工智能的美好愿景，需要整个技术生态系统和世界各国政府的通力合作。
# 　　为了实现这一愿景，产学界多年来一直在积极探索，而且一些早期的解决方案已经初见成效。各国政府和组织目前也正在积极制定战略推动人工智能的发展，来解决我们面临的一些挑战。中国、印度、英国、法国和欧盟等已经制定了正式的人工智能规划，我们需要更多国家层面的人工智能战略，最终让政府、产业届和学术界合作推动人工智能的长远发展。那么政府和行业组织要如何帮助推动人工智能发展?针对这个问题，建议优先考虑以下三点：
# 　　教育
# 　　从小学开始，学校系统在设计课程时就应该考虑到人工智能，并开发相关的教育课程。在这方面起步较早的是澳大利亚国立大学正在开发的人工智能学位课程，英特尔的资深研究员、澳大利亚计算机科学教授Genevieve Bell开创先河的设计了这门课程，我们需要看到更多这样的课程出现在学校中。学校也可以采取一些过渡措施，更好地从早期教育就鼓励实施STEM(科学、技术、工程、数学)教育。此外，例如面向数据科学家减免学费或者为他们增加更多的学位课程，将是培养更多人才的一条途径，我们急需这些人才来全面的实现人工智能效用。
# 　　另一方面，我们还要从人类本身为出发点去思考问题。比如大部分学校都教授学生基础的打字技能或者计算机技能，那么在未来的人工智能社会，学校就需要教学生学会“引导计算”技能，以便将来能够更好地利用机器去工作。因为在人工智能广泛应用的未来，很多工作肯定会实现自动化，因此不断的强化只有人类才能具备的技能也是非常至关重要的。
# 　　研发
# 　　为了制定有效的政策，要从人工智能的角度出发来采取行动。想做到这一点，最佳途径之一就是大力开展和加大研发投入。美国和欧洲等国家就正在推进关于算法可解释性的项目计划;而在英国，在政府的资助下，研发人员正在研究利用人工智能进行疾病的早期诊断、减少农作物的病害，并在公共部门提供数字化服务等等。这些做法都是值得肯定的，对于人类的发展也是多多益善的。
# 　　不同国家和行业组织间应该主动制定有效的方法来促进人类与人工智能之间的协作，以确保人工智能系统的安全性，并且应该开发可以用于人工智能训练和测试的共享公共数据集和共享环境。通过政府、产业界与学术界的互相协作，我们面临的很多的人工智能挑战都会迎刃而解。
# 　　监管环境
# 　　人工智能对法律法规体系也是有影响的。有关责任、隐私、安全和道德的法律政策数不胜数，而人工智都都可能在这些领域发挥作用，在制定法律法规之前，都需要进行周详缜密的讨论。如果单纯因为法律法规的界定而急于取缔各种形式的人工智能，这将阻碍人工智能行业的整体发展。对此，我们可以尽早采取积极措施推动数据以责任制和安全的方式被公开化，大力推动深度学习和人工智能的发展进度。
# 　　在医疗保健领域，数据的公开化将会带来很显著的影响。隐蔽掉具体身份信息后的医疗记录、基因组数据集、医疗研究和治疗计划等等都可以提供大量的数据，为人工智能提供其所需要的洞察力，帮助人类在精神健康、心血管疾病、药物治疗等方面取得突破性发现。在保护隐私和安全的前提下，如果允许研究员可以联合访问位于不同工作站的分布式存储库中的数据，这将让人工智能在人类健康建设的工程中发挥非常大的作用。
# 　　尽管我们对于人工智能的未来充满了期待，但仍然是前路漫漫。这需要政、产、学三界共同的努力，我们期待终有一天，人工智能为人类生活带来更积极的作用。
# """
# print(get_nlp_hash_code_zh(text))
# print(getAbstract_zh(title,text))
# print(compare_two_txt_accuracy(text,text))