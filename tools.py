import numpy as np
from gensim.matutils import argsort
import pickle


class W2V(object):
    def __init__(self, index2word, syn):
        self.index2word = index2word
        self.syn = syn

    def __getitem__(self, word):
        if word in self.index2word:
            return self.syn[self.index2word.index(word)]
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def __contains__(self, word):
        return word in self.index2word

    def __len__(self):
        return len(self.index2word)

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as file:
            w2v = pickle.load(file)
        return w2v


class Lexicon(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def translate(self, word, topn=1):
        dist = np.dot(self.target.syn, self.source[word])
        best = argsort(dist, topn=topn, reverse=True)
        result = [self.target.index2word[sim] for sim in best]
        return result

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as file:
            lexicon = pickle.load(file)
        return lexicon


def extract_words(model, tag='Noun'):
    '''
    :param model: word2vec
    '''
    # model : Word2Vec object
    index2word = [word for word in model.index2word if (tag in word)]
    syn0 = [model[word] for word in index2word]
    index2word = [word.split('/')[0] for word in index2word]
    return index2word, syn0

def make_train_vec(x, y, dic):
    '''
    :param x: W2V
    :param y: W2V
    :param dic: 초기사전
    :return: 학습벡터 쌍
    '''
    source_vec = []
    target_vec = []
    for k, vs in dic.items():
        source_vec.append(np.array(x[k]))
        target_mean = [y[v] for v in vs]
        target_vec.append(np.array(target_mean).mean(axis=0))
    return np.array(source_vec), np.array(target_vec)


def pivot_train_vec(p1, p2):
    '''
    :param p1: 중간언어1의 W2V
    :param p2: 중간언어2의 W2V
    :return: 학습벡터 쌍
    '''
    s = sorted(list(set(p1.index2word).intersection(set(p2.index2word))), key=lambda x: x)
    x_train_vectors = [p1[w] for w in s]
    y_train_vectors = [p2[w] for w in s]

    return np.array(x_train_vectors), np.array(y_train_vectors)


def accuracy(test_dict, lexicon, topn=1):
    '''
    :param test_dict: 평가사전
    :param lexicon: 이중언어 사전
    :param topn: 번역후보 수
    :return: 정확률
    '''
    ac = 0
    for w in test_dict.keys():
        a = lexicon.translate(w, topn=topn)
        if set(test_dict[w]).intersection(set(a)):
            ac += 1
    return ac


def recall(test_dict, lexicon, topn=1):
    '''
    :param test_dict: 평가사전
    :param lexicon: 이중언어 사전
    :param topn: 번역후보 수
    :return: 재현율
    '''
    re = 0
    for word in test_dict.keys():
        candidate_word = lexicon.translate(word, topn=topn)
        correct_word = test_dict[word]
        b = 0
        for c in candidate_word:
            if c in correct_word:
                b += 1
        b /= len(correct_word)
        re += b
    return re
