from gensim.matutils import unitvec
from scipy.linalg import eigh
import numpy as np
from tools import pivot_train_vec, W2V, Lexicon


def calc_CCA(train_vec, reg, n_components):
    '''
    :param train_vec: 학습벡터 쌍
    :param reg: 정규화 매개변수
    :param n_components: 투영할 벡터공간의 차원
    :return: 두 투영행렬
    '''
    X1, X2 = np.array(train_vec[0]), np.array(train_vec[1])
    N1, N2 = X1.shape[1], X2.shape[1]

    Cxx = np.dot(X1.T, X1)
    Cxx = Cxx + reg * np.identity(Cxx.shape[0])  # regularized covariance
    Cxy = np.dot(X1.T, X2)  # np.cov(X, Y) #C[0:sx, sx+1:sx+sy]
    Cyx = Cxy.T
    Cyy = np.dot(X2.T, X2)
    Cyy = Cyy + reg * np.identity(Cyy.shape[0])  # regularized covariance

    LH = np.zeros((N1 + N2, N1 + N2))
    RH = np.zeros((N1 + N2, N1 + N2))
    LH[0:N1, N1:N1 + N2] = Cxy
    LH[N1:N1 + N2, 0:N1] = Cyx
    RH[0:N1, 0:N1] = Cxx
    RH[N1:N1 + N2, N1:N1 + N2] = Cyy

    LH /= 2
    RH /= 2

    r, Vs = eigh(LH, RH, eigvals=(N1 + N2 - n_components, N1 + N2 - 1))
    r = np.real(r)
    idx = r.argsort()[::-1]
    Vs = Vs[:, idx]

    U = Vs[0:N1:, 0:n_components]
    V = Vs[N1:N1 + N2:, 0:n_components]

    return U, V

class CCA(object):

    def __init__(self, reg=0.001, n_components=10):
        self.reg = reg
        self.n_components = n_components

    def train(self, train_vec):
        self.U, self.V = calc_CCA(train_vec, self.reg, self.n_components)
        return self

    def transform(self, x, y):
        return np.dot(x, self.U), np.dot(y, self.V)


def project_CCA(X1, X2, train_vec, reg=0.001, n_components=100):
    '''
    :param X1: W2V
    :param X2: W2V
    :param train_vec: X1과 X2의 학습벡터 쌍
    :param reg: 정규화 매개변수
    :param n_components: 투영할 벡터공간의 차원
    :return: 투영된 언어 쌍
    '''
    model = CCA(reg, n_components)
    model.train(train_vec)

    X1_syn, X2_syn = model.transform(X1.syn, X2.syn)
    X1_syn = np.apply_along_axis(unitvec, 1, X1_syn)
    X2_syn = np.apply_along_axis(unitvec, 1, X2_syn)
    return W2V(X1.index2word, X1_syn), W2V(X2.index2word, X2_syn)


def make_lecxicon_CCA(source, pivot1, pivot2, target, train_vec1, train_vec2):
    source, pivot1 = project_CCA(source, pivot1, train_vec1, 0.001, 100)
    pivot2, target = project_CCA(pivot2, target, train_vec2, 0.001, 100)
    train_vec3 = pivot_train_vec(pivot1, pivot2)
    source, target = project_CCA(source, target, train_vec3, 0.001, 50)

    return Lexicon(source, target)




