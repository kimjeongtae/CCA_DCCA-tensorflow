import tensorflow as tf
import numpy as np
from gensim.matutils import unitvec
from tools import W2V
from cca import calc_CCA


def xaver_init(n_inputs, n_outputs, uniform=False):
    '''
    :param n_inputs: 입력층 크기
    :param n_outputs: 출력층 크기
    :param uniform: True 이면 균등분포 , False 이면 정규분포
    :return: 가중치
    '''
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def build_network(input_dim, hidden_dim, output_dim, name):
    input_layer = tf.placeholder("float", shape=[None, input_dim], name=name + '_input')
    W1 = tf.get_variable(name=name + '_W1', shape=[input_dim, hidden_dim], initializer=xaver_init(input_dim, hidden_dim))
    b1 = tf.Variable(tf.zeros([hidden_dim]), name=name + '_b1')
    hidden_layer = tf.nn.tanh(tf.matmul(input_layer, W1) + b1, name=name + '_hidden')
    W2 = tf.get_variable(name=name + '_W2', shape=[hidden_dim, output_dim], initializer=xaver_init(hidden_dim, output_dim))
    b2 = tf.Variable(tf.zeros([output_dim]), name=name + '_b2')
    output_layer = tf.nn.l2_normalize(tf.nn.tanh(tf.matmul(hidden_layer, W2) + b2), dim=1, name=name + '_output')

    return input_layer, output_layer


def build_DCCA(train_vec, reg, n_components, epochs, learning_rate, save_path):
    '''
    :param train_vec: 학습벡터 쌍
    :param reg: 정규화 매개변수
    :param n_components: 투영할 벡터공간의 차원
    :param epochs: 세대
    :param learning_rate: 확습률률
    :param save_path: 심층 신경망 모델 저장 장소
    '''

    # set up the DCCA network

    n = len(train_vec[0][0])
    f_in, f_out = build_network(n, 5*n,  n,  name='f')
    g_in, g_out = build_network(n, 5*n,  n,  name='g')

    U = tf.placeholder("float", [n, n_components])
    V = tf.placeholder("float", [n, n_components])
    UtF = tf.matmul(tf.transpose(U), tf.transpose(f_out))
    GtV = tf.matmul(g_out, V)

    canon_corr = tf.trace(tf.matmul(UtF, GtV))
    corr_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(-canon_corr)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            f_out_ = sess.run(f_out, feed_dict={f_in: train_vec[0]})
            g_out_ = sess.run(g_out, feed_dict={g_in: train_vec[1]})
            U_, V_ = calc_CCA(f_out_, g_out_, reg, n_components)
            sess.run(corr_step, feed_dict={f_in: train_vec[0], g_in: train_vec[1], U: U_, V: V_})

        saver.save(sess, save_path)


def project_DCCA(X1, X2, train_vec, restore_path, reg=0.001, n_components=100):
    '''
    :param X1: W2V
    :param X2: W2V
    :param train_vec: 학습벡터 쌍
    :param restore_path:
    :param reg: 정규화 매개 변수
    :param n_components: 투영할 벡터공간의 차원
    :return: 투영된 언어 쌍
    '''
    n = X1.syn.shape[1]
    f_in, f_out = build_network(n, 5*n, n, name='f')
    g_in, g_out = build_network(n, 5*n, n, name='g')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        f_output = sess.run(f_out, feed_dict={f_in: X1.syn})
        g_output = sess.run(g_out, feed_dict={g_in: X2.syn})
        train1_out = sess.run(f_out, feed_dict={f_in: train_vec[0]})
        train2_out = sess.run(g_out, feed_dict={g_in: train_vec[0]})

    U, V = calc_CCA(train1_out, train2_out, reg, n_components)
    x, y = np.dot(f_output, U), np.dot(g_output, V)
    X1_syn, X2_syn = np.apply_along_axis(unitvec, 1, x), np.apply_along_axis(unitvec, 1, y)

    return W2V(X1.index2vec, X1_syn), W2V(X2.index2vec, X2_syn)