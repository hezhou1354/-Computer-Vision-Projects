import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

import random


def get_mini_batch(im_train, label_train, batch_size):
    # im_train: 196 x n, a matrix storing vectorized image data
    # label_train: 1 x n, a vector storing the label for each image data
    # batch size: the size of the mini-batch for stochastic gradient descent

    n = np.shape(im_train)[1]
    n_batch = int(math.ceil(n / batch_size))

    ## one-hot encoding
    y_train = np.transpose(np.eye(10)[label_train[0]])

    ## permutation
    random.seed(5561)
    index_perm = np.random.permutation(n)

    mini_batch_x = []
    mini_batch_y = []

    for i in range(n_batch-1):
        batch_index = index_perm[np.arange(i*batch_size, (i+1)*batch_size, 1)]
        mini_batch_x.append(im_train[:, batch_index])
        mini_batch_y.append(y_train[:, batch_index])

    ## last batch may be smaller than batch_size
    last_batch_index = index_perm[np.arange((n_batch-1)*batch_size, n, 1)]
    mini_batch_x.append(im_train[:, last_batch_index])
    mini_batch_y.append(y_train[:, last_batch_index])

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    #  input:
    #      x: m x 1, the input to the fully connected layer
    #      w: n x m, the weights
    #      b: n x 1, the bias
    # output:
    #      y: n x 1, the output of the linear transform
    y = np.matmul(w, x) + b

    return y


def fc_backward(dl_dy, x, w, b, y):
    # input:
    #      dl_dy: 1 x n, the loss derivative w.r.t. the output y
    # output:
    #      dl_dx: 1 x m, the loss derivative w.r.t. the input x
    #      dl_dw: 1 x (n x m), the loss derivative w.r.t. the weights
    #      dl_db: 1 x n, the loss derivative w.r.t. the bias

    dl_dx = np.matmul(dl_dy, w)
    dl_dw = np.transpose(np.matmul(x, dl_dy)).reshape((1,) + np.shape(w))
    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # input:
    #     y_tilde: m, prediction
    #     y: {0,1}^m, ground truth label
    # output:
    #     l: loss
    #     dl_dy: loss derivative w.r.t. prediction

    l = np.sum((y - y_tilde)**2)
    dl_dy = np.transpose(2 * (y_tilde - y))

    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # input:
    #     x: m x 1
    #     y: {0,1}^m
    # output:
    #     l: the cross-entropy loss
    #     dl_dy: the loss derivative w.r.t. x

    y_tilde = np.exp(x) / np.sum(np.exp(x))
    l = - np.sum(y * np.log(y_tilde))
    dl_dy = np.transpose(- y + y_tilde)

    return l, dl_dy

def relu(x):

    epsilon = 0.01
    y = np.where(x > 0, x, epsilon * x)

    return y


def relu_backward(dl_dy, x, y):

    epsilon = 0.01
    dl_dx = dl_dy * np.where(x > 0, 1, epsilon)

    return dl_dx

def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (H, W, C)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    h, w, c = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w, c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[i*stride:i*stride+hh, j*stride:j*stride+ww, ...]
           col[i*new_w+j, :] = np.reshape(patch, -1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (h_prime,w_prime,F) matrix
      Otherwise: (h_prime,w_prime, F, C) matrix
    """
    F = mul.shape[1]
    if C == 1:
        out = np.zeros([h_prime, w_prime, F])
        for i in range(F):
            col = mul[:, i]
            out[:, :, i] = np.reshape(col, (h_prime, w_prime))
    else:
        out = np.zeros([h_prime, w_prime, F, C])
        for i in range(F):
            col = mul[:, i]
            out[:, :, i] = np.reshape(col, (h_prime, w_prime, C))

    return out



def conv(x, w_conv, b_conv):

    h, w, c1 = x.shape
    hh, ww, c1, c2 = w_conv.shape
    stride = 1
    new_h = (h + 2 - hh) // stride + 1
    new_w = (w + 2 - ww) // stride + 1
    x_padding = np.pad(x, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
    x_col = im2col(x_padding, hh, ww, stride)
    filter_col = np.reshape(w_conv, (-1, c2))
    mul = x_col.dot(filter_col)
    mul = mul + np.repeat(b_conv.T, repeats=mul.shape[0], axis=0)
    y = col2im(mul, new_h, new_w, 1)

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):

    hh, ww, c1, c2 = w_conv.shape
    stride = 1

    x_padding = np.pad(x, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
    x_col = im2col(x_padding, hh, ww, stride)

    dbias_sum = np.reshape(dl_dy, (-1, c2))
    dl_db = np.sum(dbias_sum, axis=0).reshape((-1, 1))
    dfilter_col = (x_col.T).dot(dbias_sum)
    dl_dw = np.reshape(dfilter_col, (hh, ww, c1, c2))

    return dl_dw, dl_db

def pool2x2(x):

    h, w, c1 = x.shape
    h_p = 2
    w_p = 2
    stride_p = 2

    hh = int(1 + (h - h_p) / stride_p)
    ww = int(1 + (w - w_p) / stride_p)

    y = np.zeros((hh, ww, c1))
    for depth in range(c1):
        for r in range(hh):
            for c in range(ww):
                y[r, c, depth] = np.max(x[r*stride_p:r*stride_p+h_p, c*stride_p:c*stride_p+w_p, depth])

    return y

def pool2x2_backward(dl_dy, x, y):

    h, w, c1 = x.shape
    h_p = 2
    w_p = 2
    stride_p = 2
    hh, ww, c1 = dl_dy.shape

    dl_dx = np.zeros(x.shape)
    for depth in range(c1):
        for r in range(hh):
            for c in range(ww):
                x_pool = x[r*stride_p:r*stride_p+h_p, c*stride_p:c*stride_p+w_p, depth]
                mask = (x_pool == np.max(x_pool))
                dl_dx[r*stride_p:r*stride_p+h_p, c*stride_p:c*stride_p+w_p, depth] = mask*dl_dy[r, c, depth]

    return dl_dx


def flattening(x):

    y = x.reshape((-1, 1))

    return y


def flattening_backward(dl_dy, x, y):

    dl_dx = dl_dy.reshape(x.shape)

    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # train a single-layer linear perceptron using a stochastic gradient descent method

    n_batch = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    # learning rate
    gamma = 0.01

    # decay rate
    Lambda = 0.1

    # maximum number of iterations
    nIters = 2000

    # initialize the weights with a Gaussian noise
    random.seed(5561)
    w = np.random.normal(loc=0.0, scale=1.0, size=(10, 196))
    b = np.random.normal(loc=0.0, scale=1.0, size=(10, 1))
    L_list = []

    k = 0
    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = Lambda * gamma
        dL_dw = np.zeros(((1,) + np.shape(w)))
        dL_db = np.transpose(np.zeros(np.shape(b)))

        # k-th mini-batch
        k_batch_x = mini_batch_x[k]
        k_batch_y = mini_batch_y[k]
        k_batch_size = k_batch_x.shape[1]
        for i in range(k_batch_size):
            x = k_batch_x[:, i].reshape((-1, 1))
            y = k_batch_y[:, i].reshape((-1, 1))
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_euclidean(y_tilde, y)
            L_list.append(l)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db
        k = k + 1
        if k >= n_batch:
            k = 0
        w = w - gamma * dL_dw[0, :, :] / k_batch_size
        b = b - gamma * np.transpose(dL_db) / k_batch_size

    ## plot the Loss v.s. Iterations
    # plt.plot(L_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # train a single-layer perceptron using a stochastic gradient descent method

    n_batch = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    # learning rate
    gamma = 0.135

    # decay rate
    Lambda = 0.895

    # maximum number of iterations
    nIters = 2500

    # initialize the weights with a Gaussian noise
    random.seed(5561)
    w = np.random.normal(loc=0.0, scale=1.0, size=(10, 196))
    b = np.random.normal(loc=0.0, scale=1.0, size=(10, 1))
    L_list = []

    k = 0
    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = Lambda * gamma
        dL_dw = np.zeros(((1,) + np.shape(w)))
        dL_db = np.transpose(np.zeros(np.shape(b)))

        # k-th mini-batch
        k_batch_x = mini_batch_x[k]
        k_batch_y = mini_batch_y[k]
        k_batch_size = k_batch_x.shape[1]
        for i in range(k_batch_size):
            x = k_batch_x[:, i].reshape((-1, 1))
            y = k_batch_y[:, i].reshape((-1, 1))
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L_list.append(l)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw = dL_dw + dl_dw
            dL_db = dL_db + dl_db
        k = k + 1
        if k >= n_batch:
            k = 0
        w = w - gamma * dL_dw[0, :, :] / k_batch_size
        b = b - gamma * np.transpose(dL_db) / k_batch_size

    ## plot the Loss v.s. Iterations
    # plt.plot(L_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    # train a single-layer perceptron using a stochastic gradient descent method

    n_batch = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    # learning rate
    gamma = 0.49

    # decay rate
    Lambda = 0.905

    # maximum number of iterations
    nIters = 5000

    # initialize the weights with a Gaussian noise
    random.seed(5561)
    w1 = np.random.normal(loc=0.0, scale=1.0, size=(30, 196))
    w2 = np.random.normal(loc=0.0, scale=1.0, size=(10, 30))
    b1 = np.random.normal(loc=0.0, scale=1.0, size=(30, 1))
    b2 = np.random.normal(loc=0.0, scale=1.0, size=(10, 1))
    L_list = []

    k = 0
    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = Lambda * gamma
        dL_dw1 = np.zeros(((1,) + np.shape(w1)))
        dL_db1 = np.transpose(np.zeros(np.shape(b1)))
        dL_dw2 = np.zeros(((1,) + np.shape(w2)))
        dL_db2 = np.transpose(np.zeros(np.shape(b2)))

        # k-th mini-batch
        k_batch_x = mini_batch_x[k]
        k_batch_y = mini_batch_y[k]
        k_batch_size = k_batch_x.shape[1]
        for i in range(k_batch_size):
            x = k_batch_x[:, i].reshape((-1, 1))
            y = k_batch_y[:, i].reshape((-1, 1))
            a1 = fc(x, w1, b1)
            f1 = relu(a1)
            y_tilde = fc(f1, w2, b2)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L_list.append(l)
            # back-propagation
            dl_df1, dl_dw2, dl_db2 = fc_backward(dl_dy, f1, w2, b2, y)
            dl_da1 = relu_backward(dl_df1, a1.T, f1)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_da1, x, w1, b1, a1)

            dL_dw1 = dL_dw1 + dl_dw1
            dL_db1 = dL_db1 + dl_db1
            dL_dw2 = dL_dw2 + dl_dw2
            dL_db2 = dL_db2 + dl_db2
        k = k + 1
        if k >= n_batch:
            k = 0
        w1 = w1 - gamma * dL_dw1[0, :, :] / k_batch_size
        b1 = b1 - gamma * np.transpose(dL_db1) / k_batch_size
        w2 = w2 - gamma * dL_dw2[0, :, :] / k_batch_size
        b2 = b2 - gamma * np.transpose(dL_db2) / k_batch_size

    ## plot the Loss v.s. Iterations
    # plt.plot(L_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    return w1, b1, w2, b2




def train_cnn(mini_batch_x, mini_batch_y):
    # train a CNN using a stochastic gradient descent method

    n_batch = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    # learning rate
    gamma = 0.5

    # decay rate
    Lambda = 0.89

    # maximum number of iterations
    nIters = 1000

    # initialize the weights with a Gaussian noise
    random.seed(5561)
    w_conv = np.random.normal(loc=0.0, scale=0.1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(loc=0.0, scale=0.1, size=(3, 1))
    w_fc = np.random.normal(loc=0.0, scale=0.1, size=(10, 147))
    b_fc = np.random.normal(loc=0.0, scale=0.1, size=(10, 1))

    L_list = []

    k = 0
    for iIter in range(nIters):
        if (iIter + 1) % 1000 == 0:
            gamma = Lambda * gamma
        dL_dw_conv = np.zeros(np.shape(w_conv))
        dL_db_conv = np.zeros(np.shape(b_conv))
        dL_dw_fc = np.zeros(np.shape(w_fc))
        dL_db_fc = np.zeros(np.shape(b_fc))

        # k-th mini-batch
        k_batch_x = mini_batch_x[k]
        k_batch_y = mini_batch_y[k]
        k_batch_size = k_batch_x.shape[1]
        for i in range(k_batch_size):
            x = k_batch_x[:, i].reshape((14, 14, 1), order="F")
            y = k_batch_y[:, i].reshape((-1, 1))
            a1 = conv(x, w_conv, b_conv)
            f1 = relu(a1)
            f2 = pool2x2(f1)
            f3 = flattening(f2)
            y_tilde = fc(f3, w_fc, b_fc)

            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L_list.append(l)
            # back-propagation
            dl_df3, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, f3, w_fc, b_fc, y)
            dl_df2 = flattening_backward(dl_df3, f2, f3)
            dl_df1 = pool2x2_backward(dl_df2, f1, f2)
            dl_da1 = relu_backward(dl_df1, a1, f1)
            dl_dw_conv, dl_db_conv = conv_backward(dl_da1, x, w_conv, b_conv, a1)

            dL_dw_conv = dL_dw_conv + dl_dw_conv
            dL_db_conv = dL_db_conv + dl_db_conv
            dL_dw_fc = dL_dw_fc + dl_dw_fc
            dL_db_fc = dL_db_fc + np.transpose(dl_db_fc)

        k = k + 1
        if k >= n_batch:
            k = 0
        w_conv = w_conv - gamma * dL_dw_conv / k_batch_size
        b_conv = b_conv - gamma * dL_db_conv / k_batch_size
        w_fc = w_fc - gamma * dL_dw_fc[0, :, :] / k_batch_size
        b_fc = b_fc - gamma * dL_db_fc / k_batch_size

    ## plot the Loss v.s. Iterations
    # plt.plot(L_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



