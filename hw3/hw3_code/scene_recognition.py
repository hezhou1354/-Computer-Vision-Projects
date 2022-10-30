import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list



def get_tiny_image(img, output_size):

    #  Resize image
    feature = cv2.resize(img, output_size)

    #  normalization
    ## subtract the mean
    feature = feature - np.mean(feature)
    ## divided by the norm
    feature = feature / np.linalg.norm(feature.reshape(-1))

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    #  KNN
    ## Calculate distance
    neigh = NearestNeighbors(n_neighbors=k).fit(feature_train)
    
    ## Find closest neighbors
    neigh_dist, neigh_ind = neigh.kneighbors(feature_test)
    
    ## Predict the label for test set - Vote for labels
    n_te = np.shape(feature_test)[0]
    label_test_pred = np.zeros(n_te)
    for i in range(n_te):
        neigh_label = label_train[neigh_ind[i, :]]
        label_test_pred[i] = np.argmax(np.bincount(neigh_label))
    label_test_pred = label_test_pred.astype(int)

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    output_size = (16, 16)
    k = 10
    #  Preparations
    ## feature_train for KNN & label_train for KNN
    n_tr = len(label_train_list)
    feature_train = np.zeros((n_tr, output_size[0]*output_size[1]))
    label_train = np.zeros(n_tr)
    for i in range(n_tr):
        ## feature
        train_img_name = img_train_list[i]
        img = cv2.imread(train_img_name, 0)
        tiny_image = get_tiny_image(img, output_size)
        feature_train[i, :] = tiny_image.reshape(-1)
        ## label
        train_label = label_train_list[i]
        label_train[i] = label_classes.index(train_label)
    label_train = label_train.astype(int)


    ## feature_test for KNN & label_test for confusion matrix
    n_te = len(label_test_list)
    feature_test = np.zeros((n_te, output_size[0]*output_size[1]))
    label_test = np.zeros(n_te)
    for i in range(n_te):
        ## feature
        test_img_name = img_test_list[i]
        img = cv2.imread(test_img_name, 0)
        tiny_image = get_tiny_image(img, output_size)
        feature_test[i, :] = tiny_image.reshape(-1)
        ## label
        test_label = label_test_list[i]
        label_test[i] = label_classes.index(test_label)
    label_test = label_test.astype(int)


    #  KNN
    label_test_pred = predict_knn(feature_train, label_train, feature_test, k)

    #  Calculating the confusion and accuracy matrix
    confusion = sklearn.metrics.confusion_matrix(label_test, label_test_pred)
    accuracy = np.trace(confusion) / n_te

    #  Visualize the confusion matrix
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy



def compute_dsift(img, stride, size):

    #  Creat the sift descriptor
    sift = cv2.SIFT_create()
    y_patch = int(img.shape[0] / stride)
    x_patch = int(img.shape[1] / stride)

    #  Get the keypoint for each location
    keypoints = []
    for i in range(x_patch):
        for j in range(y_patch):
            keypoint = cv2.KeyPoint(i * stride + size, j * stride + size, size)
            keypoints.append(keypoint)

    #  Compute sift descriptor at each keypoint
    keypoints, dense_feature = sift.compute(img, keypoints)

    return dense_feature



def build_visual_dictionary(dense_feature_list, dic_size):

    #  Build a pool of SIFT feature from the list
    dense_feature_pool = np.vstack(dense_feature_list)

    #  Find cluster centers from the SIFT pool using kmeans alg
    ## dic_size = 50
    kmeans_fit = KMeans(n_clusters=dic_size, n_init=20, max_iter=300).fit(dense_feature_pool)
    vocab = kmeans_fit.cluster_centers_

    return vocab


def compute_bow(feature, vocab):
    #  Use KNN to find the closest cluster center
    ## Calculate distance
    neigh = NearestNeighbors(n_neighbors=1).fit(vocab)

    ## Find closest neighbors
    neigh_dist, neigh_ind = neigh.kneighbors(feature)

    ## Calculate the BoW feature
    dic_size = vocab.shape[0]
    bow_feature = np.bincount(neigh_ind.reshape(-1), minlength=dic_size)

    ## Normalize the BoW feature
    bow_feature = bow_feature / np.linalg.norm(bow_feature)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    stride = 20
    size = 20
    n_tr = len(img_train_list)
    n_te = len(img_test_list)
    dic_size = 50

    #  BoW
    ## Calculate the list of dense feature & the training label
    train_dense_feature_list = []
    label_train = np.zeros(n_tr)
    for i in range(n_tr):
        ## dense feature
        train_img_name = img_train_list[i]
        img = cv2.imread(train_img_name, 0)
        dense_feature = compute_dsift(img, stride, size)
        train_dense_feature_list.append(dense_feature)
        ## label
        train_label = label_train_list[i]
        label_train[i] = label_classes.index(train_label)
    label_train = label_train.astype(int)

    ## Build visual dictionary
    vocab = build_visual_dictionary(train_dense_feature_list, dic_size)
    np.savetxt('vocab_knn_bow.txt', vocab)

    ## Build training BoW features
    train_bow_list = []
    for i in range(n_tr):
        dense_feature = train_dense_feature_list[i]
        bow_feature = compute_bow(dense_feature, vocab)
        train_bow_list.append(bow_feature)

    ## Build testing BoW features & testing label
    test_bow_list = []
    label_test = np.zeros(n_te)
    for i in range(n_te):
        ## BoW feature
        test_img_name = img_test_list[i]
        img = cv2.imread(test_img_name, 0)
        dense_feature = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_feature, vocab)
        test_bow_list.append(bow_feature)
        ## label
        test_label = label_test_list[i]
        label_test[i] = label_classes.index(test_label)
    label_test = label_test.astype(int)

    #  KNN
    k = 10
    label_test_pred = predict_knn(train_bow_list, label_train,  test_bow_list, k)

    #  Calculating the confusion and accuracy matrix
    confusion = sklearn.metrics.confusion_matrix(label_test, label_test_pred)
    accuracy = np.trace(confusion) / n_te

    #  Visualize the confusion matrix
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):

    # Fit 15 binary 1-vs-all SVMs to the test features
    label_test_ova = np.zeros((15, len(feature_test)))
    for i in range(15):
        # generate the one versus all label
        label_train_ova = [1 if j == i else 0 for j in label_train]
        clf_ova = LinearSVC(C=5.0, random_state=0, tol=1e-05, max_iter=2000).fit(feature_train, label_train_ova)
        label_test_ova[i, :] = clf_ova.decision_function(feature_test)

    # the classifier is the most confidently positive "wins"
    label_test_pred = np.argmax(label_test_ova, axis=0)

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    stride = 15
    size = 15
    n_tr = len(img_train_list)
    n_te = len(img_test_list)
    dic_size = 50

    #  BoW
    ## Calculate the list of dense feature & the training label
    train_dense_feature_list = []
    label_train = np.zeros(n_tr)
    for i in range(n_tr):
        ## dense feature
        train_img_name = img_train_list[i]
        img = cv2.imread(train_img_name, 0)
        dense_feature = compute_dsift(img, stride, size)
        train_dense_feature_list.append(dense_feature)
        ## label
        train_label = label_train_list[i]
        label_train[i] = label_classes.index(train_label)
    label_train = label_train.astype(int)

    ## Build visual dictionary
    vocab = build_visual_dictionary(train_dense_feature_list, dic_size)
    np.savetxt('vocab_svm_bow.txt', vocab)

    ## Build training BoW features
    train_bow_list = []
    for i in range(n_tr):
        dense_feature = train_dense_feature_list[i]
        bow_feature = compute_bow(dense_feature, vocab)
        train_bow_list.append(bow_feature)

    ## Build testing BoW features & testing label
    test_bow_list = []
    label_test = np.zeros(n_te)
    for i in range(n_te):
        ## BoW feature
        test_img_name = img_test_list[i]
        img = cv2.imread(test_img_name, 0)
        dense_feature = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_feature, vocab)
        test_bow_list.append(bow_feature)
        ## label
        test_label = label_test_list[i]
        label_test[i] = label_classes.index(test_label)
    label_test = label_test.astype(int)

    #  SVM
    label_test_pred = predict_svm(train_bow_list, label_train, test_bow_list)

    #  Calculating the confusion and accuracy matrix
    confusion = sklearn.metrics.confusion_matrix(label_test, label_test_pred)
    accuracy = np.trace(confusion) / n_te

    #  Visualize the confusion matrix
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)



