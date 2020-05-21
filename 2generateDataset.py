"""Python script to generate dataset from .tif files."""

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
import scipy.ndimage
from osgeo import gdal

# Global Variables
doPCA = False
numComponents = 3
windowSize = 11
testRatio = 0.95

PATH = os.getcwd()
print(PATH)


def loadTiff():
    data_path = os.path.join('./data/testarea2_new/sg_quac_test2.tif')
    data = gdal.Open(data_path).ReadAsArray()
    labels = gdal.Open('./data/testarea2_new/sg_quac_test2_gt.tif').ReadAsArray()

    def channel_first_to_channel_last(data):

        print("ORIGINAL_SHAPE： {}".format(data.shape))
        data_ = np.zeros((data.shape[1], data.shape[2], data.shape[0]))
        for i in range(data.shape[0]):
            data_[:, :, i] = data[i, :, :]
        for test in range(10):
            test = + 1
            x = np.random.randint(0, data.shape[1])
            y = np.random.randint(0, data.shape[2])
            c = np.random.randint(0, data.shape[0])
            assert data[c, x, y] == data_[x, y, c], "There is a problem with this function."
        print(data_.shape)

        return data_

    return channel_first_to_channel_last(data), labels


def loadIndianPinesData():
    """Method to load the IndianPines Dataset."""
    data_path = os.path.join(os.getcwd(), 'data')
    data = sio.loadmat(os.path.join(data_path,
                                    'Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = sio.loadmat(os.path.join(data_path,
                                      'Indian_pines_gt.mat'))['indian_pines_gt']

    return data, labels


def splitTrainTestSet(X, y, testRatio=0.05):
    """Method to split data into train and test set."""
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=testRatio,
                                                        random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    print(uniqueLabels, labelCounts)
    maxCount = np.max(labelCounts)
    print("maxCount", maxCount)
    labelInverseRatios = maxCount / labelCounts
    print("labelInverseRatios", labelInverseRatios)
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(
        round(labelInverseRatios[0]), axis=0)

    newY = y[y == uniqueLabels[0]].repeat(
        round(labelInverseRatios[0]), axis=0)
    print(X.shape, y.shape)
    print(newX.shape, newY.shape)
    for label, labelInverseRatio in zip(uniqueLabels[1:],
                                        labelInverseRatios[1:]):
        cX = X[y == label, :, :, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    print(rand_perm)
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    print(newX.shape, newY.shape)
    return newX, newY


def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX, scaler


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin,
                     X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createPatches(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) // 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize,
                            X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    print(patchesData.shape, patchesLabels.shape)
    print(y.shape)
    print(X.shape)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1,
                    c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def AugmentData(X_train):
    for i in range(int(X_train.shape[0] // 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch,
                                                               no,
                                                               axes=(1, 0),
                                                               reshape=False,
                                                               output=None,
                                                               order=3,
                                                               mode='constant',
                                                               cval=0.0,
                                                               prefilter=False)
    patch2 = flipped_patch
    X_train[i, :, :, :] = patch2

    return X_train


def savePreprocessedData(X_trainPatches, X_testPatches, y_trainPatches,
                         y_testPatches, windowSize, wasPCAapplied=False,
                         numPCAComponents=0, testRatio=0.05):
    if wasPCAapplied:
        with open(PATH + "/trainingData/" + "XtrainWindowSize" +
                  str(windowSize) +
                  "PCA" + str(numPCAComponents) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open(PATH + "/trainingData/" + "XtestWindowSize" +
                  str(windowSize) +
                  "PCA" + str(numPCAComponents) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open(PATH + "/trainingData/" + "ytrainWindowSize" +
                  str(windowSize) +
                  "PCA" + str(numPCAComponents) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open(PATH + "/trainingData/" + "ytestWindowSize" +
                  str(windowSize) +
                  "PCA" + str(numPCAComponents) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)
    else:
        with open("trainingData/XtrainWindowSize" +
                  str(windowSize) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open("trainingData/XtestWindowSize" +
                  str(windowSize) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open("trainingData/ytrainWindowSize" +
                  str(windowSize) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open("trainingData/ytestWindowSize" +
                  str(windowSize) +
                  "testRatio" + str(testRatio) +
                  ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)


for testRatio in [0.8, 0.95, ]:
    X, y = loadTiff()
    print(X.shape)
    if doPCA:
        X, pca = applyPCA(X, numComponents=numComponents)
    X, _scaler = standartizeData(X)
    XPatches, yPatches = createPatches(X, y, windowSize=windowSize)
    print("Patch Create Complete!")

    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches,
                                                         yPatches,
                                                         testRatio)
    print("Split DataSat Complete!")
    XPatches, yPatches = None, None
    # X_train, y_train = oversampleWeakClasses(X_train, y_train)

    X_train = AugmentData(X_train)
    print("Train Data Size", X_train.shape)
    savePreprocessedData(X_train, X_test, y_train, y_test, windowSize=windowSize,
                         wasPCAapplied=doPCA, numPCAComponents=numComponents,
                         testRatio=testRatio)
