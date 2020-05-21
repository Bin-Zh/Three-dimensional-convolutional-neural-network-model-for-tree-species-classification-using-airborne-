import os
import time
import numpy as np
from keras.utils import np_utils
from keras.models import load_model

from Utils.Utils import loadTiff, standartizeData, Patch
import spectral
import matplotlib.pyplot as plt
from Utils.classifyUtils import reports, plot_confusion_matrix

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def classify(windowSize, testRatio, kdepth, vol_num, doMapping=False, doPlot_cnf=False, is_1d=False):
    reportFilename = '3D' + 'INSize' + str(windowSize) + \
                     'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + 'vol_num' + str(vol_num) + \
                     '.txt'
    modelName = 'hyperspectralModel' + '3D' + 'INSize' + str(windowSize) + \
                'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                'vol_num' + str(vol_num) + \
                '.h5'
    if is_1d:
        reportFilename = '3D-1d' + 'INSize' + str(windowSize) + \
                         'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + 'vol_num' + str(vol_num) + \
                         '.txt'
        modelName = 'hyperspectralModel' + '3D-1d' + 'INSize' + str(windowSize) + \
                    'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                    'vol_num' + str(vol_num) + \
                    '.h5'

    def load_test_data():
        X_test = np.load("/home/beer/bl-hsi/trainingData/" + "XtestWindowSize" +
                         str(windowSize) +
                         "testRatio" + str(testRatio) +
                         ".npy")

        y_test = np.load("/home/beer/bl-hsi/trainingData/" + "ytestWindowSize" +
                         str(windowSize) +
                         "testRatio" + str(testRatio) +
                         ".npy")

        X_test = np.reshape(X_test, (X_test.shape[0],
                                     X_test.shape[1],
                                     X_test.shape[2],
                                     X_test.shape[3], 1))
        y_test = np_utils.to_categorical(y_test)
        print(X_test.shape)
        print(y_test.shape)
        return X_test, y_test

    def loadModel():
        mode = "3D"
        print(modelName)
        model = load_model("model/" + modelName)
        model.summary()
        return model

    def writeReport(model, X_test, y_test):
        start_time = time.time()
        classification, confusion_raw, Test_loss, Test_accuracy, kappa, acc_for_each_class, average_acc, overall_acc = reports(
            model,
            X_test, y_test)
        classification = str(classification)
        confusion = str(confusion_raw)
        kappa = str(kappa)
        end_time = time.time()
        test_time = end_time - start_time
        print("TEST Time: ", test_time)
        with open("reports/" + reportFilename, 'w') as x_file:
            x_file.write('{} Test loss (%)'.format(Test_loss))
            x_file.write('\n')
            x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('K: {}'.format(kappa))
            x_file.write('\n')
            x_file.write('Acc for each class: {}'.format(acc_for_each_class))
            x_file.write('\n')

            x_file.write('Overall ACC: {}'.format(overall_acc))
            x_file.write('\n')
            x_file.write('Average ACC: {}'.format(average_acc))
            x_file.write('\n')
            x_file.write('TEST TIME: {}'.format(test_time))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))

        return confusion_raw

    def mapping(model):
        X, y = loadTiff()
        X, _sclaer = standartizeData(X)
        height = y.shape[0]
        width = y.shape[1]
        PATCH_SIZE = windowSize
        outputs = np.zeros((height, width))
        time_start = time.time()
        for i in range(height - PATCH_SIZE + 1):
            # print(i / (height - PATCH_SIZE + 1))

            patch1 = Patch(X, 1, 1, PATCH_SIZE)
            pred_line = np.zeros((width - PATCH_SIZE + 1,
                                  patch1.shape[0],
                                  patch1.shape[1],
                                  patch1.shape[2],
                                  1))

            for j in range(width - PATCH_SIZE + 1):
                target = int(y[i + PATCH_SIZE // 2, j + PATCH_SIZE // 2])
                # 要不要预测无标签区域？
                # if target == 0:
                #     continue
                # else:
                image_patch = Patch(X, i, j, PATCH_SIZE)

                # print (image_patch.shape)
                X_test_image = image_patch.reshape(1, image_patch.shape[0],
                                                   image_patch.shape[1],
                                                   image_patch.shape[2], 1).astype('float32')
                pred_line[j, :, :, :, :] = X_test_image

            prediction = model.predict_classes(pred_line)
            # print(prediction)
            outputs[i + PATCH_SIZE // 2][PATCH_SIZE // 2:width - PATCH_SIZE // 2] = prediction + 1
        end_time = time.time()
        print("Prediction Time", end_time - time_start)
        ground_truth = spectral.imshow(classes=y, figsize=(5, 5))
        spectral.save_rgb("ground_truth.png", y, colors=spectral.spy_colors)
        predict_image = spectral.imshow(classes=outputs.astype(int),
                                        figsize=(5, 5))
        results_name = '3D' + 'INSize' + str(windowSize) + \
                       'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + 'vol_num' + str(vol_num) + \
                       '.png'
        if is_1d:
            results_name = '3D-1d' + 'INSize' + str(windowSize) + \
                           'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + 'vol_num' + str(vol_num) + \
                           '.png'
        spectral.save_rgb("results/" + results_name,
                          outputs.astype(int),
                          colors=spectral.spy_colors)

    X_test, y_test = load_test_data()
    Model = loadModel()
    confusion_raw = writeReport(Model, X_test, y_test)
    if doMapping:
        mapping(Model)
    if doPlot_cnf:
        target_name = ['Cunninghamia lanceolata ', 'Pinus massoniana Lamb', 'Pinus elliottii',
                       'Eucalyptus grandis x E.urophylla', 'Eucalyptus urophyllaS', 'Castanopsis hystrix Miq.',
                       'Mytilaria laosensis', 'Camellia oleifera Abel.', 'Other broadleaved forest', 'Road ',
                       'cutting-blank',
                       ' Buliding land']
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(confusion_raw, classes=target_name, normalize=True, title='Confusion matrix')
        plt.show()
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(confusion_raw, classes=target_name, normalize=False, title='Normalized confusion matrix')
        plt.show()


if __name__ == "__main__":

    classify(13, 0.95, 7, [4, 8, 16, 32, 64], doMapping=True, doPlot_cnf=False)
    # 3d-1d-cnn
    # classify(13, 0.95, 7, [4, 8, 16, 32, 64], doMapping=True, doPlot_cnf=False, is_1d=True)
