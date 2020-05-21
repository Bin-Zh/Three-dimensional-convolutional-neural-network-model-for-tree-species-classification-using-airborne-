import os
import time
import numpy as np

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import TensorBoard


K.set_image_dim_ordering('tf')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(windowSize, kdepth, vol_num, testRatio=0.95, dropout=0.5, epochs=300, is_1d=False):
    model_name = 'hyperspectralModel' + '3D' + 'INSize' + str(windowSize) + \
                 'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                 'vol_num' + str(vol_num) + \
                 '.h5'
    if is_1d:
        model_name = 'hyperspectralModel' + '3D-1d' + 'INSize' + str(windowSize) + \
                     'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                     'vol_num' + str(vol_num) + \
                     '.h5'
    if dropout != 0.5:
        model_name = 'hyperspectralModel' + '3D' + 'INSize' + str(windowSize) + \
                     'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                     'vol_num' + str(vol_num) + 'dropout' + str(dropout) + \
                     '.h5'
        if is_1d:
            model_name = 'hyperspectralModel' + '3D-1d' + 'INSize' + str(windowSize) + \
                         'testRatio' + str(testRatio) + 'kdepth' + str(kdepth) + \
                         'vol_num' + str(vol_num) + 'dropout' + str(dropout) + \
                         '.h5'

    def loadTrainData():
        X_train = np.load("/home/beer/bl-hsi/trainingData/" + "XtrainWindowSize" +
                          str(windowSize) +

                          "testRatio" + str(testRatio) +
                          ".npy")

        y_train = np.load("/home/beer/bl-hsi/trainingData/" + "ytrainWindowSize" +
                          str(windowSize) +

                          "testRatio" + str(testRatio) +
                          ".npy")

        X_train = np.reshape(X_train, (X_train.shape[0],
                                       X_train.shape[1],
                                       X_train.shape[2],
                                       X_train.shape[3], 1))

        # convert class labels to on-hot encoding
        y_train = np_utils.to_categorical(y_train)
        print(X_train.shape)
        return X_train, y_train

    X_train, y_train = loadTrainData()
    input_shape = X_train[0].shape
    print(input_shape)
    print(X_train.shape)
    species_num = y_train.shape[1]

    def defineModel():
        model = Sequential()
        model.add(Conv3D(vol_num[0], (3, 3, kdepth), activation='relu', input_shape=input_shape))

        model.add(Conv3D(vol_num[1], (3, 3, kdepth), activation='relu'))

        model.add(Conv3D(vol_num[2], (3, 3, kdepth), activation='relu',))

        model.add(Conv3D(vol_num[3], (3, 3, kdepth), activation='relu',))

        model.add(Conv3D(vol_num[3], (3, 3, kdepth), activation='relu', ))

        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(species_num, activation='softmax'))

        return model

    def defineModel_1d():
        model = Sequential()
        model.add(Conv3D(vol_num[0], (3, 3, kdepth), activation='relu', input_shape=input_shape))

        model.add(Conv3D(vol_num[1], (3, 3, kdepth), activation='relu'))

        model.add(Conv3D(vol_num[2], (3, 3, kdepth), activation='relu', ))

        model.add(Conv3D(vol_num[3], (3, 3, kdepth), activation='relu', ))

        model.add(Conv3D(vol_num[3], (3, 3, kdepth), activation='relu', ))

        model.add(Dropout(dropout))

        model.reshape(95, 64)

        model.add(Conv1D(48, 7, activation='relu'))

        model.add(Conv1D(24, 7, activation='relu'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(species_num, activation='softmax'))

        return model

    def trainModel(Model):
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        Model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])
        Model.summary()
        time_start = time.time()
        tensorboardName = model_name.replace('.h5', '')
        callbacks = [TensorBoard(log_dir='./tmp/' + str(tensorboardName))]
        Model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_split=0.5, shuffle=True, callbacks=callbacks)
        time_end = time.time()
        Model.save('model/' + model_name)
        with open('results/time.txt', 'a') as f:
            f.write(tensorboardName + str(time_end - time_start))
            f.write('\n')
    Model = defineModel()
    if is_1d:
        Model = defineModel_1d()
    trainModel(Model)


if __name__ == "__main__":
    train(11, 7, [4, 8, 16, 32, 64], dropout=0.5)
    # 3d-1d-cnn
    # train(11, 7, [4, 8, 16, 32, 64], dropout=0.5, is_1d=True)
