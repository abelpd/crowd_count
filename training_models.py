from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import SpatialDropout2D

class models_to_train:
    def basic_CNN(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=56,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(500,activation='relu'))
        model.add(Dense(1))
        return model

    def basic_CNN_v2(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=256,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model

    def basic_CNN_v2_DO(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model

    def basic_CNN_v2_BN(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model

    def basic_CNN_v2_DO_LR(IMG_SIZE, IMG_SIZE2):
        act = LeakyReLU()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256,  kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(act)
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model

    def basic_CNN_v2_BN_LR(IMG_SIZE, IMG_SIZE2):
        act = LeakyReLU()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256,  kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model

    def alex_net_trasfer(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid',  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid',  activation='relu'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model


    def alex_net_trasfer_v2(IMG_SIZE, IMG_SIZE2):
        act = LeakyReLU()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(IMG_SIZE, IMG_SIZE2,1)))
        model.add(Conv2D(filters=96,  kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(1))
        return model


    def alex_net_trasfer_actual(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(IMG_SIZE,IMG_SIZE2,1), kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(2,2), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(3000, input_shape=(IMG_SIZE*IMG_SIZE2*1,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(2000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    def alex_net_trasfer_DO(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(IMG_SIZE,IMG_SIZE2,1), kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(2,2), padding='valid', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(3000, input_shape=(IMG_SIZE*IMG_SIZE2*1,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(2000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    def alex_net_trasfer_BE(IMG_SIZE, IMG_SIZE2):
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(IMG_SIZE,IMG_SIZE2,1), kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(2,2), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(3000, input_shape=(IMG_SIZE*IMG_SIZE2*1,), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(2000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model

    def alex_net_trasfer_BE_LR(IMG_SIZE, IMG_SIZE2):
        act = LeakyReLU()
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(IMG_SIZE,IMG_SIZE2,1), kernel_size=(11,11),strides=(4,4), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(2,2), padding='valid'))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(act)
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(3000, input_shape=(IMG_SIZE*IMG_SIZE2*1,)))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Dense(2000))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Dense(1000))
        model.add(act)
        model.add(BatchNormalization())
        model.add(Dense(1))
        return model