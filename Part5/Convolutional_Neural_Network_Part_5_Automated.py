#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.color import rgb2hsv
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
import seaborn as sns
import pickle as pkl
import cv2
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout2D
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import os
import re
import copy
import time
import training_models
import types
import inspect


# In[2]:


# double checking to ensure gpu is enabled for training
K.tensorflow_backend._get_available_gpus()


# In[3]:


cwd = os.getcwd()


# In[4]:


os.listdir(cwd+ '\\color_data')


# In[5]:


target = pd.read_csv(cwd + '\\data\\count_.csv', names=['target'])


# In[6]:


target.head()


# In[7]:


input_ = np.append(pkl.load(open(r'C:\Users\abelp\machine_learning\crowd_count\data\final_input_aj_1.pkl', 'rb')),
                  pkl.load(open(r'C:\Users\abelp\machine_learning\crowd_count\data\final_input_aj_2.pkl', 'rb')), axis=0)


# In[8]:


model_location = 'C:\\Users\\abelp\\machine_learning\\crowd_count\\models\\'


# In[9]:


os.listdir(model_location)


# In[10]:


IMG_SIZE = input_.shape[1]
IMG_SIZE2 = input_.shape[2]


# In[11]:


models_ = training_models.automated_model_building


# In[12]:


target = np.array(target['target'])


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(input_, target, test_size=0.2, random_state=42)


# In[14]:


X_train.shape


# In[15]:


X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


# In[16]:


X_train.shape


# In[17]:


del input_


# In[18]:


val_loss_dict = ['placeholder',100000000]
to_test = []


# In[19]:


def data_Generator():
    while True:
        for i in range(0,len(X_train) // 100):
            time.sleep(0.01)
            yield  X_train[i*25:(i+1)*25], y_train[i*25:(i+1)*25]


# In[20]:


filter_size = [128, 256]
kernel_size = [(11,11)]
stride_size = [(3,3), (2,2)]


# In[21]:


filter_size_2 = [128, 256]
kernel_size_2 = [(8,8), (4,4)]
stride_size_2 = [(2,2), (1,1)]


# In[22]:


a = list(product(filter_size, kernel_size, stride_size))


# In[23]:


b = list(product(filter_size_2, kernel_size_2, stride_size_2))


# In[24]:


to_test = list(product(a,b))


# In[25]:


to_test


# In[26]:


to_test[0]


# In[27]:


def auto_train_model(model_params):        
    #filepath = model_location + method + '.h5'
    
    es_callback = EarlyStopping(monitor='mse', patience=4)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_mse', save_best_only=True, mode='min')
    
    model = models_.model_builder(IMG_SIZE, IMG_SIZE2, model_params)

    model.compile(optimizer='ADAM', loss='mse', metrics=['mse'])
    
    history = model.fit_generator(data_Generator(), steps_per_epoch=25, epochs=500,  callbacks=[es_callback], verbose=0, validation_data=(X_test, y_test))
    
    if val_loss_dict[1] > min(history.history['val_mse']):
        val_loss_dict[0] = model_params
        val_loss_dict[1] = min(history.history['val_mse'])
    
    del history
    del model


# In[28]:


for x in to_test:
    auto_train_model(x)


# In[ ]:


#exec_methods = ["basic_CNN", "basic_CNN_v2", "basic_CNN_v2_DO", "basic_CNN_v2_BN","basic_CNN_v2_DO_LR", "basic_CNN_v2_BN_LR", "alex_net_trasfer", "alex_net_trasfer_v2"]
#for method in exec_methods:
#    auto_train_model(method)


# In[ ]:


val_loss_dict


# In[ ]:


with open('output.txt', 'w') as file:  # Use file to refer to the file object
    file.write(val_loss_dict)

