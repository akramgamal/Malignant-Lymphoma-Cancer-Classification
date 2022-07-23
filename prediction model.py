from keras.models import Sequential, Input, Model
from keras import applications
from keras.models import Model
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Conv3D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import glob
from tensorflow import keras
import os
import numpy as np
import shutil
import random
from keras.layers import Activation
import image_slicer

#Load Model
model = tf.keras.models.load_model("/Users/Zamzam/Desktop/best acc 3/CNN.h5")
cll='/Users/Zamzam/Desktop/malignant lymphoma classification/input/CLL/'
os.makedirs(cll+'slicing')
os.makedirs(cll+'slicing/0')
os.makedirs(cll+'slicing/1')
os.makedirs(cll+'slicing/2')
tiles=image_slicer.slice(cll+'1'+'.tif', 16,save=False)
image_slicer.save_tiles(tiles, directory=cll+'slicing/0',format='png')
image_slicer.save_tiles(tiles, directory=cll+'slicing/1',format='png')
image_slicer.save_tiles(tiles, directory=cll+'slicing/2',format='png')


test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=cll+'slicing',
    target_size=(224,224),
    batch_size=48,
    class_mode='categorical')

cll=0
fl=0
mcl=0
for images, labels in test_generator:
    preds = model.predict(images)
    for i in range(len(preds)):
         if np.argmax(labels[i])==np.argmax(preds[i]):
             if np.argmax(labels[i])==0:
                 cll+=1
             elif np.argmax(labels[i])==1:
                 fl+=1
             else:
                 mcl+=1
    break
pred=[cll,fl,mcl]
max_value = max(pred)
index =pred.index(max_value)
if index==0:
    print("CLL")
elif index==1:
    print("FL")
else:
    print("MCL")

shutil.rmtree('/Users/Zamzam/Desktop/malignant lymphoma classification/input/CLL/slicing')