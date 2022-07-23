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
from tensorflow import keras
import os
import numpy as np
import shutil
import random
import image_slicer
from keras.layers import Activation

img_archive = "/Users/Zamzam/Desktop/GP"
'''
root_dir = img_archive + '/'
classes_dir = ['CLL', 'FL', 'MCL']
val_ratio = 0.25
test_ratio = 0.05
os.makedirs(root_dir+'train')
os.makedirs(root_dir+'val')
os.makedirs(root_dir+'test')

train_path=root_dir+'train/'
val_path=root_dir+'val/'
test_path=root_dir+'test/'

for cls in classes_dir:
    os.makedirs(train_path +'train' + cls)
    os.makedirs(val_path +'val' + cls)
    os.makedirs(test_path +'test' + cls)
    
    src = root_dir + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - val_ratio + test_ratio)), 
                                                               int(len(allFileNames)* (1 - test_ratio))])
    
    
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    i=1
    for path in train_FileNames:
        tiles=image_slicer.slice(path,16,save=False)
        image_slicer.save_tiles(tiles, directory=train_path+'train'+cls,prefix=str(i)+'_' ,format='png')
        i+=1
    i=1
    for path in val_FileNames:
        tiles=image_slicer.slice(path,16,save=False)
        image_slicer.save_tiles(tiles, directory=val_path+'val'+cls ,prefix=str(i)+'_' ,format='png')
        i+=1
    i=1
    for path in test_FileNames:
        tiles=image_slicer.slice(path,16,save=False)
        image_slicer.save_tiles(tiles, directory=test_path+'test'+cls ,prefix=str(i)+'_' ,format='png')
        i+=1
        

'''
batch_size = 16
target_size = (1388, 1040)



def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = color,
    rescale=1./255,
    rotation_range=45,
    horizontal_flip=True,
    fill_mode='reflect'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=img_archive + '/train',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    directory=img_archive + '/val',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')


plt.figure(figsize=(10, 10))
for images, labels in validation_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis("off")
    break


def ourModel():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                     input_shape=(1388,1040,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=32, kernel_size=(3,3),
              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3),
              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3),
              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(3))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
 
    model.compile(optimizer=Adam(0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy']) 
    model.summary()
    return model



model = ourModel()
epochs = 20
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1)


def scheduler(epoch):
  if epoch < 12:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.5 * (12 - epoch))


learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

with tf.device('/GPU:0'):
  model_train = model.fit(
      train_generator, epochs=epochs, validation_data=validation_generator,
      steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID, callbacks=[
          early_stopping  , learning_rate_scheduler]
  )


model.save("/Users/Zamzam/Desktop/malignant lymphoma classification/CNN.h5")


test_generator = test_datagen.flow_from_directory(
    directory=img_archive + '/test',
    target_size=target_size,
    class_mode='categorical')

my_model = tf.keras.models.load_model("/Users/Zamzam/Desktop/malignant lymphoma classification/CNN.h5")

test_eval = my_model.evaluate(test_generator)  # [0] loss, [1] acc

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1]*100, "%")

plt.figure(figsize=(10, 10))
for images, labels in test_generator:
    preds = my_model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(
            "Truth:"+str(np.argmax(labels[i]))+" Pred:"+str(np.argmax(preds[i])))
        plt.axis("off")
    break