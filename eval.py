from __future__ import absolute_import, division, print_function

from os import environ, getcwd
from os.path import join

import keras
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications import DenseNet169, InceptionResNetV2, DenseNet201
from keras.applications import NASNetMobile
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from mura import Mura

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
print("sklearn : {}".format(skl.__version__))

# Hyper-parameters / Globals
BATCH_SIZE = 8  # tweak to your GPUs capacity
IMG_HEIGHT = 420  # ResNetInceptionv2 & Xception like 299, ResNet50/VGG/Inception 224, NASM 331
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # blame theano
MODEL_TO_EVAL1 = './models/DenseNet169_221_NEW_HIST.hdf5'
MODEL_TO_EVAL2 = './models/DenseNet169_320_NEW_HIST.hdf5'
MODEL_TO_EVAL3 = './models/DenseNet169_420_NEW_HIST.hdf5'
MODEL_TO_EVAL4 = './models/DenseNet169_520_NEW_HIST.hdf5'
MODEL_TO_EVAL5 = './models/DenseNet169_620_NEW_HIST.hdf5'
#MODEL_TO_EVAL6 = './models/DenseNet169_420_SHOULDER_NEW_HIST.hdf5'
#MODEL_TO_EVAL7 = './models/DenseNet169_420_HAND_NEW_HIST.hdf5'

#MODEL_TO_EVAL3 = './models/DenseNet169_420_FOREARM.hdf5'
#MODEL_TO_EVAL4 = './models/DenseNet169_420_HAND.hdf5'
#MODEL_TO_EVAL5 = './models/DenseNet169_420_HUMERUS.hdf5'
#MODEL_TO_EVAL6 = './models/DenseNet169_420_SHOULDER.hdf5'
#MODEL_TO_EVAL7 = './models/DenseNet169_420_WRIST.hdf5'


DATA_DIR = 'MURA-v1.1/'
EVAL_CSV = 'valid.csv'
EVAL_DIR = 'data/val'


# load up our csv with validation factors
data_dir = join(getcwd(), DATA_DIR)
eval_csv = join(data_dir, EVAL_CSV)
df = pd.read_csv(eval_csv, names=['img', 'label'], header=None)
eval_imgs = df.img.values.tolist()
eval_labels = df.label.values.tolist()

eval_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = eval_datagen.flow_from_directory(
    EVAL_DIR, class_mode='binary', shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
n_samples = eval_generator.samples

base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL1)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
#print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred1 = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)

#IMG_HEIGHT = 520
#IMG_WIDTH  = 520
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
eval_generator = eval_datagen.flow_from_directory(
    EVAL_DIR, class_mode='binary', shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)

base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL2)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
#print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred2 = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)

IMG_HEIGHT = 420
IMG_WIDTH  = 420
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
eval_generator = eval_datagen.flow_from_directory(
    EVAL_DIR, class_mode='binary', shuffle=False,target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size =BATCH_SIZE)

base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL3)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
#print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred3 = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)


#IMG_HEIGHT = 520
#IMG_WIDTH  = 520
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
eval_generator = eval_datagen.flow_from_directory(
       EVAL_DIR, class_mode='binary', shuffle=False,target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL4)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
#print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred4 = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)

IMG_HEIGHT = 420
IMG_WIDTH  = 420
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
eval_generator = eval_datagen.flow_from_directory(
     EVAL_DIR, class_mode='binary', shuffle=False,target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL5)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
#print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred5 = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)

#print(y_pred)
df_filenames = pd.Series(np.array(eval_generator.filenames), name='filenames')
df_classes   = pd.Series(np.array(eval_generator.classes), name='classes')

#print(eval_generator.filenames)
#print(eval_generator.classes)

mura = Mura(eval_generator.filenames, y_true=eval_generator.classes, y_pred1=y_pred1, y_pred2=y_pred2, y_pred3=y_pred3, y_pred4=y_pred4, y_pred5=y_pred5)
print('==> Metrics with predict')
print(mura.metrics())
print(mura.metrics_by_encounter())
print(mura.metrics_by_study_type())
