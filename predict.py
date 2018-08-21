from __future__ import absolute_import, division, print_function

from os import environ, getcwd
from os.path import join

import shutil
import re
import os
import argparse
import keras
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications import DenseNet169, InceptionResNetV2, DenseNet201
from keras.applications import NASNetMobile
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy, kappa_error
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from custom_layers import *
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
BATCH_SIZE = 4  # tweak to your GPUs capacity
IMG_HEIGHT = 420  # ResNetInceptionv2 & Xception like 299, ResNet50/VGG/Inception 224, NASM 331
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # blame theano
MODEL_TO_EVAL1 = './models/DenseNet169_420_HUMERUS.hdf5'
MODEL_TO_EVAL2 = './models/DenseNet169_420_HAND.hdf5'
MODEL_TO_EVAL3 = './models/DenseNet169_420_FINGER.hdf5'
MODEL_TO_EVAL4 = './models/DenseNet169_420_FOREARM.hdf5'
MODEL_TO_EVAL5 = './models/DenseNet169_420_ELBOW.hdf5'
MODEL_TO_EVAL6 = './models/DenseNet169_420_SHOULDER.hdf5'
MODEL_TO_EVAL7 = './models/DenseNet169_420_WRIST.hdf5'
MODEL_TO_EVAL8 = './models/DenseNet169_420_NEW_HIST.hdf5'
DATA_DIR = 'data_val/'
EVAL_CSV = 'valid.csv'
EVAL_DIR = 'data/val/'

parser = argparse.ArgumentParser(description='Input Path')
parser.add_argument('input_filename',default='valid_image_paths.csv', type=str)
parser.add_argument('output_path', default='prediction.csv', type=str)
proc_data_dir = join(os.getcwd(), 'data/val/')
proc_train_dir = join(proc_data_dir, 'train')
proc_val_dir = join(proc_data_dir, 'val')


class ImageString(object):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, img_filename):

        self.img_filename = img_filename
        self.patient = self._parse_patient()
        self.study = self._parse_study()
        self.image_num = self._parse_image()
        self.study_type = self._parse_study_type()
        self.image = self._parse_image()
        self.normal = self._parse_normal()
        self.valid = self._parse_valid()


    def flat_file_name(self):
        return "{}_{}_patient{}_study{}_{}_image{}.png".format(self.valid,  self.study_type, self.patient, self.study,
                                                            self.normal, self.image)

    def _parse_patient(self):
        return int(self._patient_re.search(self.img_filename).group(1))

    def _parse_study(self):
        return int(self._study_re.search(self.img_filename).group(1))

    def _parse_image(self):
        return int(self._image_re.search(self.img_filename).group(1))

    def _parse_study_type(self):
        return self._study_type_re.search(self.img_filename).group(1)

    def _parse_normal(self):
        return "normal" if ("negative" in self.img_filename) else "abnormal"

    def _parse_normal_label(self):
        return 1 if("negative" in self.img_filename) else 0

    def _parse_valid(self):
        return "valid" if ("valid" in self.img_filename) else "test"

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

def eval(args=None):

    args= parser.parse_args()

    # load up our csv with validation factors
    data_dir = join(getcwd(), DATA_DIR)
    eval_csv = join(data_dir, EVAL_CSV)

    true_labels=[]

    ###########################################
    df = pd.read_csv(args.input_filename, names=['img', 'label'], header=None)
    samples = [tuple(x) for x in df.values]
 #   for img, label in samples:
 #       #assert ("negative" in img) is (label is 0)
 #       enc = ImageString(img)
 #       true_labels.append(enc._parse_normal_label())
 #       cat_dir = join(proc_val_dir, enc.normal)
 #       if not os.path.exists(cat_dir):
 #           os.makedirs(cat_dir)
 #       shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))


    ###########################################

    eval_datagen = ImageDataGenerator(rescale=1./255
#                                    , histogram_equalization=True
                                      )
    eval_generator = eval_datagen.flow_from_directory(
         EVAL_DIR, class_mode='binary', shuffle=False,target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
    n_samples = eval_generator.samples
    base_model = DenseNet169(input_shape=DIMS, weights='imagenet', include_top=False)  #weights='imagenet'
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
 #   x = WildcatPool2d()(x)

    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(MODEL_TO_EVAL8)
    model.compile(optimizer=Adam(lr=1e-3)
                  , loss=binary_crossentropy
#                  , loss=kappa_error
                  , metrics=['binary_accuracy'])
    score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
    print(model.metrics_names)
    print('==> Metrics with eval')
    print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
    y_pred = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)

#    print(y_pred)
#    df_filenames = pd.Series(np.array(eval_generator.filenames), name='filenames')
#    df_classes   = pd.Series(np.array(y_pred), name='classes')

#    prediction_data = pd.concat([df_filenames, df_classes,])
#    prediction_data.to_csv(args.output_path + "/prediction.csv")

    mura = Mura(eval_generator.filenames, y_true = eval_generator.classes, y_pred1=y_pred, y_pred2=y_pred, y_pred3=y_pred, y_pred4= y_pred, y_pred5= y_pred, output_path= args.output_path)
    print(mura.metrics_by_encounter())


if __name__ == '__main__':
    eval()
