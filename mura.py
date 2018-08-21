from __future__ import absolute_import, division, print_function

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score)

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)


class Mura(object):
    """`MURA <https://stanfordmlgroup.github.io/projects/mura/>`_ Dataset :
    Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.
    """
    url = "https://cs.stanford.edu/group/mlgroup/mura-v1.0.zip"
    filename = "mura-v1.0.zip"
    md5_checksum = '4c36feddb7f5698c8bf291b912c438b1'
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'_(\w+)_patient')

    def __init__(self, image_file_names, y_true, y_pred1=None, y_pred2=None, y_pred3=None, y_pred4=None, y_pred5=None, output_path=None):
        self.imgs = image_file_names
        df_img = pd.Series(np.array(image_file_names), name='img')
        self.y_true = y_true
        df_true = pd.Series(np.array(y_true), name='y_true')
        self.y_pred1 = y_pred1
        self.y_pred2 = y_pred2
        self.y_pred3 = y_pred3
        self.y_pred4 = y_pred4
        self.y_pred5 = y_pred5
        self.output_path = output_path
        # number of unique classes
        self.patient = []
        self.study = []
        self.study_type = []
        self.image_num = []
        self.encounter = []
        self.valid =[]
        for img in image_file_names:
            self.patient.append(self._parse_patient(img))
            self.study.append(self._parse_study(img))
            self.image_num.append(self._parse_image(img))
            self.study_type.append(self._parse_study_type(img))
            self.valid.append(self._parse_valid(img))
            self.encounter.append("MURA-v1.1/{}/XR_{}/patient{}/study{}_{}".format(
                self._parse_valid(img),
                self._parse_study_type(img),
                self._parse_patient(img),
                self._parse_study(img),
                self._parse_normal(img)))

        self.classes = np.unique(self.y_true)
        df_patient = pd.Series(np.array(self.patient), name='patient')
        df_study = pd.Series(np.array(self.study), name='study')
        df_image_num = pd.Series(np.array(self.image_num), name='image_num')
        df_study_type = pd.Series(np.array(self.study_type), name='study_type')
        df_encounter = pd.Series(np.array(self.encounter), name='encounter')

        self.data = pd.concat(
            [
                df_img,
                df_encounter,
                df_true,
                df_patient,
        #        df_patient,
                df_study,
                df_image_num,
                df_study_type,
            ], axis=1)

   #     print(self.data)

        if self.y_pred1 is not None:
            self.y_pred1_probability = self.y_pred1.flatten()
            self.y_pred1 = self.y_pred1_probability.round().astype(int)
            df_y_pred1 = pd.Series(self.y_pred1, name='y_pred1')
            df_y_pred1_probability = pd.Series(self.y_pred1_probability, name='y_pred1_probs')
            self.data = pd.concat((self.data, df_y_pred1, df_y_pred1_probability), axis=1)

        if self.y_pred2 is not None:
            self.y_pred2_probability = self.y_pred2.flatten()
            self.y_pred2 = self.y_pred2_probability.round().astype(int)
            df_y_pred2 = pd.Series(self.y_pred2, name='y_pred2')
            df_y_pred2_probability = pd.Series(self.y_pred2_probability, name='y_pred2_probs')
            self.data = pd.concat((self.data, df_y_pred2, df_y_pred2_probability), axis=1)

        if self.y_pred3 is not None:
            self.y_pred3_probability = self.y_pred3.flatten()
            self.y_pred3 = self.y_pred3_probability.round().astype(int)
            df_y_pred3 = pd.Series(self.y_pred3, name='y_pred3')
            df_y_pred3_probability = pd.Series(self.y_pred3_probability, name='y_pred3_probs')
            self.data = pd.concat((self.data, df_y_pred3, df_y_pred3_probability), axis=1)

        if self.y_pred4 is not None:
            self.y_pred4_probability = self.y_pred4.flatten()
            self.y_pred4 = self.y_pred4_probability.round().astype(int)
            df_y_pred4 = pd.Series(self.y_pred4, name='y_pred4')
            df_y_pred4_probability = pd.Series(self.y_pred4_probability, name='y_pred4_probs')
            self.data = pd.concat((self.data, df_y_pred3, df_y_pred4_probability), axis=1)

        if self.y_pred5 is not None:
            self.y_pred5_probability = self.y_pred5.flatten()
            self.y_pred5 = self.y_pred5_probability.round().astype(int)
            df_y_pred5 = pd.Series(self.y_pred5, name='y_pred5')
            df_y_pred5_probability = pd.Series(self.y_pred5_probability, name='y_pred5_probs')
            self.data = pd.concat((self.data, df_y_pred5, df_y_pred5_probability), axis=1)

    def __len__(self):
        return len(self.imgs)

    def _parse_normal(self, img_filename):
        return "positive" if ("abnormal" in img_filename ) else "negative"

    def _parse_valid(self, img_filename):
        return "valid" if ("valid" in img_filename ) else "test"

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def metrics(self):
        return "per image metrics:\n\taccuracy : {:.3f}\tf1 : {:.3f}\tprecision : {:.3f}\trecall : {:.3f}\tcohen_kappa : {:.3f}".format(
            accuracy_score(self.y_true, self.y_pred2),
            f1_score(self.y_true, self.y_pred2),
            precision_score(self.y_true, self.y_pred2),
            recall_score(self.y_true, self.y_pred2),
            cohen_kappa_score(self.y_true, self.y_pred2), )

    def metrics_by_encounter(self):
        y_pred1 = self.data.groupby(['encounter'])['y_pred1_probs'].mean()
        y_pred2 = self.data.groupby(['encounter'])['y_pred2_probs'].mean()
        y_pred3 = self.data.groupby(['encounter'])['y_pred3_probs'].mean()
        y_pred4 = self.data.groupby(['encounter'])['y_pred4_probs'].mean()
        y_pred5 = self.data.groupby(['encounter'])['y_pred5_probs'].mean()
        week_group  = (list( self.data.groupby(['encounter']).groups.keys()))

        y_pred = ((y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5)/5).round()
        y_pred_ = (y_pred + 1) % 2
        #y_pred = y_pred.round()
        df_pred = pd.Series(np.array(y_pred_, np.int32), index=week_group)

        df_pred.to_csv(self.output_path)
        self.data.to_csv("data.csv", mode="a", header=True)

    #    print(df_pred)
        #df_filename = pd.Series(np.array(week_group))
   #     self.group_data = pd.concat([df_pred])

  #      self.group_data.to_csv(self.output_path)

        y_true = self.data.groupby(['encounter'])['y_true'].mean().round()
        return "per encounter metrics:\n\taccuracy : {:.3f}\tf1 : {:.3f}\tprecision : {:.3f}\trecall : {:.3f}\tcohen_kappa : {:.3f}".format(
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred), )

    def metrics_by_study_type(self):
        y_pred1 = self.data.groupby(['patient'])['y_pred1_probs'].mean()
        y_pred2 = self.data.groupby(['patient'])['y_pred2_probs'].mean()
        y_pred3 = self.data.groupby(['patient'])['y_pred3_probs'].mean()
        y_pred4 = self.data.groupby(['patient'])['y_pred4_probs'].mean()
        y_pred5 = self.data.groupby(['patient'])['y_pred5_probs'].mean()

        y_pred = ((y_pred1 + y_pred5 + y_pred3 + y_pred3 + y_pred5)/5).round()
#        y_pred = y_pred1
        y_true = self.data.groupby(['patient'])['y_true'].mean().round()

        self.data.to_csv("data.csv",mode="a",header=True)
        self.group_data =  pd.concat([self.data, y_pred, y_true,], axis=1)
        self.group_data.to_csv("group_data.csv", mode="a", header=True)

        return "per study_type metrics:\n\taccuracy : {:.3f}\tf1 : {:.3f}\tprecision : {:.3f}\trecall : {:.3f}\tcohen_kappa : {:.3f}".format(
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred), )
