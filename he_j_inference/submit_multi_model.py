import os
import sys
from keras import backend as K
import tensorflow as tf
from keras_model import ModelFactory
#from configparser import ConfigParser
import numpy as np
import pandas as pd
from PIL import Image
from random import shuffle
#from skimage.transform import resize
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(0)  # all new operations will be in test mode from now on

batch_size=16
class_names = [u'Normal', u'Abnormal']
#SIZE = 448
#model_id=0

base_models=[
#{'base_model_name':u'DenseNet121','SIZE':448,'model_id':0,'model_weights_file':'experiments/extra_data/densenet121_448/best_weights.h5'},
{'base_model_name':u'DenseNet169','SIZE':448,'model_id':7,'model_weights_file':'dense169_448v2.h5'},
{'base_model_name':u'InceptionV3','SIZE':499,'model_id':7,'model_weights_file':'inception499.h5'},
]
use_base_model_weights = False
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
enable_batch=False
write_prob=False

eval_csv = sys.argv[1]
df = pd.read_csv(eval_csv, names=['img', ], header=None)
eval_imgs = df.img.values.tolist()
print (eval_imgs[:10])
#eval_imgs=eval_imgs[:10]
if enable_batch:shuffle(eval_imgs)
patients={}
img_prob={}



def load_image(image_file,SIZE):
        image = Image.open(image_file)
        image=image.resize((SIZE,SIZE),Image.ANTIALIAS)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        #image_array = resize(image_array, (SIZE,SIZE))
        return image_array

def transform_batch_images(batch_x):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x


model_factory = ModelFactory()
for m in base_models:
    model = model_factory.get_model(
    class_names,
    model_name=m['base_model_name'],
    use_base_weights=use_base_model_weights,
    weights_path=m['model_weights_file'],
    input_shape=(m['SIZE'], m['SIZE'], 3),
    model_id=m['model_id'])
    print ('loaded {}'.format(m['base_model_name']))
    right=0
    #batch process
    if enable_batch:
        for i in range(len(eval_imgs)/batch_size):
            batch_x_path = eval_imgs[i * batch_size:(i + 1) * batch_size]
            batch_x = np.asarray([load_image(x_path,m['SIZE']) for x_path in batch_x_path])
            batch_x = transform_batch_images(batch_x)
            result = model.predict(batch_x)
            for j in range(batch_size):
                img_file=eval_imgs[i*batch_size+j]
                if img_file not in img_prob:img_prob[img_file]=[]
                img_prob[img_file].append(result[j][1])
                label = 1 if 'positive' in img_file else 0
                #print img_file,label,result[j][1]
                right+=int((int(result[j][1]>0.5)==label))
                patient=img_file[:-10]
                if patient not in patients:
                    patients[patient]=[]
                    patients[patient].append(result[j][1])
                else:
                    patients[patient].append(result[j][1])
        rem=len(eval_imgs)-len(eval_imgs)/batch_size*batch_size
        if rem>0:
            batch_x_path = eval_imgs[(i + 1) * batch_size:]
            batch_x = np.asarray([load_image(x_path,m['SIZE']) for x_path in batch_x_path])
            batch_x = transform_batch_images(batch_x)
            result = model.predict(batch_x)
            for j in range(rem):
                img_file=eval_imgs[len(eval_imgs)/batch_size*batch_size+j]
                if img_file not in img_prob:img_prob[img_file]=[]
                img_prob[img_file].append(result[j][1])
                label = 1 if 'positive' in img_file else 0
                #print img_file,label,result[j][1]
                right+=int((int(result[j][1]>0.5)==label))
                patient=img_file[:-10]
                if patient not in patients:
                    patients[patient]=[]
                    patients[patient].append(result[j][1])
                else:
                    patients[patient].append(result[j][1])
    else:
        for i in range(len(eval_imgs)):
            img_file=eval_imgs[i]
            #MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png
            image=Image.open(img_file)
            image=image.resize((m['SIZE'],m['SIZE']),Image.ANTIALIAS)
            image_array = np.asarray(image.convert("RGB"))
            image_array = image_array / 255.
            #image_array = resize(image_array, (SIZE,SIZE))
            image_array = (image_array- imagenet_mean) / imagenet_std
            x_data = np.expand_dims(np.asarray(image_array, dtype='float32'), 0)
            result = model.predict(x_data)
            if img_file not in img_prob:img_prob[img_file]=[]
            img_prob[img_file].append(result[0][1])
            label = 1 if 'positive' in img_file else 0
            #print img_file,label,result[0][1]
            right+=int((int(result[0][1]>0.5)==label))
            #output prob for [normal,abnormal],in csv file,0-normal,1-abnormal
    K.clear_session()
    tf.reset_default_graph()
    if write_prob:
        f1=open('pred_'+m['base_model_name']+'.csv','w')
        for fn in img_prob:
            f1.write(fn+','+str(img_prob[fn])+'\n')
        f1.close()
    print ('acc:{}'.format(float(right)/len(eval_imgs)))

#model ensemble
preds=np.zeros((len(base_models),len(eval_imgs),len(class_names)))
cnt=0
for fn in img_prob:
    #print img_prob[fn]
    #preds.append([1.0-preds_dict[fn],preds_dicpreds_dict[fn]])
    for i in range(len(base_models)):
        preds[i][cnt][0],preds[i][cnt][1]=1.0-float(img_prob[fn][i]),float(img_prob[fn][i])
    cnt+=1
weights=[0.43, 0.57]
weighted_predictions = np.zeros((len(eval_imgs), len(class_names)), dtype='float32')
for weight, prediction in zip(weights, preds):
    weighted_predictions += weight * prediction
cnt=0
for img_file in img_prob:
    patient=img_file[:-10]
    if patient not in patients:patients[patient]=[]
    patients[patient].append(weighted_predictions[cnt][1])
    cnt+=1
f=open(sys.argv[2],'w')
for patient in patients:
    img_num=len(patients[patient])
    average_score=sum(patients[patient])/img_num
    label=int(average_score>0.5)
    f.write(patient+','+str(label)+'\n')
f.close()
print ("done!")


