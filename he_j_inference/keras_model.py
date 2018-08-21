#coding:utf-8
import importlib
import keras.backend as K
from keras.engine import InputSpec
from keras.layers import Input,Lambda,Dropout,Concatenate
from keras.activations import softmax
from keras.layers.core import Dense
from keras.layers import Conv2D,Average,MaxPooling2D,AveragePooling2D,Add,Flatten
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D,Multiply,LocallyConnected2D
from keras.models import Model
#import cv2
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from custom_layers import *
#from cbof import *
#from LearnToPayAttention import AttentionVGG
#batch_size=24

'''
class WildcatPool2d(Layer):
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter
    def __init__(self,kmax=0.2,kmin=0.2,alpha=0.7, **kwargs):
        #self.axis = axis
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha
        self.result = None
        super(WildcatPool2d, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        #print(input_shape)
        super(WildcatPool2d, self).build(input_shape)

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return K.cast(K.round(K.cast(n, dtype="float32")*
                   K.cast(k, dtype="float32")),dtype="int32")
        elif k > n:
            return n
        else:
            return int(k)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, x, **kwargs):
        batch_size, h, w, num_channels = K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],K.shape(x)[3]
        n = h * w  # number of regions
        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)
        x = K.reshape(x,(batch_size,n,num_channels))
        x = K.permute_dimensions(x,(0,2,1))
        x = tf.contrib.framework.sort(x,axis=-1,direction='DESCENDING')
        x_max = K.sum(x[:,:,:kmax],axis=-1,keepdims=False)/K.cast(kmax,dtype="float32")
        x_min = (K.sum(x[:,:,n-kmin:n],axis=-1,keepdims=False)
                     *self.alpha / K.cast(kmin,dtype="float32"))
        self.result = Average()([x_max,x_min])
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        #return K.int_shape(self.result)#(batch_size,num_classes)
        return tuple([input_shape[0],input_shape[3]])
'''




#################################################################

def target_category_loss(x, category_index, nb_classes):
    #batch_label=K.zeros((K.shape(x)[0],nb_classes))
    #batch_label=batch_label[:,category_index].assign(K.ones((K.shape(x)[0],)))
    batch_label=K.zeros((batch_size,nb_classes))
    batch_label=batch_label[:,category_index].assign(K.ones((batch_size,)))
    return tf.multiply(x, batch_label)

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x),axis=(1,2,3),keepdims=True)) + 1e-5)


class Get_grads(Layer):
    def __init__(self, **kwargs):
        #self.axis = axis
        self.result = None
        super(Get_grads, self).__init__(**kwargs)
    def build(self, input_shape):
        print(input_shape)
        super(Get_grads, self).build(input_shape)
    def call(self, x, **kwargs):
        self.result = normalize(K.gradients(x[0], x[1])[0])
        return self.result
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False

'''
def lr_multiply(base_model):
        for layer in base_model.layers:
            layer.W_learning_rate_multiplier = args_dict.lrmult_conv
            layer.b_learning_rate_multiplier = args_dict.lrmult_conv
'''


class ModelFactory:
    """
    Model facotry for Keras default models
    """
    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            DenseNet169=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
            DarkNet19_448=dict(
                input_shape=(224, 224, 3),
                module_name="darknet19_448",
                last_conv_layer="activation_260",
            ),
            Xception=dict(
                input_shape=(299, 299, 3),
                module_name="xception",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, class_names, model_name="DenseNet121"
                  , use_base_weights=True, weights_path=None
                  , input_shape=None, model_id=7):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                #f"keras.applications.{self.models_[model_name]['module_name']}"
                 "keras.applications."+self.models_[model_name]['module_name']
            ),
            model_name)

        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)
        base_model = None
        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        '''
        train bcnn with two steps:
        1.freeze base models,only train bilinear pooling and last fc layers with high lr=0.01
        2.train all layers with lr=0.001
        '''
        #setup_to_transfer_learn(base_model)

        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        conv_outputs = None #last conv output
        if model_name=="VGG16":
            block4_conv3 = layer_dict["block4_conv3"]
            block4_conv3_outputs = block4_conv3.output
            final_conv_layer = layer_dict["block5_conv3"]
            conv_outputs = final_conv_layer.output
        if model_name=="DenseNet121" or model_name=="DenseNet169":
            final_conv_layer = layer_dict["bn"]
            conv_outputs = final_conv_layer.output
        if model_name=="InceptionV3":
            final_conv_layer = layer_dict["mixed10"]
            conv_outputs = final_conv_layer.output
        if model_id == 0:
            x = base_model.output
            '''x = conv_outputs

            ##############SE module####################
            squeeze = GlobalAveragePooling2D()(x)
            excitation = Dense(units=512 // 4, activation='relu')(squeeze)
            #excitation = Activation('relu')(excitation)
            excitation = Dense(units=512, activation='sigmoid')(excitation)
            #excitation = Activation('sigmoid')(excitation)
            excitation = Reshape((1,1,512))(excitation)
            x = Multiply()([x,excitation])
            #x = SqueezeExcitation(512)(x)
            ###########################################
            spatial_att = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(x)
            spatial_att = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='loc')(spatial_att)
            x = Multiply()([x,spatial_att])
            x = GlobalAveragePooling2D()(x)'''
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 1:
            loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            #conv6 = LocallyConnected2D(32, (3, 3), activation='relu', padding='valid', name='conv6')(cccp)
            loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            x = base_model.output
            #x = conv_outputs
            #x = x * loc
            #AttributeError: 'Tensor' object has no attribute '_keras_history'此处不能用后端函数
            #x = Multiply()([x,loc])
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            #x = Dropout(rate=0.5)(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 2:
            #x = base_model.output
            x = conv_outputs
            #x = Multiply()([x,loc])
            z_l2=BilinearPooling()(x)
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(z_l2)
            #freeze_model = Model(inputs=img_input, output=predictions)
            #setup_to_transfer_learn(freeze_model)
            loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
        elif model_id == 3:
            loc0 = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            loc0 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc0)
            loc0 = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc0')(loc0)
            loc1 = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(block4_conv3_outputs)
            loc1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv7')(loc1)
            loc1 = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc1')(loc1)
            my_resize1 = Lambda(lambda x: K.repeat_elements(x, 2, axis=1))
            x = conv_outputs
            x_att = Multiply()([x,loc0])
            loc0 = my_resize1(loc0)
            my_resize2 = Lambda(lambda x: K.repeat_elements(x, 2, axis=2))
            loc0 = my_resize2(loc0)
            #loc = Add(name='loc')([loc0, loc1])
            loc = Average(name='loc')([loc0, loc1])
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            #x = GlobalAveragePooling2D()(x)
            x1 = GlobalMaxPooling2D()(x_att)
            #x = Dropout(rate=0.5)(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x1)
            x = my_resize1(x)
            x = my_resize2(x)
            x_merge = Concatenate(axis=-1)([x,block4_conv3_outputs])
            x_att1 = Multiply()([x_merge,loc])
            x2 = GlobalMaxPooling2D()(x_att1)
            predictions1 = Dense(len(class_names), activation="softmax", name="cls_pred1")(x2)
        elif model_id == 4:
            loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp')(conv_outputs)
            loc = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            #TypeError: Output tensors to a Model must be Keras tensors. Found: Tensor("Squeeze:0", shape=(?, 14, 14), dtype=float32)
            #loc = K.squeeze(loc,axis=3)
            x = conv_outputs
            x = AveragePooling2D(pool_size=(2, 2))(x)
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp2')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            x = NoisyAnd()(x)
            #x = GlobalMaxPooling2D()(x)
            #print predictions.shape
            #my_reshape = Lambda(lambda x: K.reshape(x, (-1, x.shape[3])))
            #x = my_reshape(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 5:
            loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            #x = base_model.output
            x = conv_outputs
            x = AveragePooling2D(pool_size=(2, 2))(x)
            x1 = AveragePooling2D(pool_size=(2, 2))(x)
            #x = Multiply()([x,loc])
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            x = Conv2D(2, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            x1 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cccp2')(x1)
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            x = Softmax4D()(x)
            #x = GlobalMaxPooling2D()(x)
            x1 = Softmax4D()(x1)
            x = MaxPooling2D(pool_size=(14, 14))(x)
            x1 = MaxPooling2D(pool_size=(7, 7))(x1)
            x = Flatten(name='flatten')(x)
            x1 = Flatten(name='flatten1')(x1)
            predictions = Recalc(axis=1, name='cls_pred0')(x)
            predictions1 = Recalc(axis=1, name='cls_pred1')(x1)
            #predictions1 = Recalc(axis=1)(x1)
            predictions = Average(name='cls_pred')([predictions, predictions1])
            #predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 6:
            x = base_model.output
            pred = Dense(len(class_names), activation="softmax", name="pred")(x)
            target_layer = Lambda(lambda x: target_category_loss(x, 1, 2),output_shape = target_category_loss_output_shape)
            gc = target_layer(pred)
            get_loss = Lambda(lambda x: K.sum(x,axis=1))
            loss = get_loss(gc)
            grads = Get_grads()([loss, conv_outputs])
            get_weights = Lambda(lambda x: K.mean(x, axis = (1, 2),keepdims=True))
            weights = get_weights(grads)
            my_resize1 = Lambda(lambda x: K.repeat_elements(x, conv_outputs.shape[1], axis=1))
            weights = my_resize1(weights)
            my_resize2 = Lambda(lambda x: K.repeat_elements(x, conv_outputs.shape[2], axis=2))
            weights = my_resize2(weights)
            grad_cam = Multiply()([conv_outputs,weights])
            loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(conv_outputs)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            #x = base_model.output
            #x = Multiply()([x,loc])
            #x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp1')(x)
            #x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(x)
            #x = GlobalAveragePooling2D()(x)
            #x = GlobalMaxPooling2D()(x)
            x = GlobalMaxPooling2D()(grad_cam)
            #x1 = MaxPooling2D(pool_size=(7, 7))(x1)
            #x = Flatten(name='flatten')(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 7:
            #loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            #loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            #loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            x = conv_outputs
            #x = Multiply()([x,loc])
            #num_maps=8
            classes=2
            #x = Conv2D(num_maps*classes, (1, 1), activation='relu', padding='same', name='cccp')(x)
            #x = ClassWisePool()(x)
            x = WildcatPool2d()(x)
            #x = LogSumExp()(x)
            #predictions = Recalc(axis=1, name='cls_pred')(x)#
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
            #predictions = Dense(len(class_names), activation='sigmoid', name='cls_pred')(x)
        elif model_id == 8:
            #loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            #loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            #loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            x = conv_outputs
            #x = Multiply()([x,loc])
            x = LogSumExp(r=1)(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 9:
            loc = Conv2D(512, (1, 1), activation='relu', padding='same', name='cccp0')(conv_outputs)
            loc = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6')(loc)
            loc = Conv2D(1, (1, 1), activation='relu', padding='same', name='loc')(loc)
            x = conv_outputs
            n_codewords=128
            x=BoF_Pooling(n_codewords, spatial_level=0)(x)
            predictions = Dense(len(class_names), activation="softmax", name="cls_pred")(x)
        elif model_id == 10:
            base_model=AttentionVGG(img_input, outputclasses=2, batchnorm=False, batchnormalizeinput=False).model
        model = Model(inputs=img_input, output=#base_model.output#predictions,#predictions1,
                                                predictions
                                                #loc

                                                )
        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            #print(f"load model weights_path: {weights_path}")
            print ("load model weights_path: {}".format(weights_path))
            model.load_weights(weights_path, by_name=True)
        return model
