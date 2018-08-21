#coding:utf-8
import keras.backend as K
from keras.engine import InputSpec
from keras.layers import Input,Lambda,Dropout,Concatenate
from keras.activations import softmax
from keras.layers.core import Dense
from keras.layers import Conv2D,Average,MaxPooling2D,AveragePooling2D,Add,Flatten
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D,Multiply,LocallyConnected2D
from keras.layers import Activation,Reshape,Multiply, multiply
from keras.models import Model
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf



class NoisyAnd(Layer):
    '''
    https://github.com/dancsalo/TensorBase/blob/master/tensorbase/base.py
    Arguments:
    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.
    Returns:
    '''
    def __init__(self, a_init=1.0, b_init=0.0, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        super(NoisyAnd, self).__init__(**kwargs)
        self.a_init=a_init
        self.b_init=b_init
        #self.output_dim=output_dim
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        a = self.a_init * 1.0
        self.a = K.variable(a, name='{}_a'.format(self.name))
        b = self.b_init * np.ones((1,input_shape[self.axis]))
        self.b = K.variable(b, name='{}_a'.format(self.name))
        self.trainable_weights = [self.a, self.b]
        super(NoisyAnd, self).build(input_shape)
    def call(self, x, mask=None):
        mean = K.mean(x, axis=[1,2],keepdims=False)
        #mean = K.mean(mean, axis=2,keepdims=False)
        output = (K.sigmoid(self.a * (mean - self.b)) - K.sigmoid(-self.a * self.b))/ (
                K.sigmoid(self.a * (1 - self.b)) - K.sigmoid(-self.a * self.b))
        #output = softmax(output)
        output = K.reshape(output,(-1, x.shape[3]))
        #return x-self.a-self.b
        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)
    def build(self,input_shape):
        pass
    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s
    def compute_output_shape(self, input_shape):
        return input_shape

class Recalc(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Recalc, self).__init__(**kwargs)
    def build(self,input_shape):
        pass
    def call(self, x,mask=None):
        #print x.shape
        response = K.reshape(x[:,self.axis], (-1,1))
        #print K.concatenate([1-response, response], axis=self.axis).shape
        return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        #return e / s
    def compute_output_shape(self, input_shape):
        return input_shape
        #axis_index = self.axis % len(input_shape)
        #return tuple([input_shape[i] for i in range(len(input_shape)) \
        #              if i != axis_index ])



class BilinearPooling(Layer):
    '''
    bilinear pooling 
    https://github.com/abhaydoke09/Bilinear-CNN-TensorFlow/blob/master/core/bcnn_finetuning.py

    '''
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        self.z_l2 = None
        super(BilinearPooling, self).__init__(**kwargs)
    def build(self,input_shape):
        pass
    def call(self, x,mask=None):
            ''' Reshape conv5_3 from [batch_size, height, width, number_of_filters] 
            to [batch_size, number_of_filters, height, width]'''
            conv5_3 = tf.transpose(x, perm=[0,3,1,2])
            ''' Reshape conv5_3 from [batch_size, number_of_filters, height*width]'''
            conv5_3 = tf.reshape(conv5_3,[-1,512,784])
            ''' A temporary variable which holds the transpose of conv5_3'''
            conv5_3_T = tf.transpose(conv5_3, perm=[0,2,1])
            '''Matrix multiplication [batch_size,512,784] x [batch_size,784,512] '''
            phi_I = tf.matmul(conv5_3, conv5_3_T)
            '''Reshape from [batch_size,512,512] to [batch_size, 512*512] '''
            phi_I = tf.reshape(phi_I,[-1,512*512])
            print('Shape of phi_I after reshape', phi_I.get_shape())
            phi_I = tf.divide(phi_I,784.0)
            print('Shape of phi_I after division', phi_I.get_shape())
            '''Take signed square root of phi_I'''
            y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
            print('Shape of y_ssqrt', y_ssqrt.get_shape())
            '''Apply l2 normalization'''
            self.z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
            print('Shape of z_l2', self.z_l2.get_shape())
            return self.z_l2
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.z_l2)

#################################################################



#################################################################
'''
https://github.com/ameya005/Deep-Segmentation/blob/master/test_logsum.py

'''
class LogSumExp(Layer):
    #initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self,r=3, **kwargs):
        #self.axis = axis
        self.r=r
        self.result = None
        super(LogSumExp, self).__init__(**kwargs)
    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(LogSumExp, self).build(input_shape)
    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, x, **kwargs):
        shape = K.int_shape(x)
        print(shape)
        self.result = (1./self.r)*K.log((1./(shape[1]*shape[2]))*
                      K.sum( K.exp(self.r*x), axis=[1,2]))
        return self.result
    # return output shape
    def compute_output_shape(self, input_shape):
        #shape = list(input_shape)
        #return tuple([shape[0],shape[-1]])
        return K.int_shape(self.result)





#########WILDCAT#####################
'''
https://github.com/durandtibo/wildcat.pytorch/blob/master/wildcat/pooling.py

'''
class ClassWisePool(Layer):
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter
    def __init__(self,num_maps=8, **kwargs):
        #self.axis = axis
        self.num_maps=num_maps
        self.result = None
        super(ClassWisePool, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        #print(input_shape)
        super(ClassWisePool, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, x, **kwargs):
        batch_size, h, w, num_channels = K.shape(x)[0],K.shape(x)[1],K.shape(x)[2],K.shape(x)[3]
        num_outputs = num_channels / self.num_maps
        x=K.reshape(x,(batch_size, h, w, num_outputs, self.num_maps))
        x=K.sum(x,axis=4,keepdims=False)
        self.result = x/self.num_maps
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

###########################################################

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
#################################################################



###############################################################
'''
implement channel-wise attention
https://github.com/yoheikikuta/senet-keras/blob/master/model/SEResNeXt.py

'''


class SqueezeExcitation(Layer):
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter
    def __init__(self,out_dim,reduction_ratio=4, **kwargs):
        self.out_dim=out_dim
        self.ratio=reduction_ratio # ratio of channel reduction in SE module
        self.result = None
        super(SqueezeExcitation, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        #print(input_shape)
        super(SqueezeExcitation, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, x, **kwargs):
        '''
        SE module performs inter-channel weighting.
        '''
        squeeze = GlobalAveragePooling2D()(x)
        excitation = Dense(units=self.out_dim // self.ratio)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=self.out_dim)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1,1,self.out_dim))(excitation)
        self.result = multiply([x,excitation])
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return input_shape
