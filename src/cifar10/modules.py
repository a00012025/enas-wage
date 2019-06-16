import src.cifar10.Quantize as Quantize
import src.cifar10.myInitializer as myInitializer
import src.cifar10.Option as Option

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

W_q_op = []
W_clip_op = []

def arr(stride_or_ksize, data_format='NCHW'):
  # data format NCHW
  if data_format == 'NCHW':
    return [1, 1, stride_or_ksize, stride_or_ksize]
  elif data_format == 'NHWC':
    return [1, stride_or_ksize, stride_or_ksize, 1]
  else:
      raise NotImplementedError

def get_variable(shape, name):
  with tf.name_scope(name) as scope:
    w = tf.get_variable(
        name=name, shape=shape,
        initializer=myInitializer.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True
        )
    )

    if Quantize.bitsW <= 16:
      # manually clip and quantize W if needed
      W_q_op.append(tf.assign(w, Quantize.Q(w, Quantize.bitsW)))
      W_clip_op.append(tf.assign(w,Quantize.C(w, Quantize.bitsW)))

      scale = Option.W_scale[-1]
      return Quantize.W(w, scale)
      # return W_q[-1]
    else:
      raise NotImplementedError

def conv(x, ksize, c_out, stride=1, padding='SAME', data_format='NCHW', name='conv'):
  c_in = x.get_shape()[1 if data_format=='NCHW' else 3]
  W = get_variable([ksize, ksize, c_in, c_out], name)
  x = tf.nn.conv2d(x, W, arr(stride, data_format), padding=padding, data_format=data_format, name=name)\
  return x

def depth_conv(x, ksize, c_mul, c_out, stride=1, padding='SAME', data_format='NCHW', name='depth_conv'):
  c_in = x.get_shape()[1 if data_format=='NCHW' else 3]
  W_depth = get_variable([ksize, ksize, c_in, c_mul], name+'-depth')
  W_point = get_variable([1, 1, c_in * c_mul, c_out], name+'-point')
  x = tf.nn.separable_conv2d(x, W_depth, W_point, arr(stride, data_format), padding=padding, data_format=data_format, name=name)
  return x

def fc(x, c_out, name='fc'):
  c_in = x.get_shape().as_list()[1]
  W = get_variable([c_in, c_out], name)
  x = tf.matmul(x, W)
  return x

def batch_norm(x, is_training, data_format='NCHW'):
  x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, decay=0.9, epsilon=1e-5, fused=True, data_format=data_format)
  return x

def QA(x):
  if Option.bitsA <= 16:
    x = Quantize.A(x)
  return x

def QE(x):
  if Option.bitsE <= 16:
    x = Quantize.E(x)
  return x

def activation(x):
  x = tf.nn.relu(x)
  x = QE(x)
  x = QA(x)
  return x

def pool(x, mtype, ksize, stride=1, padding='SAME', data_format='NCHW'):
  if mtype == 'MAX':
    x = tf.nn.max_pool(x, arr(ksize, data_format), arr(stride, data_format), 
                       padding=padding, data_format=data_format)
  elif mtype == 'AVG':
    x = tf.nn.avg_pool(x, arr(ksize, data_format), arr(stride, data_format),
                       padding=padding, data_format=data_format)
  else:
    assert False, ('Invalid pooling type:' + mtype)
  return x

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[1, 3, 32, 32])
    x_conv = conv(x, 3, 5)
    x_depth_conv = depth_conv(x, 3, 4, 5)
    x_batch_norm = batch_norm(x, True)
    x_actv = activation(x)
    y = tf.placeholder(tf.float32, shape=[1, 32])
    y_fc = fc(y, 64)
