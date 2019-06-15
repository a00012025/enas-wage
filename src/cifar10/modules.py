import tensorflow as tf
import Quantize


def _get_variable(self, shape, name):
  with tf.name_scope(name) as scope:
    # self.W.append(tf.get_variable(name=name, shape=shape, initializer=self.initializer))

    # print 'W:', self.W[-1].device, scope, shape,
    if Quantize.bitsW <= 16:
      # manually clip and quantize W if needed
      # self.W_q_op.append(tf.assign(self.W[-1], Quantize.Q(self.W[-1], Quantize.bitsW)))
      # self.W_clip_op.append(tf.assign(self.W[-1],Quantize.C(self.W[-1],Quantize.bitsW)))

      # scale = Option.W_scale[len(self.W)-1]
      scale = 1.0
      print 'Scale:%d' % scale
      return Quantize.W(self.W[-1], scale)
      # return self.W_q[-1]
    else
      raise NotImplementedError
    #   a
    #   print ''
    #   return self.W[-1]

def _conv(self, x, ksize, c_out, stride=1, padding='SAME', name='conv'):
  c_in = x.get_shape().as_list()[1]
  W = self._get_variable([ksize, ksize, c_in, c_out], name)
  x = tf.nn.conv2d(x, W, self._arr(stride), padding=padding, data_format='NCHW', name=name)
  # self.H.append(x)
  return x

def _depth_conv(self, x, ksize, c_mul, c_out, stride=1, padding='SAME', name='conv'):
  c_in = x.get_shape().as_list()[1]
  W_depth = self._get_variable([ksize, ksize, c_in, c_mul], name)
  W_point = self._get_variable([1, 1, c_in * c_mul, c_out], name)
  x = tf.nn.separable_conv2d(x, W_depth, W_point, self._arr(stride), padding=padding, data_format='NCHW', name=name)
  # self.H.append(x)
  return x

def _fc(self, x, c_out, name='fc'):
  c_in = x.get_shape().as_list()[1]
  W = self._get_variable([c_in, c_out], name)
  x = tf.matmul(x, W)
  # self.H.append(x)
  return x

def _batch_norm(self, x, data_format='NCHW'):
  x = batch_norm(x, center=True, scale=True, is_training=self.is_training, decay=0.9, epsilon=1e-5, fused=True, data_format=data_format)
  # self.H.append(x)
  return x
