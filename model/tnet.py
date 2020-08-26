import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from model.custom_conv import CustomConv
from model.custom_dense import CustomDense

class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConv(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = CustomConv(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = CustomConv(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.fc0 = CustomDense(512, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = CustomDense(256, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)


    def build(self, input_shape):
        self.K = input_shape[-1]

        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer,
                        trainable=True, name='weights')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                        trainable=True, name='bias')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)


    def call(self, x, training=None):
        input_x = x                                                     # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)                             # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)                                       # BxNx1024

        # Global features
        x = tf.reduce_max(x, axis=1)                                    # Bx1024

        # Fully-connected layers
        x = self.fc0(x, training=training)                              # Bx512
        x = self.fc1(x, training=training)                              # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)                                   # Bx1x256
        x = tf.matmul(x, self.w)                                  # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)


    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})


    @classmethod
    def from_config(cls, config):
        return cls(**config)
