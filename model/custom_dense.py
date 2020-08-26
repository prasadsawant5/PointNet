import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Dense

class CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)