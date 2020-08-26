import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization

class CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='VALID', activation=None, 
        apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomConv, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)


    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)

        if self.activation:
            x = self.activation(x)

        return x


    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)
