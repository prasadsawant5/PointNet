import tensorflow as tf
from model.tnet import TNet
from model.custom_conv import CustomConv
from model.custom_dense import CustomDense
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model

def get_model(bn_momentum):
    pt_cloud = Input(shape=(None, 3), dtype=tf.float32, name='pt_cloud')    # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)         # for weight-sharing of conv
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(pt_cloud_transform)
    embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_64)
    embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64)

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    embed_64_transform = tf.expand_dims(embed_64_transform, axis=2)
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_64_transform)
    hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64)
    embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_128)
    embed_1024 = tf.squeeze(embed_1024, axis=2)

    # Global feature vector (B x N x 1024 -> B x 1024)
    global_descriptor = tf.reduce_max(embed_1024, axis=1)

    # FC layers to output k scores (B x 1024 -> B x 40)
    hidden_512 = CustomDense(512, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(global_descriptor)
    hidden_512 = Dropout(rate=0.3)(hidden_512)

    hidden_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(hidden_512)
    hidden_256 = Dropout(rate=0.3)(hidden_256)

    logits = CustomDense(40, apply_bn=False)(hidden_256)

    return Model(inputs=pt_cloud, outputs=logits)