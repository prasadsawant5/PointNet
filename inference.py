import os
import sys
import argparse

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from model.model import get_model
from utils.visualize import plot

WEIGHTS = os.path.join('saved-model', 'weights')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='CLI for inference')
    PARSER.add_argument('file', type=str, help='Image on which to perform inference')

    ARGS = PARSER.parse_args()

    FILE = ARGS.file

    # Load point cloud
    pt_cloud = np.load(FILE)
    pt_cloud = np.expand_dims(pt_cloud, axis=0)


    # Load model and perform inference
    model = get_model(bn_momentum=None)
    model.load_weights(WEIGHTS)
    logits = model(pt_cloud, training=False)
    logits = tf.squeeze(logits, axis=0)
    probs = tf.math.sigmoid(logits)

    # Print top k predictions and visualize (if input arg)
    class_map = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle',
                6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain',
                12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box',
                17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel',
                22: 'monitor', 23: 'night_stand', 24: 'person', 25: 'piano', 26: 'plant',
                27: 'radio', 28: 'range_hood', 29: 'sink', 30: 'sofa', 31: 'stairs',
                32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase',
                38: 'wardrobe', 39: 'xbox'}
    max_idxs = tf.argsort(probs, direction='DESCENDING')
    print('Object\tProbability')
    for i in range(5):
        obj_id = max_idxs[i].numpy()
        print(f'{class_map[obj_id]}\t{probs[obj_id]}')

    plot(FILE)