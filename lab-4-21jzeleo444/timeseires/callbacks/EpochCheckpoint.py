# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:34:28 2021

@author: arif
"""

import tensorflow as tf

class EpochCheckpoint:
    def __init__(self, checkpoints=None, every=5, startAt=0):
        # store the image data format
        self.checkpoints = checkpoints
        self.every=every
        self.startAt=startAt
    
    def EpochCheckpoin(self, checkpoints,every):
    # apply the Keras utility function that correctly rearranges
    # the dimensions of the image
        
        return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints,save_best_only=True, save_weights_only=True, verbose=1,save_freq=every)
 # Create a callback that saves the model's weights
