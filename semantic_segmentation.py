import sys
import tensorflow as tf
import numpy as np
from models.fcn8s_mobilenet import FCN8sMobileNet

from train import *
from test import *
from utils.misc import timeit

import os
import pickle
from utils.misc import calculate_flops
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Segmentor():
    def __init__(self, args):
        self.args = args

        # Get the class from globals by selecting it by arguments
        self.model = FCN8sMobileNet
        self.saver_best = tf.train.Saver(max_to_keep=1,
                                         save_relative_paths=True)
        
        # Reset the graph
        tf.reset_default_graph()

        # Create the sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Create Model class and build it
        with self.sess.as_default():
            self.build_model()
        
    @timeit
    def build_model(self):
        print('Building Test Network...')
        with tf.variable_scope('network') as scope:
            self.train_model = None
            self.model = self.model(self.args)
            self.model.build()
            calculate_flops()

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        self.load_best_model()
        
    def load_best_model(self):
        """
        Load the best model checkpoint
        :return:
        """
        print("loading a checkpoint for BEST ONE...")
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_best_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver_best.restore(self.sess, latest_checkpoint)
        else:
            print("ERROR NO best checkpoint found")
            exit(-1)
        print("BEST MODEL LOADED..")
                
    @timeit
    def run(self, image):
        """
        Initiate the Graph, sess, model, operator
        :return:
        """
        print("Agent is running now...\n\n")
        x_batch = np.expand_dims(image, axis=0) # (1, H, W, 3)?    
        y_batch = x_batch[:,:,:,0] # placeholder, value doesn't matter

        feed_dict = {
                    self.model.x_pl: x_batch,
                    self.model.y_pl: y_batch,
                    self.model.is_training: False
                    }

        # run the feed_forward
        segmentation = self.sess.run(
            [self.model.out_argmax],
            feed_dict=feed_dict)
            
        return segmentation

    def close_session(self):
        self.sess.close()
        print("\nAgent is exited...\n")