import sys
import tensorflow as tf
import numpy as np
from models.fcn8s_mobilenet import FCN8sMobileNet

from agent import Agent

class Segmentor():
    def __init__(self, args):
        self.args = args

        # Get the class from globals by selecting it by arguments
        self.model = 
        self.operator = globals()[args.operator]

        self.sess = None
        
    @timeit
    def build_model(self):

        
        else:  # inference phase
            print('Building Test Network')
            with tf.variable_scope('network') as scope:
                self.train_model = None
                self.model = self.model(self.args)
                self.model.build()
                calculate_flops()
                
    def semantic_segment(self, image):
    