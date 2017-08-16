from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from layers.convolution import conv2d_transpose, conv2d_f

import tensorflow as tf


class FCN8s(BasicModel):
    """
    FCN8s Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        self.params.pretrained_path = self.params.pretrained_path
        # init network layers
        self.upscore2 = None
        self.score_feed1 = None
        self.fuse_feed1 = None
        self.upscore4 = None
        self.score_feed2 = None
        self.fuse_feed2 = None
        self.upscore8 = None

    def build(self):
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()

    def init_network(self):
        """
        Building the Network here
        :return:
        """

        # Init and build a VGG16 as an encoder
        encoder = VGG16(x_input=self.x_pl,
                        num_classes=self.params.num_classes,
                        pretrained_path=self.params.pretrained_path,
                        train_flag=self.is_training,
                        reduced_flag=False,
                        weight_decay=self.args.weight_decay)
        encoder.build()

        # Build Decoding part
        with tf.name_scope('upscore_2s'):
            self.upscore2 = conv2d_transpose('upscore2', encoder.score_fr, tf.shape(encoder.feed1).as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=encoder.wd)
            self.score_feed1 = conv2d_f('score_feed1', encoder.feed1, self.params.num_classes, kernel_size=(1, 1),
                                        l2_strength=encoder.wd)
            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            self.upscore4 = conv2d_transpose('upscore4', self.fuse_feed1, tf.shape(encoder.feed2).as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=encoder.wd)
            self.score_feed2 = conv2d_f('score_feed2', encoder.feed2, self.params.num_classes, kernel_size=(1, 1),
                                        l2_strength=encoder.wd)
            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            self.upscore8 = conv2d_transpose('upscore8', self.fuse_feed2, tf.shape(self.x_pl).as_list(),
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=encoder.wd)

        self.logits = self.upscore8