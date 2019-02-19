import logging
import tensorflow as tf
import numpy as np

from utils.od_utils import build_trt_pb, load_trt_pb, \
                           write_graph_tensorboard, segment

def main():

    pb_path = "final_model.pb"#"frozen_model.pb"# "optimized_model.pb"#"mobilenet_fcn8s.pb"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Ask tensorflow logger not to propagate logs to parent (which causes
    # duplicated logging)
    logging.getLogger('tensorflow').propagate = False

    logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)




    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    # if args.do_tensorboard:
    #     logger.info('writing graph summary to TensorBoard')
    #     write_graph_tensorboard(tf_sess, log_path)

    logger.info('warming up the TRT graph with a dummy image')
    # od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'
    dummy_img = np.zeros((8, 512, 512, 3), dtype=np.uint8)
    segmentation = segment(dummy_img, tf_sess)
    print("segmentation size:", segmentation.shape)

if __name__ == "__main__":
    main()
