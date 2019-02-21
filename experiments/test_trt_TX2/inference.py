import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import time

from utils.od_utils import build_trt_pb, load_trt_pb, \
                           write_graph_tensorboard, segment

def main():

    pb_path = "mobilenet_fcn8s.pb"#"frozen_model.pb"# "optimized_model.pb"#"mobilenet_fcn8s.pb"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Ask tensorflow logger not to propagate logs to parent (which causes
    # duplicated logging)
    logging.getLogger('tensorflow').propagate = False

#     build_trt_pb(model_name, pb_path, download_dir='data')
    
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
    # dummy_img = np.zeros((8, 512, 512, 3), dtype=np.uint8)
    all_images = np.load('/media/DATA/UnrealLandingDataset/AirSimCollectData/X_val.npy')
    
    elipse = 0

    for i in range(0, all_images.shape[0], 1):
        try:
            image = all_images[i:i+1,:,:,:].astype('float16')
            start = time.time()
            segmentation = segment(image, tf_sess)
            elipse += time.time() - start
            if i%100 == 0:
                print(i)
        except:
            pass
    print(elipse/i)
    print("segmentation size:", segmentation.shape)

if __name__ == "__main__":
    main()
