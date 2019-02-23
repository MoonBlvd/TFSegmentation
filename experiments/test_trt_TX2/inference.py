import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import copy
import os 
from utils.od_utils import build_trt_pb, load_trt_pb, \
                           write_graph_tensorboard, segment
import sys
sys.path.append('../../')
from metrics.metrics import Metrics
from convert_ckpt_to_pb import float2half

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",help="model_name")
    parser.add_argument("-p","--model_path",help="path to the pb file")
    parser.add_argument("-o","--out_path",help="path to save the segmentation numpy")
    
    args = parser.parse_args()

    pb_path = args.model_path#"../fcn8s_mobilenet/checkpoints/best/final_model.pb"#"mobilenet_fcn8s.pb"#"unet_mobilenet"# "optimized_model.pb"#"mobilenet_fcn8s.pb"
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
    all_images = np.load('/media/DATA/UnrealLandingDataset/AirSimCollectData/CV_Random_Histogram/X_val.npy')
    all_labels = np.load('/media/DATA/UnrealLandingDataset/AirSimCollectData/CV_Random_Histogram/Y_val.npy')
    
    elipse = 0

    metrics = Metrics(nclasses=18)
    means = [73.29132098, 83.04442645, 72.5238962] # bgr
    for i in range(0, all_images.shape[0], 1):
#         try:
        # pre process        
        # subtract mean, normalize, then rgb to bgr
        image = all_images[i:i+1,:,:,:]
        new_image = copy.deepcopy(image).astype(float)
        new_image[0,:,:,0] = (image[0,:,:,2] - means[0])/255.0 #b
        new_image[0,:,:,1] = (image[0,:,:,1] - means[1])/255.0 #g
        new_image[0,:,:,2] = (image[0,:,:,0] - means[2])/255.0
        
        start = time.time()
        segmentation = segment(new_image, tf_sess)
        if i > 0:
            elipse += time.time() - start
        
#         print("segmentation: ", segmentation.shape)
        segmentation = np.argmax(segmentation, axis=1).astype(int)#tf.argmax(segmentation, axis=1, output_type=tf.int32)
        segmentation = segmentation.reshape((512, 512))#tf.reshape(segmentation,[512, 512])
        
        img_name = str(format(i,'06'))+".png"
        seg_img = Image.fromarray(segmentation)
        seg_img.save(os.path.join(args.save_path, img_name))
#         save_dir = os.path.join("fcn8s_mobilenet_trt",str(format(i,'06'))+".npy")
#         np.save(save_dir, segmentation)
        
        # update metrics
        label = all_labels[i:i+1,:,:]
        metrics.update_metrics(segmentation, label, 0, 0)

        if i%100 == 0:
            print(i)
#         except:
#             print("error")
#             pass
    print(elipse/(i))
    print("segmentation size:", segmentation.shape)
    nonignore = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    iou, mean_iou = metrics.compute_final_metrics(1, nonignore=nonignore)
    print("mean IOU: ", mean_iou)
    print("Per class IOU: ", iou)
if __name__ == "__main__":
    main()
