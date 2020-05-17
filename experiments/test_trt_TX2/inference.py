import logging
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import copy
import os 
from utils.od_utils import build_trt_pb, load_trt_pb, \
                           write_graph_tensorboard, segment

import json
import sys
sys.path.append('../../')
from metrics.metrics import Metrics
from convert_ckpt_to_pb import float2half

import argparse
record_fields = ['command', 'environment', 'tag', 'uid', 'building', 'time', 'metric', 'misc']

output_records = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",help="model_name")
    parser.add_argument("-p","--model_path",help="path to the pb file")
    parser.add_argument("-o","--out_path",help="path to save the segmentation numpy")
    parser.add_argument("-im","--image_path",help="path to the numpy that rgb images are saved")
    parser.add_argument("-gt","--label_path",help="path to the numpy that labels are saved")
    
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

    tf_input = tf_sess.graph.get_tensor_by_name('network/input/Placeholder:0')
    tf_output = tf_sess.graph.get_tensor_by_name('network/output/ArgMax:0')


    logger.info('warming up the TRT graph with a dummy image')
    all_images = np.load(args.image_path)
    all_labels = np.load(args.label_path)
    print("------------------Data loaded!!------------------")
    #uid_name_map = []
    #with open('map_uid_img_name.txt','r') as f:
    #    for row in f:
    #        row = row.strip('\n')
    #        uid_name_map.append(row)
    #print(uid_name_map)  
    elipse = 0

    metrics = Metrics(nclasses=18)
    means = [73.29132098, 83.04442645, 72.5238962] # bgr
    print("------------------Start Test!!------------------")
    for i in range(0, all_images.shape[0], 1):
        # pre process        
        # subtract mean, normalize, then rgb to bgr
        image = all_images[i:i+1,:,:,:]
        new_image = copy.deepcopy(image).astype(float)
        new_image[0,:,:,0] = (image[0,:,:,2] - means[0])/255.0 #b
        new_image[0,:,:,1] = (image[0,:,:,1] - means[1])/255.0 #g
        new_image[0,:,:,2] = (image[0,:,:,0] - means[2])/255.0
        
        start = time.time()
        segmentation = tf_sess.run(tf_output, feed_dict={tf_input: new_image})
        elipse = time.time() - start
        
        # write records
        #img_name = uid_name_map[i] 
        #uid=int(img_name.split('-')[0])
        #curr_record = dict(uid=uid,
        #                   command='predict_segmentation', 
        #                   environment='tx2',
        #                   building=None,
        #                   time=elipse*1000,
        #                   metric=None,
        #                   misc=None,
        #                   tag=None)
        #output_records.append(curr_record)
        #print(curr_record)
        #print("segmentation: ", segmentation.shape)
        #segmentation = np.argmax(segmentation, axis=1)#.astype(int)#tf.argmax(segmentation, axis=1, output_type=tf.int32)
        #segmentation = segmentation.reshape((512, 512))#tf.reshape(segmentation,[512, 512])
        
        #if args.out_path is not None:
        #    seg_img = Image.fromarray(np.uint8(segmentation[0]))
        #    seg_img.save(os.path.join(args.out_path, img_name))
            

        # update metrics
        label = all_labels[i:i+1,:,:]
        metrics.update_metrics(segmentation, label, 0, 0)

        if i%100 == 0:
            print(i)
    
    #with open(args.model+'.json','w') as f:
    #    json.dump(output_records, f, indent=2)

#     print(elipse/(i+1))
    print("segmentation size:", segmentation.shape)
    nonignore = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    iou, mean_iou = metrics.compute_final_metrics(1, nonignore=nonignore)
    print("mean IOU: ", mean_iou)
    print("Per class IOU: ", iou)
if __name__ == "__main__":
    main()
