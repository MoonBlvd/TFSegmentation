import os, argparse

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--pb_file", type=str, default="", help="frozen graph file")
args = parser.parse_args()

#with tf.Session(graph=tf.Graph()) as sess:
gf = tf.GraphDef()
gf.ParseFromString(open(args.pb_file,'rb').read())
for n in gf.node:
    print(n.name + '=>' +  n.op)

