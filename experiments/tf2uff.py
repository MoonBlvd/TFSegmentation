import tensorflow as tf
import uff

output_names = ['network/output/Softmax']
frozen_graph_filename = 'fcn8s_mobilenet/checkpoints/best/final_model.pb'

# convert frozen graph to uff
uff_model = uff.from_tensorflow_frozen_model(frozen_graph_filename, output_names, output_filename='mobilenet_fcn8s.uff')

