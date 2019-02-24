import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
import struct

DT_FLOAT = 1
DT_HALF = 19



def model_to_graph(model, ops, drop_scope = ['Switch', 'Merge'], 
                   output_node_names='network/output/Softmax',
                   verbose=True, 
                   save_file=None,
                   remove_training=True,
                   strip_nodes=False,
                   conver_to_half=False):
    '''Load graph from model'''
#     output_node_names = "network/output/Softmax"#"network/output/Reshape_1"#
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                model.sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names.split(",") # The output node names are used to select the usefull nodes
            ) 
    print("original graph nodes:",len(output_graph_def.node))
    if not remove_training:
        return output_graph_def
    
    output_graph_def = tf.graph_util.remove_training_nodes(
                        output_graph_def,
                        protected_nodes=None
                        )
    
    print("graph nodes after removing training:",len(output_graph_def.node))
    if not strip_nodes:
        return output_graph_def
    
    '''Drop nodes that TRT doesn't support'''
    new_output_graph_def = strip(output_graph_def, drop_scope)
    print("graph nodes after stripping:",len(new_output_graph_def.node))
    
    if not conver_to_half:
        return new_output_graph_def

    '''Convert model from float32 to float16 '''
    new_output_graph_def = float2half(new_output_graph_def)
#     for node in new_output_graph_def.node:
#         if node.name == 'network/input/Placeholder':
#             node.attr['dtype'].type = DT_HALF
        
#         if node.op in ops:
#     #     try:
#             node.attr['T'].type = DT_HALF
#     #     except:
#     #         pass
            
#         if node.op == 'Const':
#             node.attr['dtype'].type = DT_HALF
#             node.attr['value'].tensor.dtype = DT_HALF

#             floats = node.attr['value'].tensor.tensor_content

#             floats = struct.unpack('f' * int(len(floats) / 4), floats)
#             halfs = np.array(floats).astype(np.float16).view(np.uint16)
#             node.attr['value'].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)

    if save_file is not None:
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(save_file, "wb") as f:
            f.write(new_output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(new_output_graph_def.node))

    if verbose:
        for node in new_output_graph_def.node:
            print(node)

    return new_output_graph_def

def float2half(graph_def):
    '''Convert model from float32 to float16 '''
    for node in graph_def.node:
        if node.name == 'network/input/Placeholder':
            node.attr['dtype'].type = DT_HALF
        
        if node.op in ops:
    #     try:
            node.attr['T'].type = DT_HALF
    #     except:
    #         pass
            
        if node.op == 'Const':
            node.attr['dtype'].type = DT_HALF
            node.attr['value'].tensor.dtype = DT_HALF

            floats = node.attr['value'].tensor.tensor_content

            floats = struct.unpack('f' * int(len(floats) / 4), floats)
            halfs = np.array(floats).astype(np.float16).view(np.uint16)
            node.attr['value'].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
    return graph_def

def strip(input_graph, drop_scope):
    nodes_after_strip = []
    # save all nodes to a hashtable
    all_nodes_hash = {}
    for node in input_graph.node:
#         print ("{0} : {1} ( {2} )".format(node.name, node.op, node.input))
        all_nodes_hash[node.name] = node
    print("Nodes hash table is built!")
    # go through the hash table to get rid of switch node
    nodes_after_strip = []

    # create a new input node to get rid of split and normalization    
    # new_input_node = node_def_pb2.NodeDef()
    # new_input_node.CopyFrom(all_nodes_hash["network/input/Placeholder"])
    for node_name, node in all_nodes_hash.items():
        if node.op in drop_scope:
            continue
        if node.op == 'Const':
            # const node doesn't need input
            try:
                del node.input
            except:
                pass
        if 'Pre_Processing' in node.name:
            # get rid of all pre_processing nodes
            continue 
        if node.name == 'network/input/Placeholder_2':
            continue

        #if node.name == 'network/input/Placeholder':
        #    print("changing the input to batch_size = 1")
        #    node.attr['shape'].shape.dim[0].size = 1
        #    print("the changed size input node is:", node)

        old_input = node.input
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        for input_idx, input_name in enumerate(old_input):
            # preprocess the input name so that it can be used as hashtabel keys
            filtered_input_name = input_name.split(':')[0]
            if filtered_input_name.startswith('^'):
                filtered_input_name = filtered_input_name[1:]
            
            # change the input of the first Conv2D from concat to placeholder
            if input_name == 'network/mobilenet_encoder/Pre_Processing/concat' or input_name == 'network/shufflenet_encoder/Pre_Processing/concat':
                # note that the order of the inputs to a node matters!
                new_node.input[input_idx] = all_nodes_hash['network/input/Placeholder'].name

            # try:
                # if all_nodes_hash[filtered_input_name].op == 'Switch':
                #     # if one input to the current node is a Switch node, then get rid of that Switch node 
                #     # by changing the input of the current tobe the input to that Switch node.
                #     input_detoured = False
                #     for input_of_switch in all_nodes_hash[filtered_input_name].input:
                #         if input_of_switch != "network/input/Placeholder_2":
                #             new_node.input[input_idx] = input_of_switch
                #             input_detoured = True
                #             #new_node.input.append(input_of_input)
                #     if not input_detoured:
                #         #remove the input if it has two inputs and both are placeholder_2
                #         new_node.input.remove(input_name)
                
                # if all_nodes_hash[filtered_input_name].op == 'Merge':
                #     '''
                #     Merge node merges multiple input by forwarding the first recieved input as the output
                #     Thus we remove Merge nodes by always forwarding the first input to the Merge node to teh output
                #     '''
                #     new_node.input[input_idx] = all_nodes_hash[filtered_input_name].input[0]

                # detour the input of the node until the input is not Switch or Merge
            node_input = filtered_input_name
            if all_nodes_hash[node_input].op in drop_scope:
                while all_nodes_hash[node_input].op in drop_scope:
                    detoured = False
                    for i in range(len(all_nodes_hash[node_input].input)):
                        if all_nodes_hash[node_input].input[i] != "network/input/Placeholder_2":
                            detoured = True
                            node_input = all_nodes_hash[node_input].input[i]
                            break
                    if not detoured:
                        new_node.input.remove(input_name)
                        node_input = new_node.name # input is itself
                if detoured:
                    new_node.input[input_idx] = node_input
                else:
                    pass
            # except:
            #     print("Error node: ", new_node)
            #     raise NameError(filtered_input_name)
        nodes_after_strip.append(new_node)

    print("New graph is built!")
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph

