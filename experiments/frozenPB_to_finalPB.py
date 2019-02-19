import argparse

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

def print_graph(input_graph):
    for node in input_graph.node:
        print ("{0} : {1} ( {2} )".format(node.name, node.op, node.input))

def strip(input_graph, drop_scope):
    nodes_after_strip = []
    # save all nodes to a hashtable
    all_nodes_hash = {}
    for node in input_graph.node:
        print ("{0} : {1} ( {2} )".format(node.name, node.op, node.input))
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

        if node.name == 'network/input/Placeholder':
            print("changing the input to batch_size = 1")
            node.attr['shape'].shape.dim[0].size = 1
            print("the changed size input node is:", node)

        old_input = node.input
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        for input_idx, input_name in enumerate(old_input):
            # preprocess the input name so that it can be used as hashtabel keys
            filtered_input_name = input_name.split(':')[0]
            if filtered_input_name.startswith('^'):
                filtered_input_name = filtered_input_name[1:]
            
            # change the input of the first Conv2D from concat to placeholder
            if input_name == 'network/mobilenet_encoder/Pre_Processing/concat':
                # note that the order of the inputs to a node matters!
                new_node.input[input_idx] = all_nodes_hash['network/input/Placeholder'].name

            try:
                if all_nodes_hash[filtered_input_name].op == 'Switch':
                    # if one input to the current node is a Switch node, then get rid of that Switch node 
                    # by changing the input of the current tobe the input to that Switch node.
                    input_detoured = False
                    for input_of_switch in all_nodes_hash[filtered_input_name].input:
                        if input_of_switch != "network/input/Placeholder_2":
                            new_node.input[input_idx] = input_of_switch
                            input_detoured = True
                            #new_node.input.append(input_of_input)
                    if not input_detoured:
                        #remove the input if it has two inputs and both are placeholder_2
                        new_node.input.remove(input_name)
                
                if all_nodes_hash[filtered_input_name].op == 'Merge':
                    '''
                    Merge node merges multiple input by forwarding the first recieved input as the output
                    Thus we remove Merge nodes by always forwarding the first input to the Merge node to teh output
                    '''
                    new_node.input[input_idx] = all_nodes_hash[filtered_input_name].input[0]

            except:
                print("Error node: ", new_node)
                raise NameError(filtered_input_name)
        nodes_after_strip.append(new_node)

    print("New graph is built!")
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-graph', action='store', dest='input_graph')
    parser.add_argument('--input-binary', action='store_true', default=True, dest='input_binary')
    parser.add_argument('--output-graph', action='store', dest='output_graph')
    parser.add_argument('--output-binary', action='store_true', dest='output_binary', default=True)

    args = parser.parse_args()

    input_graph = args.input_graph
    input_binary = args.input_binary
    output_graph = args.output_graph
    output_binary = args.output_binary

    if not tf.gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return

    input_graph_def = tf.GraphDef()
    mode = "rb" if input_binary else "r"
    with tf.gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read().decode("utf-8"), input_graph_def)

    print ("Before:")
    print_graph(input_graph_def)
    output_graph_def = strip(input_graph_def, drop_scope=['Switch', 'Merge'])
    print ("After:")
    print_graph(output_graph_def)

    if output_binary:
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with tf.gfile.GFile(output_graph, "w") as f:
            f.write(text_format.MessageToString(output_graph_def))
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()

