python -m tf2onnx.convert \
	--input fcn8s_mobilenet/checkpoints/best/optimized_model.pb \
	--inputs network/input/Placeholder:0 \
	--outputs network/output/Softmax:0 \
	--output fcn8s_mobilenet.onnx \
	--continue_on_error \
	--verbose \
	--fold_const >> fcn8s_mobilenet/checkpoints/best/print_graph.log

#python -m tf2onnx.convert \
#	--input fcn8s_shufflenet/checkpoints/best/frozen_model.pb \
#	--inputs network/input/Placeholder:0 \
#	--outputs network/output/Softmax:0 \
#	--output fcn8s_shufflenet.onnx \
#	--continue_on_error \
#	--verbose \
#	--fold_const >> fcn8s_shufflenet/checkpoints/best/print_graph.log
