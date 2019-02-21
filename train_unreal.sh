#python main.py --load_config=unet_mobilenet_train.yaml train Train UNetMobileNet >&1 | tee unet_mobilenet_train.log
#python main.py --load_config=unet_mobilenet_train.yaml train Train UNetMobileNet
python main.py --load_config=unet_shufflenet_train.yaml train Train UNetShuffleNet
python main.py --load_config=fcn8s_mobilenet_train.yaml train Train FCN8sMobileNet
python main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet
python main.py --load_config=unet_mobilenet_train.yaml train Train UNetMobileNet

