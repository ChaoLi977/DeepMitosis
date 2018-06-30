The deep verification model is based on ResNet-50.

We use Caffe framework to train the model. We follow the ImageNet example in $CAFFE_root/examples/imagenet to prepare the lmdb format data and the mean file.

The detection results produced by DeepDetection model on MITOSIS 2014 dataset are utilized for training the DeepVer model. Detection patches are cropped with a 96x96 size.  

