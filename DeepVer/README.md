The deep verification model is based on ResNet-50.

The detection results produced by DeepDetection model on MITOSIS 2014 dataset are utilized for training the DeepVer model. Detection patches are cropped with a 96x96 size. 

We use Caffe framework to train the model. We follow the ImageNet example in $CAFFE_root/examples/imagenet to train our model.
You can refer to the "create_imagenet.sh" to create LMDB data for image patches, and "make_imagenet_mean.sh" to produce the image_mean.binaryproto. 

In test stage, we use the deploy.py to run the trained model on image patches.



 

