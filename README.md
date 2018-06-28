# DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks
By Chao Li, Xinggang Wang, Wenyu Liu and Longin Jan Latecki
Codes for our MIA paper "DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks".


Our detection model is based on Faster R-CNN model. We use the py-faster-rcnn framework. You need to firstly install the py-faster-rcnn  following the directions at: https://github.com/rbgirshick/py-faster-rcnn.

To download the model pre-trained on ImageNet:
./data/scripts/fetch_imagenet_models.sh
Rename the "VGG_CNN_M_1024.v2.caffemodel" to "VGG_CNN_M_1024_Scale.v2.caffemodel".





