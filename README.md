# DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks
By Chao Li, Xinggang Wang, Wenyu Liu and Longin Jan Latecki
Codes for our MIA paper "DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks".


Our detection model is based on Faster R-CNN model. We use the py-faster-rcnn framework. You need to firstly install the py-faster-rcnn  following the directions at: https://github.com/rbgirshick/py-faster-rcnn.

To download the model pre-trained on ImageNet:
./data/scripts/fetch_imagenet_models.sh
Rename the "VGG_CNN_M_1024.v2.caffemodel" to "VGG_CNN_M_1024_Scale.v2.caffemodel".



Train:
To train a deep detector on MITOSIS 2012 dataset:
./experiments/scripts/faster_rcnn_end2end_12.sh 0 VGG_CNN_M_1024_Scale mitos --set EXP_DIR XX

Test:
The test on MITOSIS 2012 dataset:
./experiments/scripts/faster_rcnn_end2end_test_12.sh 0 VGG_CNN_M_1024_Scale mitos --set EXP_DIR XX

The test on MITOSIS 2014 dataset:
./experiments/scripts/faster_rcnn_end2end_test_14.sh 0 VGG_CNN_M_1024_Scale mitos14 --set EXP_DIR XX

Deploying the detection model as a RPN model, discarding its following fast rcnn classification subnet.
cp lib/fast_rcnn/test.rpn.py lib/fast_rcnn/test.py
./experiments/scripts/faster_rcnn_end2end_test_rpn.sh 0 VGG_CNN_M_1024_rpn mitos --set EXP_DIR XX







