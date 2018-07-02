Our DeepDet model is based on faster-rcnn architecture.

Please consult the "readme_faster_rcnn" about the details of faster-rcnn.

To download the model pre-trained on ImageNet:

```Shell
./data/scripts/fetch_imagenet_models.sh
```

Rename the "VGG_CNN_M_1024.v2.caffemodel" to "VGG_CNN_M_1024_Scale.v2.caffemodel".



### Train:
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
