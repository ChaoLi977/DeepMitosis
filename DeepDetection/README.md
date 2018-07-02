Our DeepDet model is based on faster-rcnn architecture.

Please consult the "readme_faster_rcnn" about the details of faster-rcnn.

We use the VGG_CNN_M_1024 model pre-trained on ImageNet. To download it and rename to our tailored model name:

```Shell
./data/scripts/fetch_imagenet_models.sh
mv ./data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel ./data/imagenet_models/VGG_CNN_M_1024_Scale.v2.caffemodel
```


### Train:
To train a deep detector on MITOSIS 2012 dataset:

```Shell
./experiments/scripts/faster_rcnn_end2end_12.sh 0 VGG_CNN_M_1024_Scale mitos --set EXP_DIR XX
```

### Test:
The test on MITOSIS 2012 dataset:

```Shell
./experiments/scripts/faster_rcnn_end2end_test_12.sh 0 VGG_CNN_M_1024_Scale mitos --set EXP_DIR XX
```

The test on MITOSIS 2014 dataset:

```Shell
./experiments/scripts/faster_rcnn_end2end_test_14.sh 0 VGG_CNN_M_1024_Scale mitos14 --set EXP_DIR XX
```

Deploying the trained DeepDet model as a RPN model, discarding its classification subnet after the RPN. 

Replace the "test.py" file with the RPN based "test.rpn.py" file, and then run the specific RPN shell file.

```Shell
cp lib/fast_rcnn/test.rpn.py lib/fast_rcnn/test.py
./experiments/scripts/faster_rcnn_end2end_test_rpn.sh 0 VGG_CNN_M_1024_rpn mitos --set EXP_DIR XX
```
