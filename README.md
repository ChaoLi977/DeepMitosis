# DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks
By Chao Li, Xinggang Wang(http://www.xinggangw.info/), Wenyu Liu(http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu) and Longin Jan Latecki(https://cis.temple.edu/~latecki/)

Codes for our MIA (Medical Image Analysis) paper "DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks". Please see the paper(https://www.sciencedirect.com/science/article/pii/S1361841517301834) for more details.

### Citing DeepMitosis

If you find DeepMitosis useful in your research, please consider citing:
    
    @article{li2018deepmitosis,
  title={DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks},
  author={Li, Chao and Wang, Xinggang and Liu, Wenyu and Latecki, Longin Jan},
  journal={Medical image analysis},
  volume={45},
  pages={121--133},
  year={2018},
  publisher={Elsevier}
}
    
### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

All the three deep models use Caffe to train. 

Our detection model is based on Faster R-CNN model. We use the py-faster-rcnn framework. You need to firstly install the py-faster-rcnn, more details see https://github.com/rbgirshick/py-faster-rcnn.

Our segmentation model is based on a FCN derived from VGG-16.

The deep verificaiton model is based on ResNet-50 architecture.

### Requirements: software

We use a TITAN X GPU with ~12GB memory in our experiments. However, a good GPU with at least 8G of memory suffices.


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

# next , the data folder






