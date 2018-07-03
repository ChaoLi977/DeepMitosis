# DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks
By Chao Li, [Xinggang Wang](http://www.xinggangw.info/), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu) and [Longin Jan Latecki](https://cis.temple.edu/~latecki/)

Codes for our MIA (Medical Image Analysis) paper "DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks". Please see the [paper](https://www.sciencedirect.com/science/article/pii/S1361841517301834) for more details.

### Citing DeepMitosis

If you find DeepMitosis useful in your research, please consider citing:

    @article{li2018deepmitosis,
        title = {DeepMitosis: Mitosis detection via deep detection, verification and segmentation networks},
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
3. [Basic installation](#installation)
4. [Data preparation](#data-preparation)

### Requirements: software

All the three deep models use Caffe framework to train. 

Our detection model is based on Faster R-CNN model. We use the py-faster-rcnn. You need to firstly install the py-faster-rcnn, more details see https://github.com/rbgirshick/py-faster-rcnn.

Our segmentation model is based on a FCN model derived from VGG-16.

The deep verificaiton model is based on ResNet-50 architecture.

### Requirements: hardware

We use a TITAN X GPU with ~12GB memory in our experiments. However, a good GPU with at least 8G of memory suffices.

### Installation

1.Install the Caffe framework.

2.Install the py-faster-rcnn and train the DeepDet model on 2012 MITOSIS dataset.

3.Install the FCN (fully convolutionnal networks) to train the DeepSeg model on 2012 MITOSIS dataset, and then deploy DeepSeg on 2014 MITOSIS dataset.

4.Train a DeepVer model using the detection results produced by DeepDet on 2014 MITOSIS dataset.

### Data preparation

Download the 2012 MITOSIS dataset and 2014 MITOSIS dataset. 

For DeepDet model, the data are arranged as the VOC data in py-faster-rcnn. We transfer the mitosis' annotation to bounding box format.

For DeepSeg model, we convert the annotations to mask images.

We perform data augmentations and image crop to produce more training samples. The data augmentation includes image rotation, mirror. Please see our paper for more details. Noted that in detection model training, we remove the image patches that contain small cross- boundary mitotic cells from the training data.









