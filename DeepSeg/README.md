We use the FCN framework to train the deep segmentation model. The code of FCN is in https://github.com/shelhamer/fcn.berkeleyvision.org. 

We modify the related files to train a model with the 2012 MITOSIS dataset. The DeepSeg model is then applied on the 2014 MITOSIS data to produce segmentation map.

### Data preparation
We use the precisely annotated 2012 MITOSIS dataset to train the DeepSeg model. We sample patches of 521 ×521 pixels from HPF images, and then mirror and rotate them to augment the training data. Since there are much more negative pixels than positive pixels, we perform more augmentation on patches containing positive pixels than patches only having negative pixels to balance the data.
