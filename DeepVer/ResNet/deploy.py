import sys
import caffe
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array

import cv2
from PIL import Image
import matplotlib
import numpy as np
import lmdb

caffe_root = '../'
caffe.set_mode_gpu()
caffe.set_device(0)
MODEL_FILE = '/caffe_root_path/examples/ResNet/deploy_mitos.prototxt'
PRETRAINED = 'caffe_root_path/examples/ResNet/final.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
MEAN_FILE= 'caffe_root_path/data/mitos/mean/mitos_mean.npy'  # you need to produce the mean file in npy format
mu = np.load(MEAN_FILE)
mu = mu.mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
mitos_root='caffe_root_path/data/mitos/'
testList = 'test_image_patch.txt'
with open(testList) as fid:
    lines = fid.readlines()
fid.close()
    
for line in lines:
    line=line.strip()
    C=line.split(' ')
    addr=C[0]
    label=int(C[1])
    image = caffe.io.load_image(mitos_root + 'test/' + addr )
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    out = net.forward()
    output_prob = out['prob'][0] 
    print 'prediction is:', output_prob
    predicted_label = output_prob.argmax(axis=0)
