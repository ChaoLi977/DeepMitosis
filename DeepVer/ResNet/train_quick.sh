#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  -solver examples/ResNet/solver.prototxt -weights examples/ResNet/ResNet-50-model.caffemodel -gpu 0
