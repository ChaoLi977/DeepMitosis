import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
weights = '../mitos-fcn32s/fcn32s-heavy-pascal.caffemodel'
#weights = '/home/lc/ali/hed-0723/examples_my/mitos_1081/2012_32s/train1_iter_60000.caffemodel'

# init
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
solver.restore('snapshot/train_iter_220000.solverstate')
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)

solver.step(180000)
#for _ in range(25):
#    solver.step(4000)
#    score.seg_tests(solver, False, val, layer='score')
