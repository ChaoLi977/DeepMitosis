# read the detections.pkl and convert it to txt file
import os
import pickle, cPickle
import numpy as np
import math

def save_detection(filename,det_m):  
    """save the detections in txt file, 
    as the standard format: all detections in a file,
    imagename score x1 y1 x2 y2"""
    
    fid_all= open('/home/lc/ali/py-faster-rcnn/my_tools/detection/det.txt','w')
    num_img=len(det_m[1])
    det_res0 = [] 
    det_res1 = []
    
    for i in range(num_img):
        bbs_img = det_m[1][i]
        num_det = len(bbs_img)
        for n in range(num_det):            
            b=bbs_img[n]
            fid_all.write('{} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} \n'.format( \
                    filename[i],b[4],b[0],b[1],b[2],b[3]))
        det_res0.append([])          
        det_res1.append(bbs_img)
        det_res=[det_res0,det_res1]
    fid_all.close()
    return det_res

def parse_anno(filename):
    objects = []
    
    f=open(filename,'r')
    line = f.readline()
    while line:
        line=line.split(' ')
        obj_struct = {}
        name=line[0]
        cen_x = line[1]
        cen_y = line[2]
        obj_struct['centroid']=[cen_x,cen_y]
        obj_struct['imagename'] = [name]
        obj_struct['det'] = False
        objects.append(obj_struct)    
        line = f.readline()
        
    f.close()
    imagesetfile = '/home/lc/ali/py-faster-rcnn/my_tools/detection/test_git.txt'
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        class_recs[imagename]= [obj for obj in objects if obj['imagename'][0]==imagename]
        #print('imagename={}'.format(imagename))
    return class_recs     # class_recs[imagename] is the ground truth of an image
    
    
if __name__=='__main__':
    dithresh = 20
    filename=['A00_v2/A00_00','A00_v2/A00_08','A01_v2/A01_04','A01_v2/A01_06', \
              'A01_v2/A01_09','A02_v2/A02_00','A02_v2/A02_01','A02_v2/A02_03', \
              'A02_v2/A02_07','A03_v2/A03_00','A03_v2/A03_01','A03_v2/A03_04', \
              'A04_v2/A04_03','A04_v2/A04_07','A04_v2/A04_09']
              
    dir_det='/home/lc/ali/py-faster-rcnn/my_tools/original_det/'
    dets=os.path.join(dir_det,'detections.pkl')
    pkl_det = open(dets, 'rb')
    det_m = cPickle.load(pkl_det)
    print('len of detection is {}'.format(len(det_m[1])))
    det_res=save_detection(filename, det_m)  


    # read the gt
    test_anno_file = '/home/lc/ali/py-faster-rcnn/my_tools/anno_test_101_git.txt'
    class_recs = parse_anno(test_anno_file)    
    
    #read dets
    # load the converted detections
    with open('/home/lc/ali/py-faster-rcnn/my_tools/detection/det.txt','r') as f:
        lines = f.readlines()
        
    splitlines = [x.strip().split() for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    #sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB=BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    # go down dets and mark TPs and FPs
    nd = len(image_ids)   #number of detections
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):   #for every detected bbs
        R = class_recs[image_ids[d]]
        
        BBGT =[x['centroid'] for x in R]
        BBGT = np.transpose(np.array(BBGT).astype(float))

        distmin = np.inf
        bb = BB[d, :].astype(float)
        bb_x = (bb[0] + bb[2])/2
        bb_y = (bb[1] + bb[3])/2
        
        if BBGT.size > 0:
            dist = np.sqrt(np.square(BBGT[0]-bb_x) +np.square(BBGT[1]-bb_y))
           # print('dist {0}'.format(dist))
            distmin = np.min(dist)
            jmin = np.argmin(dist)
            
        if distmin < dithresh:
            if not R[jmin]['det']:
                tp[d] = 1.
                R[jmin]['det'] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
     # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    npos = 101 # the number of mitosis in ground truth
    rec = tp / float(npos)
    prec = tp /np.maximum(tp + fp, np.finfo(np.float64).eps)
    F= 2*rec*prec/(rec+prec)
    F = [x for x in F if not math.isnan(x)]
    F_max=np.max(F)
    
    print('fp {}'.format(fp))
    print('tp {}'.format(tp))
    print('npos {}'.format(npos))
    max_index = F.index(F_max)
    score_thresh = sorted_scores[max_index]
    prec_m=prec[max_index]
    rec_m = rec[max_index]
    print('the max F is {}'.format(F_max))   
    print('the prec is {}, and the rec is {}'.format(prec_m, rec_m))
