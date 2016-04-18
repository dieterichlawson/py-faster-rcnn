#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""


import matplotlib
matplotlib.use('Agg')
import random
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


CLASSES = ('__background__', 'gable')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print "No detection"
        return
    else:
        print str(len(inds))+" gables detected with thresh "+str(thresh)

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    print "Saving plt fig"
    sys.stdout.flush()
    plt.draw()
    plt.savefig('test'+str(random.randint(0,10000))+'.png')

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    print im_file
    im = cv2.imread(im_file)
    #if im is None:
    #  return

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.8
    CONF_THRESH = 0.49
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print "calling vis detect"
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            #'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #prototxt = '/home/will/dev/py-faster-rcnn/models/hover/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
    prototxt = 'models/hover/VGG16/faster_rcnn_end2end/test.prototxt'
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                            #  NETS[args.demo_net][1])

    #caffemodel = '/home/will/dev/py-faster-rcnn/output/faster_rcnn_alt_opt/train/vgg16_rpn_stage1_iter_40000.caffemodel'
    #caffemodel = 'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_end2end_iter_10000.caffemodel'
    #caffemodel = 'output/faster_rcnn_end2end/train/vgg16_places_faster_rcnn_end2end_iter_25000.caffemodel'
    #caffemodel = 'output/faster_rcnn_alt_opt/train/vgg16_rpn_stage2_iter_20000.caffemodel'
    #caffemodel = 'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_end2end_iter_5000.caffemodel'
    caffemodel = '/home/will/dev/py-faster-rcnn/output/faster_rcnn_end2end/train/vgg16_faster_rcnn_end2end_iter_20000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    ims224 =['/home/dlaw/dev/zulu/data/img/256x256/image_24338_order_2294.jpg','/home/dlaw/dev/zulu/data/img/224x224/image_22106_order_2069.jpg', '/home/dlaw/dev/zulu/data/img/224x224/image_22109_order_2069.jpg',  '/home/dlaw/dev/zulu/data/img/224x224/image_22112_order_2070.jpg',  '/home/dlaw/dev/zulu/data/img/224x224/image_22114_order_2070.jpg',
'/home/dlaw/dev/zulu/data/img/224x224/image_22107_order_2069.jpg',  '/home/dlaw/dev/zulu/data/img/224x224/image_22110_order_2069.jpg', '/home/dlaw/dev/zulu/data/img/224x224/image_22113_order_2070.jpg']
    im_names = ['/scratch/deep_learning_data/downloaded_orders/order_26663/image_250970_order_26663.jpg','/scratch/deep_learning_data/markup_images_224/image_223406_order_23392_0.jpg','000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    fimgs = ['/scratch/deep_learning_data/downloaded_orders/order_26644/image_250825_order_26644.jpg',
             '/scratch/deep_learning_data/downloaded_orders/order_24667/image_234271_order_24667.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_26480/image_249480_order_26480.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24155/image_229922_order_24155.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_23563/image_224861_order_23563.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_25803/image_243760_order_25803.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24450/image_232409_order_24450.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_25664/image_242663_order_25664.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24157/image_229937_order_24157.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_27280/image_256122_order_27280.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_26629/image_250693_order_26629.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_27132/image_254897_order_27132.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_25353/image_239981_order_25353.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_28242/image_264159_order_28242.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_25565/image_241801_order_25565.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24191/image_230215_order_24191.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24517/image_232966_order_24517.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24772/image_235218_order_24772.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24129/image_229695_order_24129.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_27055/image_254286_order_27055.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_27315/image_256397_order_27315.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_24401/image_232015_order_24401.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_28731/image_268253_order_28731.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_28685/image_267876_order_28685.jpg',
	      '/scratch/deep_learning_data/downloaded_orders/order_27556/image_258398_order_27556.jpg']
    #im_names = ims224+im_names
    lines = [line.rstrip('\n') for line in open('/home/will/dev/zulu/data_processing/rtest')]
    print len(lines)
    #for im_name in fimgs+ims224:
    for im_name in lines:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'Demo for '.format(im_name)
        print "Demo for "+im_name
        demo(net, str(im_name))

    #plt.show()
