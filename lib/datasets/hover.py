# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


from itertools import izip_longest
import datasets
import os
from datasets.imdb import imdb
import datasets.hover
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import pdb
import json
from fast_rcnn.config import cfg

class hover(imdb):
  def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
#         pdb.set_trace()
        print image_set
        self._image_set = image_set
        self._devkit_path = devkit_path
        if image_set == 'train':
          self._data_path = os.path.join(self._devkit_path)
        else:
          self._data_path = os.path.join(self._devkit_path)
#           print 'testing data import'
        #self.annotations_file = '/home/will/dev/py-faster-rcnn/hover/rcnn_tri_box_sz' 
        self.annotations_file = '/home/will/dev/py-faster-rcnn/hover/rcnn_tri_box_test5_gt' 
        self._classes = ('__background__', # always index 0
                         'gable',
                         )
        print 'number of classes: ', self.num_classes
#         pdb.set_trace()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_diff' : False,
                       'use_salt' : True,
                       'rpn_file' : None }
                       #'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
#         print self.image_path_from_index(self._image_index[i])
        return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            #print image_path
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

  def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

  def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        annotations_data = open(self.annotations_file, 'r')
        #self.ann_json_data = json.load(annotations_data)
        gt_roidb = [self._load_hover_annotation(line)
                    for line in annotations_data]
        annotations_data.close()
#         pdb.set_trace()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
  
  def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
	with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

  def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        print filename
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()
# 	pdb.set_trace()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

# 	pdb.set_trace()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

  def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        print cache_file
#         pdb.set_trace()
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

  def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        print IJCV_PATH
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)
  
  def floor_or_ceil(self, val, dim_max):
      if val < 0:
        return 0
      if val >= dim_max:
        return dim_max - 1
      return val

  def parse_line(self,line):
      def grouper(n, iterable, fillvalue=None):
          args = [iter(iterable)] * n
          return izip_longest(fillvalue=fillvalue, *args)

      fields = line.split()
      im_id = fields[0]
      #hw = fields[2]
      #h = float(hw[0])
      #w = float(hw[1])
      #num_gables, num_windows = fields[1:3]
      #num_gables = int(num_gables)
      #num_windows = int(num_windows)
      num_gables= int(fields[1])
      #print fields
      gable_bboxes = [tuple([float(x) for x in bbox]) for bbox in grouper(4,fields[2:])]
      #gable_bboxes = [tuple([float(x) for x in bbox]) for bbox in grouper(4,fields[3:3+4*num_gables])]
      #window_bboxes = [tuple([float(x) for x in bbox]) for bbox in grouper(4,fields[3+4*num_gables:])]
      #return im_id, gable_bboxes, window_bboxes
      return im_id, gable_bboxes

  def _load_hover_annotation(self, index):
      """
      Load image and bounding boxes info.
      """
      # Load object bounding boxes
      
      im_id, gable_bboxes = self.parse_line(index)
      num_objs = len(gable_bboxes)
      boxes = np.array(gable_bboxes, dtype=np.float32)
      gt_classes = np.empty(num_objs, dtype=np.int32)
      gt_classes.fill(1) # all bboxes are gables currently
      overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
      overlaps[:,1] = 1.0 # all boxes are gables...
      overlaps = scipy.sparse.csr_matrix(overlaps)
      seg_areas = np.zeros((num_objs), dtype=np.float32)
      for ix, bbox in enumerate(boxes):
        print bbox
	x1 = bbox[0] - 1
	y1 = bbox[1] - 1
	x2 = bbox[2] - 1
	y2 = bbox[3] - 1
	seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        print seg_areas[ix]

      return {'boxes' : boxes,
              'gt_classes': gt_classes,
              'gt_overlaps' : overlaps,
              'seg_areas': seg_areas,
              'flipped' : False}


  def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

  def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _do_detection_eval(self, res_file, output_dir):
    # evaluate recall
    print "detection eval not implemented yet"

  # create list of dicts of results for one class
  # boxes is #images x # dets x 5
  def _hover_results_one_class(self, boxes, class_id):
    results = []
    from IPython import embed
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []: 
        continue
      results.extend(
        [{'image_id': index,
          'category_id': class_id,
          'bbox': dets[k,0:4].tolist(),
          'score': dets[k,-1]} for k in xrange(dets.shape[0])])
    return results

  # write out json file of results, with format
  # [{image_id: 42, category_id: 18, bbox: [2.3, 41.29, 348.26, 243.78], score: 0.236} , ... ]
  def _write_hover_results_file(self, all_boxes,res_file):
    results = []
    # for each class
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__': continue # if background, skip
      print 'Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1)
      class_ind = self._class_to_ind[cls]
      # generate json for class
      results.extend(self._hover_results_one_class(all_boxes[cls_ind],class_ind))
    print 'Writing results json to {}'.format(res_file)
    for item in results:
      assert item['bbox'][0] <= item['bbox'][2]
      assert item['bbox'][1] <= item['bbox'][3]
      assert item['score'] <= 1.0
      assert item['score'] >= 0.0
    with open(res_file, 'w') as f:
      json.dump(results, f)

  def evaluate_detections(self, all_boxes, output_dir):
    # create name of results file
    res_file = os.path.join(output_dir, ('detections_' + 
                                     self._image_set + 
                                    '_results.json'))
    # write out results to res_file
    self._write_hover_results_file(all_boxes, res_file)
    # evaluate results, print metrics
    self.evaluate_recall(candidate_boxes=all_boxes)

  def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.hover('train', '')
    res = d.roidb
    from IPython import embed; embed()
