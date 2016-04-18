# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

# import datasets.pascal_voc
from datasets.hover import hover
import numpy as np
import pdb

# inria_devkit_path = '/Volumes/Data/hover-temp-data/zulu_data/downloaded_orders'
# hover_devkit_path = '/Volumes/Data/hover-temp-data/zulu_data/downloaded_orders/new_orders'
hover_devkit_path = '/scratch/deep_learning_data/fast-rcnn-data'

# def _selective_search_IJCV_top_k(split, year, top_k):
#     """Return an imdb that uses the top k proposals from the selective search
#     IJCV code.
#     """
#     imdb = datasets.pascal_voc(split, year)
#     imdb.roidb_handler = imdb.selective_search_IJCV_roidb
#     imdb.config['top_k'] = top_k
#     return imdb
# 
# # Set up voc_<year>_<split> using selective search "fast" mode
# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'voc_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year:
#                 datasets.pascal_voc(split, year))
# 
# # Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# # but only returning the first k boxes
# for top_k in np.arange(1000, 11000, 1000):
#     for year in ['2007', '2012']:
#         for split in ['train', 'val', 'trainval', 'test']:
#             name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
#             __sets[name] = (lambda split=split, year=year, top_k=top_k:
#                     _selective_search_IJCV_top_k(split, year, top_k))

# Set up inria_<split> using selective search "fast" mode
for split in ['train', 'test']:
    name = '{}_{}'.format('hover', split)
#     pdb.set_trace()
    __sets[name] = (lambda split=split: hover(split, hover_devkit_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    print name
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
