#!/bin/bash

source /mnt/neocortex/scratch/tsechiw/subsystem/init.sh

export PYTHONPATH=/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/python

python detector.py --gpu --input_file=image_cat.txt --crop_mode=selective_search \
--model_def=/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt \
--pretrained_model=/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model \
--images_mean_file=imagenet/ilsvrc_2012_mean.npy \
--output_file=selective_cat.h5

##########################################################################################

import pandas as pd
import numpy as np

df = pd.read_hdf('selective_cat.h5', 'df')
with open('../../../examples/synset_words.txt') as f:
  labels = [' '.join(l.strip().split(' ')[1:]).split(',')[0] for l in f.readlines()]

feats_df = pd.DataFrame(np.vstack(df.feat.values), columns=labels)
feats_df

max_s = feats_df.max(0)
max_s.sort(ascending=False)
print(max_s[:10])

##########################################################################################

GLOG_logtostderr=1 ../examples/train_net.bin lenet_solver.prototxt

##########################################################################################

my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'
my_crop_mode='list'
my_input_file='ss1.csv'
my_output_file='ss1_feat.h5'

from detector import *
extract_feats(
  model_def=my_model_df,
  pretrained_model=my_pretrained_model,
  gpu=True,
  crop_mode='list',
  input_file=my_input_file,
  output_file=my_output_file)

########################################################################################

import detector

detector.extract_feats(model_def="", pretrained_model="", gpu=False, crop_mode="center_only",
          input_file="", output_file="", target_layers=[-1], images_dim=256, 
          images_mean_file="../imagenet/ilsvrc_2012_mean.npy"):

########################################################################################

import detector

my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'

my_tr_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes'
my_va_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSValBoxes'
my_te_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTestBoxes'

my_te_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats'

import sys
reload(detector)
import ipdb,traceback

try:
  detector.mass_extract_feats(my_model_def,my_pretrained_model,True,'list',my_te_input_root_dir,
      my_te_output_root_dir,[16],256,"../imagenet/ilsvrc_2012_mean.npy",30)
except:
  (type, value, tb) = sys.exc_info()
  traceback.print_exc()
  ipdb.post_mortem(tb)