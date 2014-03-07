import os
import sys
import laststage

"""
Some Paths
"""
VOC2007_SSBoxes_Train = '/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes/'
VOC2007_GroundTruth = '/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/'

VOC2007_TrainFeats = '/mnt/neocortex/scratch/bhwang/data/pascal07/TrainFeats/'
VOC2007_ValFeats = '/mnt/neocortex/scratch/bhwang/data/pascal07/ValFeats/'
VOC2007_TestFeats = '/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats/'

VOC2007_RawJPEG = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/JPEGImages/'

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

if __name__ == '__main__':
  tr_gt_fns = [VOC2007_GroundTruth+'VOC2007trainGT.mat']
  tr_feats_dirs = [VOC2007_TrainFeats]
  va_gt_fns = [VOC2007_GroundTruth+'VOC2007valGT.mat']
  va_feats_dirs = [VOC2007_ValFeats]

  mylsc = laststage.LastStageClassifier()
  mylsc.setup_training(tr_gt_fns,tr_feats_dirs,[],[],feats_choice=16,num_classes=1)
  #mylsc.setup_training(tr_gt_fns,tr_feats_dirs,va_gt_fns,va_feats_dirs,feats_choice=16,num_classes=20)

  #for class_i in range(len(mylsc.Xfn_tr)):
  #  assert(len(mylsc.Xfn_tr[0])==len(mylsc.Xwindows_tr[0]))
  #  assert(len(mylsc.Xfn_tr[0])==len(mylsc.y_tr[0]))
  #  assert(len(mylsc.Xfn_tr[0])==len(mylsc.gtBoxes_tr[0]))
  #  for im_idx in range(len(mylsc.Xfn_tr[0])):
  #    assert(mylsc.Xfn_tr[0][im_idx].shape[0] == mylsc.Xwindows_tr[0][im_idx].shape[0])
  #    assert(mylsc.Xfn_tr[0][im_idx].shape[0] == mylsc.y_tr[0][im_idx].shape[0])

  print('pickling!!!')
  import cPickle
  #f = open('mylsc_new_big.cpickle','w')
  #cPickle.dump(mylsc,f)
  #f.close()

  #f = open('mylsc_new_big.cpickle')
  #mylsc = cPickle.load(f)
  #f.close()  

  #f = open('tmpsaves')
  #tr_scores = cPickle.load(f)
  #f.close()
  import pdb; pdb.set_trace()

  mylsc.train_selected_classes([1]+[0]*19)

  #mylsc.generate_files_for_ytz_finetune('all_labels_train.txt',mylsc.Xfn_tr,mylsc.Xwindows_tr,mylsc.y_tr)
  #mylsc.generate_files_for_ytz_finetune('all_labels_val.txt',mylsc.Xfn_va,mylsc.Xwindows_va,mylsc.y_va)
  #mylsc.generate_files_for_ytz_finetune('all_labels_train_1.txt',mylsc.Xfn_tr,mylsc.Xwindows_tr,mylsc.y_tr)
