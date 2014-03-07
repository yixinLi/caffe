import os
import sys
import laststage
import cPickle

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

def unit_test_basic_data_length_matches(mylsc):
  for class_i in range(len(mylsc.Xfn_tr)):
    assert(len(mylsc.Xfn_tr[0])==len(mylsc.Xwindows_tr[0]))
    assert(len(mylsc.Xfn_tr[0])==len(mylsc.y_tr[0]))
    assert(len(mylsc.Xfn_tr[0])==len(mylsc.gtBoxes_tr[0]))
    for im_idx in range(len(mylsc.Xfn_tr[0])):
      assert(mylsc.Xfn_tr[0][im_idx].shape[0] == mylsc.Xwindows_tr[0][im_idx].shape[0])
      assert(mylsc.Xfn_tr[0][im_idx].shape[0] == mylsc.y_tr[0][im_idx].shape[0])

if __name__ == '__main__':
  tr_gt_fns = [VOC2007_GroundTruth+'VOC2007trainGT.mat']
  tr_feats_dirs = [VOC2007_TrainFeats]
  va_gt_fns = [VOC2007_GroundTruth+'VOC2007valGT.mat']
  va_feats_dirs = [VOC2007_ValFeats]

  mylsc = laststage.LastStageClassifier()
  #mylsc.setup_training(tr_gt_fns,tr_feats_dirs,[],[],feats_choice=16,num_classes=1)
  print("run_laststage.py <number_of_classes> <class_choice> [<pickle filename>]")
  print("<class_choice> := 'all' or 2.4.15.17.10")
  if int(sys.argv[1]) > 20 or int(sys.argv[1]) < 1:
    print('number of classes too big or too small')
    sys.exit()
  else:
    my_num_classes = int(sys.argv[1])
  if sys.argv[2] == 'all':
    my_class_choice = [1]*my_num_classes
  else:
    my_class_choice = [0]*my_num_classes
    for class_i in map(int,sys.argv[2].split('.')):
      my_class_choice[class_i] = 1
  if len(sys.argv) >= 4:
    pickle_fn = sys.argv[3]
  else:
    pickle_fn = 'mylsc_final.cpickle'

  if not os.path.isfile(pickle_fn):
    print('setup_training and then pickling the loaded data structure')
    mylsc.setup_training(tr_gt_fns,tr_feats_dirs,va_gt_fns,va_feats_dirs,feats_choice=16,num_classes=my_num_classes)
    
    f = open(pickle_fn,'w')
    cPickle.dump(mylsc,f)
  else:
    print('loading previously pickled data structure')
    f = open(pickle_fn)
    mylsc = cPickle.load(f)
  f.close()

  print("finish setup / loading!")

  #f = open('tmpsaves')
  #tr_scores = cPickle.load(f)
  #f.close()
  #import pdb; pdb.set_trace()

  mylsc.train_selected_classes(my_class_choice)

  #mylsc.generate_files_for_ytz_finetune('all_labels_train.txt',mylsc.Xfn_tr,mylsc.Xwindows_tr,mylsc.y_tr)
  #mylsc.generate_files_for_ytz_finetune('all_labels_val.txt',mylsc.Xfn_va,mylsc.Xwindows_va,mylsc.y_va)
  #mylsc.generate_files_for_ytz_finetune('all_labels_train_1.txt',mylsc.Xfn_tr,mylsc.Xwindows_tr,mylsc.y_tr)
