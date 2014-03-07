# debugging the features extraction of detector.py script

import detector

my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'

my_tr_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes'
my_va_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSValBoxes'
my_te_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTestBoxes'

my_te_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats'

detector.mass_extract_feats(my_model_def,my_pretrained_model,True,'list',my_te_input_root_dir,
      my_te_output_root_dir,[16],256,"../imagenet/ilsvrc_2012_mean.npy",2500)
