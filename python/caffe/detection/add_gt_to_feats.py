import detector

my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'

my_tr_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes'
my_va_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSValBoxes'
my_te_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTestBoxes'

my_tr_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TrainFeats'
my_va_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/ValFeats'
my_te_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats'

import multiprocessing

if __name__=='__main__':
  detector.append_more_windows_to_feats(my_model_def,my_pretrained_model,True,'list',my_va_output_root_dir,[15],256,
    "../imagenet/ilsvrc_2012_mean.npy",
     "/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007valGT.mat",0)

  processes = []

  args_16tr = [my_model_def,my_pretrained_model,True,'list',my_tr_output_root_dir,[16],256,
    "../imagenet/ilsvrc_2012_mean.npy",
     "/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007trainGT.mat",0]
  args_15tr = [my_model_def,my_pretrained_model,True,'list',my_tr_output_root_dir,[15],256,
    "../imagenet/ilsvrc_2012_mean.npy",
     "/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007trainGT.mat",0] 
  
  args_16va = [my_model_def,my_pretrained_model,True,'list',my_va_output_root_dir,[15],256,
    "../imagenet/ilsvrc_2012_mean.npy",
     "/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007valGT.mat",0]
  args_15va = [my_model_def,my_pretrained_model,True,'list',my_va_output_root_dir,[15],256,
    "../imagenet/ilsvrc_2012_mean.npy",
     "/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007valGT.mat",0] 
  
  process = multiprocessing.Process(target=detector.append_more_windows_to_feats, args=args_16va)
  #process.start()
  #processes.append(process) 
  
  import time
  time.sleep(30)

  process = multiprocessing.Process(target=detector.append_more_windows_to_feats, args=args_15va)
  #process.start()
  #processes.append(process)

  for p in processes:
    p.join()

#  detector.append_more_windows_to_feats(model_def=my_model_def, \
#    pretrained_model=my_pretrained_model, \
#    gpu=True, crop_mode="list",\
#    feats_root_dir=my_tr_output_root_dir, \
#    target_layer=16,
#    images_dim=256, images_mean_file="../imagenet/ilsvrc_2012_mean.npy",\
#    gt_input_fn="/mnt/neocortex/scratch/bhwang/data/pascal07/GroundTruth/VOC2007trainGT.mat",\
#    gpu_device_id=0)

