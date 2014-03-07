
my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'

my_tr_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes'
my_va_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSValBoxes'
my_te_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTestBoxes'

my_tr_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TrainFeats'
my_va_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/ValFeats'
my_te_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats'

import sys

if len(sys.argv) != 5:
  print("Help command arguments:")
  print("python run_mass_ext.py <target layers> <gpu to use> <number of parallel jobs> <tr|va|te>")
  print("python run_mass_ext.py 15.16 2.0.2.1.0 5 tr")
  print("python run_mass_ext.py 16 0.1.2 3 va")
  print("<gpu to use> := gpu_id_of_1st_job.gpu_id_of_2nd_job.(etc.)")
  sys.exit()

if sys.argv[4] == 'tr':
	my_input_dir = my_tr_input_root_dir
	my_output_dir = my_tr_output_root_dir
	print("mass extract features on training dataset")
elif sys.argv[4] == 'va':
	my_input_dir = my_va_input_root_dir
	my_output_dir = my_va_output_root_dir
	print("mass extract features on validation dataset")
elif sys.argv[4] == 'te':
	my_input_dir = my_te_input_root_dir
	my_output_dir = my_te_output_root_dir
	print("mass extract features on testing dataset")
else:
	print "wrong input for choice of dataset tr|va|te"

import multiprocessing
import detector

processes = []
target_layers = [int(i) for i in sys.argv[1].split('.')]
gpu_to_use = [int(i) for i in sys.argv[2].split('.')]
n = int(sys.argv[3]) # num processes
for i in range(n):
  print(target_layers)
  print(gpu_to_use[i])
  print((i,n))

  my_args = [my_model_def,my_pretrained_model,True,'list',my_input_dir,
      my_output_dir,target_layers,256,"../imagenet/ilsvrc_2012_mean.npy",2500,gpu_to_use[i],(i,n)]
	
  process = multiprocessing.Process(target=detector.mass_extract_feats, args=my_args)
  process.start()
  processes.append(process)

print "!+!+!+! started all jobs"

for p in processes:
  p.join()

print "!+!+!+! finished all jobs"
#detector.mass_extract_feats(my_model_def,my_pretrained_model,True,'list',my_tr_input_root_dir,
#      my_tr_output_root_dir,[14],256,"../imagenet/ilsvrc_2012_mean.npy",2500,my_gpu_id,)
