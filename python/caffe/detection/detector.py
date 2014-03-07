"""
Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows
  Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
  Rich feature hierarchies for accurate object detection and semantic
  segmentation.
  http://arxiv.org/abs/1311.2524

The selective_search_ijcv_with_python code is available at
  https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- refactor into class (without globals)
- update demo notebook with new options
"""
import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe
import pandas.io.pytables as PyTables

NET = None

IMAGE_DIM = None
CROPPED_DIM = None
IMAGE_CENTER = None

IMAGE_MEAN = None
CROPPED_IMAGE_MEAN = None

BATCH_SIZE = None
NUM_OUTPUT = None

CROP_MODES = ['list', 'center_only', 'corners', 'selective_search']


def load_image(filename):
  """
  Input:
    filename: string

  Output:
    image: an image of size (H x W x 3) of type uint8.
  """
  img = skimage.io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def format_image(image, window=None, cropped_size=False):
  """
  Input:
    image: (H x W x 3) ndarray
    window: (4) ndarray
      (ymin, xmin, ymax, xmax) coordinates, 0-indexed
    cropped_size: bool
      Whether to output cropped size image or full size image.

  Output:
    image: (3 x H x W) ndarray
      Resized to either IMAGE_DIM or CROPPED_DIM.
    dims: (H, W) of the original image
  """
  dims = image.shape[:2]

  # Crop a subimage if window is provided.
  if window is not None:
    image = image[window[0]:window[2], window[1]:window[3]]

  # Resize to input size, subtract mean, convert to BGR
  image = image[:, :, ::-1]
  if cropped_size:
    image = skimage.transform.resize(image, (CROPPED_DIM, CROPPED_DIM)) * 255
    image -= CROPPED_IMAGE_MEAN
  else:
    image = skimage.transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255
    image -= IMAGE_MEAN

  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image, dims


def _image_coordinates(dims, window):
  """
  Calculate the original image coordinates of a
  window in the canonical (IMAGE_DIM x IMAGE_DIM) coordinates

  Input:
    dims: (H, W) of the original image
    window: (ymin, xmin, ymax, xmax) in the (IMAGE_DIM x IMAGE_DIM) frame

  Output:
    image_window: (ymin, xmin, ymax, xmax) in the original image frame
  """
  h, w = dims
  h_scale, w_scale = h / IMAGE_DIM, w / IMAGE_DIM
  image_window = window * np.array((1. / h_scale, 1. / w_scale,
                                   h_scale, w_scale))
  return image_window.round().astype(int)


def _assemble_images_list(input_df):
  """
  For each image, collect the crops for the given windows.

  Input:
    input_df: pandas.DataFrame
      with 'filename', 'ymin', 'xmin', 'ymax', 'xmax' columns

  Output:
    images_df: pandas.DataFrame
      with 'image', 'window', 'filename' columns
  """
  # unpack sequence of (image filename, windows)
  coords = ['ymin', 'xmin', 'ymax', 'xmax']
  image_windows = (
    (ix, input_df.iloc[np.where(input_df.index == ix)][coords].values)
    for ix in input_df.index.unique()
  )

  # extract windows
  data = []
  for image_fname, windows in image_windows:
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_center_only(image_fnames):
  """
  For each image, square the image and crop its center.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  crop_start, crop_end = IMAGE_CENTER, IMAGE_CENTER + CROPPED_DIM
  crop_window = np.array((crop_start, crop_start, crop_end, crop_end))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    data.append({
      'image': image[np.newaxis, :,
                     crop_start:crop_end,
                     crop_start:crop_end],
      'window': _image_coordinates(dims, crop_window),
      'filename': image_fname
    })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_corners(image_fnames):
  """
  For each image, square the image and crop its center, four corners,
  and mirrored version of the above.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  # make crops
  indices = [0, IMAGE_DIM - CROPPED_DIM]
  crops = np.empty((5, 4), dtype=int)
  curr = 0
  for i in indices:
    for j in indices:
      crops[curr] = (i, j, i + CROPPED_DIM, j + CROPPED_DIM)
      curr += 1
  crops[4] = (IMAGE_CENTER, IMAGE_CENTER,
              IMAGE_CENTER + CROPPED_DIM, IMAGE_CENTER + CROPPED_DIM)
  all_crops = np.tile(crops, (2, 1))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    image_crops = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for crop in crops:
      image_crops[curr] = image[:, crop[0]:crop[2], crop[1]:crop[3]]
      curr += 1
    image_crops[5:] = image_crops[:5, :, :, ::-1]  # flip for mirrors
    for i in range(len(all_crops)):
      data.append({
        'image': image_crops[i][np.newaxis, :],
        'window': _image_coordinates(dims, all_crops[i]),
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_selective_search(image_fnames):
  """
  Run Selective Search window proposals on all images, then for each
  image-window pair, extract a square crop.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  windows_list = selective_search.get_windows(image_fnames)

  data = []
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def assemble_batches(inputs, crop_mode='center_only'):
  """
  Assemble DataFrame of image crops for feature computation.

  Input:
    inputs: list of filenames (center_only, corners, and selective_search mode)
      OR input DataFrame (list mode)
    mode: string
      'list': take the image windows from the input as-is
      'center_only': take the CROPPED_DIM middle of the image windows
      'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
        the image windows, as well as their flipped versions: a total of 10.
      'selective_search': run Selective Search region proposal on the
        image windows, and take each enclosing subwindow.

  Output:
    df_batches: list of DataFrames, each one of BATCH_SIZE rows.
      Each row has 'image', 'filename', and 'window' info.
      Column 'image' contains (X x 3 x CROPPED_DIM x CROPPED_IM) ndarrays.
      Column 'filename' contains source filenames.
      Column 'window' contains [ymin, xmin, ymax, xmax] ndarrays.
      If 'filename' is None, then the row is just for padding.

  Note: for increased efficiency, increase the batch size (to the limit of gpu
  memory) to avoid the communication cost
  """
  if crop_mode == 'list':
    images_df = _assemble_images_list(inputs)

  elif crop_mode == 'center_only':
    images_df = _assemble_images_center_only(inputs)

  elif crop_mode == 'corners':
    images_df = _assemble_images_corners(inputs)

  elif crop_mode == 'selective_search':
    images_df = _assemble_images_selective_search(inputs)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Make sure the DataFrame has a multiple of BATCH_SIZE rows:
  # just fill the extra rows with NaN filenames and all-zero images.
  N = images_df.shape[0]
  remainder = N % BATCH_SIZE
  if remainder > 0:
    zero_image = np.zeros_like(images_df['image'].iloc[0])
    zero_window = np.zeros((1, 4), dtype=int)
    remainder_df = pd.DataFrame([{
      'filename': None,
      'image': zero_image,
      'window': zero_window
    }] * (BATCH_SIZE - remainder))
    images_df = images_df.append(remainder_df)
    N = images_df.shape[0]

  # Split into batches of BATCH_SIZE.
  ind = np.arange(N) / BATCH_SIZE
  df_batches = [images_df[ind == i] for i in range(N / BATCH_SIZE)]
  return df_batches

def compute_feats(images_df, target_layers=[-1]):
  input_blobs = [np.ascontiguousarray(
    np.concatenate(images_df['image'].values), dtype='float32')]

  if len(target_layers)==0 or target_layers[0]==-1:
    output_blobs = [np.empty((BATCH_SIZE, NUM_OUTPUT, 1, 1), dtype=np.float32)]

    NET.Forward(input_blobs, output_blobs)
    feats = [output_blobs[0][i].flatten() for i in range(len(output_blobs[0]))]
  else:
    all_blobs = []
    for blob in NET.blobs():
      all_blobs.append(np.empty((BATCH_SIZE, blob.channels, blob.height, blob.width), dtype=np.float32))

    NET.ForwardAndExtractFeats(input_blobs, all_blobs)
    feats = []
    for i in range(BATCH_SIZE):
      feats.append( np.hstack([all_blobs[l_idx][i].flatten() for l_idx in target_layers]) )

  # Add the features and delete the images.
  del images_df['image']
  images_df['feat'] = feats
  return images_df

def config(model_def, pretrained_model, gpu, image_dim, image_mean_file, gpu_device_id):
  global IMAGE_DIM, CROPPED_DIM, IMAGE_CENTER, IMAGE_MEAN, CROPPED_IMAGE_MEAN
  global NET, BATCH_SIZE, NUM_OUTPUT

  # Initialize network by loading model definition and weights.
  t = time.time()
  print("Loading Caffe model.")
  NET = caffe.CaffeNet(model_def, pretrained_model)
  NET.set_phase_test()
  if gpu:
    NET.set_mode_gpu()
    NET.set_device(gpu_device_id)
  print("Caffe model loaded in {:.3f} s".format(time.time() - t))

  # Configure for input/output data
  IMAGE_DIM = image_dim
  CROPPED_DIM = NET.blobs()[0].width
  IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

    # Load the data set mean file
  IMAGE_MEAN = np.load(image_mean_file)

  CROPPED_IMAGE_MEAN = IMAGE_MEAN[IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  :]
  BATCH_SIZE = NET.blobs()[0].num  # network batch size
  NUM_OUTPUT = NET.blobs()[-1].channels  # number of output classes

def extract_feats(inputs, crop_mode="list", target_layers=[-1]):
  # Assemble into batches
  t = time.time()
  print('Assembling batches...')
  image_batches = assemble_batches(inputs, crop_mode)
  print('{} batches assembled in {:.3f} s'.format(len(image_batches),
                                                  time.time() - t))

  # Process the batches.
  t = time.time()
  print 'Processing {} files in {} batches'.format(len(inputs),
                                                   len(image_batches))
  dfs_feats = []
  for i in range(len(image_batches)):
    if i % 10 == 0:
      print('Batch {}/{}, elapsed time: {:.3f} s'.format(i,
                                                         len(image_batches),
                                                         time.time() - t))
    # NOTES: target_layers for alexnet: [14,15,16] -> [pool5, fc6, fc7]
    dfs_feats.append(compute_feats(image_batches[i],target_layers))

  # Concatenate, droppping the padding rows
  df = pd.concat(dfs_feats).dropna(subset=['filename'])
  print("Processing complete after {:.3f} s.".format(time.time() - t))

  # Label coordinates
  coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
  df[coord_cols] = pd.DataFrame(data=np.vstack(df['window']),
                                index=df.index,
                                columns=coord_cols)
  del(df['window'])

  df[range(df['feat'][0].size)] = pd.DataFrame(data=np.vstack(df['feat']))
  del(df['feat'])

  return df

import scipy
import glob

def append_more_windows_to_feats(model_def="", pretrained_model="", gpu=True, crop_mode="list",
           feats_root_dir="", target_layers=[16], images_dim=256, 
           images_mean_file="../imagenet/ilsvrc_2012_mean.npy", gt_input_fn="",
           gpu_device_id=3):
  # Configure network, input, output
  config(model_def, pretrained_model, gpu, images_dim, images_mean_file, gpu_device_id)

  batch_inputs = pd.DataFrame()

  all_feats_fns = os.listdir(feats_root_dir)
  all_feats_fns.sort()

  gtmat = scipy.io.loadmat(gt_input_fn)
  gtbox_by_im = [None]*gtmat['setIms'].shape[0]
  for im_idx in range(len(gtbox_by_im)):
    gtbox_by_im[im_idx] = []

  for class_i in range(gtmat['gtIms'].shape[0]):
    for i in range(gtmat['gtIms'][class_i,0].shape[0]):
      im_idx = gtmat['gtImIds'][class_i,0][i,0]-1
      gtbox_by_im[im_idx].append(gtmat['gtBoxes'][class_i,0][i,:])
  

  jpeg_dir = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/JPEGImages/'

  inputs = []
  for im_idx in range(len(gtbox_by_im)):
    jpeg_fn = os.path.join(jpeg_dir,str(gtmat['setIms'][im_idx,0][0])+'.jpg')
    for gtbox in gtbox_by_im[im_idx]:
      inputs.append({'filename': jpeg_fn,
        'ymin': gtbox[0],
        'xmin': gtbox[1],
        'ymax': gtbox[2],
        'xmax': gtbox[3]
        })

  inputs = pd.DataFrame(inputs)
  inputs.set_index('filename', inplace=True)
  
  gtdf = extract_feats(inputs, "list", target_layers)

  for im_idx in range(len(gtbox_by_im)):
    jpeg_fn = os.path.join(jpeg_dir,str(gtmat['setIms'][im_idx,0][0])+'.jpg')
    
    feats_fn = os.path.join(feats_root_dir,str(gtmat['setIms'][im_idx,0][0])+'Feat'+str(target_layers[0])+'Test2500.h5')
    feats = pd.read_hdf(feats_fn,'df')
    feats_gt = gtdf.ix[gtdf['filename']==jpeg_fn,:]
    feats = pd.DataFrame.append(feats_gt,feats)
    feats.index = pd.Int64Index(range(feats.shape[0]))
    feats_new_fn = os.path.join(feats_root_dir,str(gtmat['setIms'][im_idx,0][0])+'Feat'+str(target_layers[0])+'Val2500.h5')
    feats.to_hdf(feats_new_fn,'df',mode='w')

def mass_extract_feats(model_def="", pretrained_model="", gpu=True, crop_mode="list",
           input_root_dir="", output_root_dir="", target_layers=[-1], images_dim=256, 
           images_mean_file="../imagenet/ilsvrc_2012_mean.npy", num_win_to_use_per_file=2000,
           gpu_device_id=3,para_config=(0,1)):
  # Configure network, input, output
  config(model_def, pretrained_model, gpu, images_dim, images_mean_file, gpu_device_id)

  batch_inputs = pd.DataFrame()

  all_input_fns = os.listdir(input_root_dir)
  all_input_fns.sort()

  # para_config := (block num, total number of blocks)
  if type(para_config)==type((1,)) and len(para_config)==2 and para_config[0]<para_config[1]:
    num_fs = len(all_input_fns)/para_config[1] + 1 # NOTE: should be still integer
    print("mass_extract_feats{}: Processing {} number of files".format(para_config[0],num_fs))
    list_of_fs_to_proc = map(None, *(iter(all_input_fns),) * num_fs)
    list_of_fs_to_proc[-1] = tuple([ i for i in list_of_fs_to_proc[-1] if i != None ])
    all_input_fns = list_of_fs_to_proc[para_config[0]]
    print("mass_extract_feats{}: Processing from {} to {}".format(para_config[0],all_input_fns[0],all_input_fns[-1]))

  for fn in all_input_fns:
    # only process the right input files
    if not(os.path.isfile(os.path.join(input_root_dir,fn)) and fn.lower().endswith('.csv')):
      continue

    ##
    import glob
    if len(glob.glob(os.path.join(output_root_dir,fn[2:-4]+'*'+str(target_layers[0])+'*.h5')))>0:
      print("skipping "+fn+". already handled.")
      continue
    ##

    input_fn = os.path.join(input_root_dir,fn)
    # Load input
    # .txt = list of filenames
    # .csv = dataframe that must include a header
    #        with column names filename, ymin, xmin, ymax, xmax
    if input_fn.lower().endswith('txt'):
      with open(input_fn) as f:
        inputs = [_.strip() for _ in f.readlines()]
    elif input_fn.lower().endswith('csv'):
      inputs = pd.read_csv(input_fn, sep=',', dtype={'filename': str})
      inputs.set_index('filename', inplace=True)
    else:
      raise Exception("Uknown input file type: not in txt or csv")

    inputs = inputs[:num_win_to_use_per_file]
    # batch up inputs
    #batch_inputs = batch_inputs.append(inputs)

    df = extract_feats(inputs, "list", target_layers)

    #import ipdb; ipdb.set_trace()
    img_name = df['filename'][0][-10:-4]

    target_str=''
    if len(target_layers)==1 and target_layers[0]==-1:
      target_str='Last'
    else:
      for i in target_layers:
        target_str = target_str + '.' + str(i)
        target_str = target_str[1:]

    output_fn = os.path.join(output_root_dir,fn[2:-4]+'Feat{}_{}.h5'.format(
      target_str,num_win_to_use_per_file))
    print ("Handling "+fn+" of ")

    # Write out the results
    df.to_hdf(output_fn,'df',mode='w')

#from detector import *
if __name__ == "__main__":
  my_model_def='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/imagenet_deploy.prototxt'
  my_pretrained_model='/mnt/neocortex/scratch/bhwang/sharedhome/ytz_caffe/cafferefnet/caffe_reference_imagenet_model'
  
  my_tr_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTrainBoxes'
  my_va_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSValBoxes'
  my_te_input_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/SSTestBoxes'

  my_te_output_root_dir='/mnt/neocortex/scratch/bhwang/data/pascal07/TestFeats'

  mass_extract_feats(my_model_def,my_pretrained_model,True,'list',my_te_input_root_dir,
        my_te_output_root_dir,[16],256,"../imagenet/ilsvrc_2012_mean.npy",30)

#if __name__ == "__main__":
#  # Parse cmdline options
#  gflags.DEFINE_string(
#    "model_def", "", "Model definition file.")
#  gflags.DEFINE_string(
#    "pretrained_model", "", "Pretrained model weights file.")
#  gflags.DEFINE_boolean(
#    "gpu", False, "Switch for gpu computation.")
#  gflags.DEFINE_string(
#    "crop_mode", "center_only", "Crop mode, from {}".format(CROP_MODES))
#  gflags.DEFINE_string(
#    "input_file", "", "Input txt/csv filename.")
#  gflags.DEFINE_string(
#    "output_file", "", "Output h5/csv filename.")
#  gflags.DEFINE_string(
#    "images_dim", 256, "Canonical dimension of (square) images.")
#  gflags.DEFINE_string(
#    "images_mean_file",
#    os.path.join(os.path.dirname(__file__), '../imagenet/ilsvrc_2012_mean.npy'),
#    "Data set image mean (numpy array).")
#  FLAGS = gflags.FLAGS
#  FLAGS(sys.argv)
#
#  # Configure network, input, output
#  config(FLAGS.model_def, FLAGS.pretrained_model, FLAGS.gpu, FLAGS.images_dim,
#         FLAGS.images_mean_file)
#
#  # Load input
#  # .txt = list of filenames
#  # .csv = dataframe that must include a header
#  #        with column names filename, ymin, xmin, ymax, xmax
#  t = time.time()
#  print('Loading input and assembling batches...')
#  if FLAGS.input_file.lower().endswith('txt'):
#    with open(FLAGS.input_file) as f:
#      inputs = [_.strip() for _ in f.readlines()]
#  elif FLAGS.input_file.lower().endswith('csv'):
#    inputs = pd.read_csv(FLAGS.input_file, sep=',', dtype={'filename': str})
#    inputs.set_index('filename', inplace=True)
#  else:
#    raise Exception("Uknown input file type: not in txt or csv")
#
#  # Assemble into batches
#  image_batches = assemble_batches(inputs, FLAGS.crop_mode)
#  print('{} batches assembled in {:.3f} s'.format(len(image_batches),
#                                                  time.time() - t))
#
#  # Process the batches.
#  t = time.time()
#  print 'Processing {} files in {} batches'.format(len(inputs),
#                                                   len(image_batches))
#  dfs_with_feats = []
#  for i in range(len(image_batches)):
#    if i % 10 == 0:
#      print('Batch {}/{}, elapsed time: {:.3f} s'.format(i,
#                                                         len(image_batches),
#                                                         time.time() - t))
#    dfs_with_feats.append(compute_feats(image_batches[i]))
#
#  # Concatenate, droppping the padding rows.
#  df = pd.concat(dfs_with_feats).dropna(subset=['filename'])
#  df.set_index('filename', inplace=True)
#  print("Processing complete after {:.3f} s.".format(time.time() - t))
#
#  # Label coordinates
#  coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
#  df[coord_cols] = pd.DataFrame(data=np.vstack(df['window']),
#                                index=df.index,
#                                columns=coord_cols)
#  del(df['window'])
#
#  # Write out the results.
#  t = time.time()
#  if FLAGS.output_file.lower().endswith('csv'):
#    # enumerate the class probabilities
#    class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
#    df[class_cols] = pd.DataFrame(data=np.vstack(df['feat']),
#                                  index=df.index,
#                                  columns=class_cols)
#    df.to_csv(FLAGS.output_file, sep=',',
#              cols=coord_cols + class_cols,
#              header=True)
#  else:
#    df.to_hdf(FLAGS.output_file, 'df', mode='w')
#  print("Done. Saving to {} took {:.3f} s.".format(
#    FLAGS.output_file, time.time() - t))
#
#  sys.exit()
