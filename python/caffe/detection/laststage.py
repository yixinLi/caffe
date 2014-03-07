"""
SVM Classification using the features extracted from ImageNet
pre-trained CNN. One SVM for each Pascal VOC category.
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
from sklearn.svm import SVC
import detector

import scipy.io
import glob
import time

VOC2007_RawJPEG = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/JPEGImages/'

class LastStageClassifier:

  def __init__(self):
    self.IoU_pos_example_thresh = 0.999 # use gt only as positive examples
    self.IoU_neg_example_thresh = 0.3
    self.IoU_thresh_nms = 0.5
    self.VOC_min_IoU = 0.5

    self.num_classes = 0
    self.classes_to_train = []
    self.svms = []
    # self.max_proposed_box_per_im = -1
    self.feats_size = 0
    self.max_iter = 3

    self.max_X_rows = 10000

  def init_svms_and_datasets(self):
    self.svms = []
    if self.num_classes == 0:
      print("Bad value: number of classes is 0 during init_svms_and_datasets")

    for i in range(self.num_classes):
      a_linear_svm = SVC(kernel='linear',probability=True)
      self.svms.append(a_linear_svm)

    self.Xfn_tr = [None]*self.num_classes
    self.Xwindows_tr = [None]*self.num_classes
    self.y_tr = [None]*self.num_classes
    self.gtBoxes_tr = [None]*self.num_classes

    self.Xfn_va = [None]*self.num_classes
    self.Xwindows_va = [None]*self.num_classes
    self.y_va = [None]*self.num_classes
    self.gtBoxes_va = [None]*self.num_classes

    # set with no y labels
    #self.Xfn_te = [None]*self.num_classes
    #self.Xwindows_te = [None]*self.num_classes

  def setup_training(self,tr_gt_fns,tr_feats_dirs,va_gt_fns,va_feats_dirs,feats_choice,num_classes):
    self.num_classes = num_classes
    self.init_svms_and_datasets()

    # setup training set
    print("Setup training set")
    for set_i in range(len(tr_gt_fns)):
      (Xfn,Xwindows,y,gtBoxes) = self.setup_examples(tr_gt_fns[set_i],tr_feats_dirs[set_i],feats_choice)
      for i in range(num_classes):
        if self.Xfn_tr[i] == None:
          self.Xfn_tr[i] = Xfn[i]
          self.Xwindows_tr[i] = Xwindows[i]
          self.y_tr[i] = y[i]
          self.gtBoxes_tr[i] = gtBoxes[i] 
        else:
          self.Xfn_tr[i] = np.append(self.Xfn_tr[i],Xfn[i])
          self.Xwindows_tr[i] = np.append(self.Xwindows_tr[i],Xwindows[i],axis=0)
          self.y_tr[i] = np.append(self.y_tr[i],y[i])
          self.gtBoxes_tr[i] = np.append(self.gtBoxes_tr[i],gtBoxes[i],axis=0)
    
    # setup validation set
    print("Setup validation set")
    for set_i in range(len(va_gt_fns)):
      (Xfn,Xwindows,y,gtBoxes) = self.setup_examples(va_gt_fns[set_i],va_feats_dirs[set_i],feats_choice)
      for i in range(num_classes):
        if self.Xfn_va[i] == None:
          self.Xfn_va[i] = Xfn[i]
          self.Xwindows_va[i] = Xwindows[i]
          self.y_va[i] = y[i]
          self.gtBoxes_va[i] = gtBoxes[i]
        else:
          self.Xfn_va[i] = np.append(self.Xfn_va[i],Xfn[i])
          self.Xwindows_va[i] = np.append(self.Xwindows_va[i],Xwindows[i],axis=0)
          self.y_va[i] = np.append(self.y_va[i],y[i])
          self.gtBoxes_va[i] = np.append(self.gtBoxes_va[i],gtBoxes[i],axis=0)

  def tmp_generate_files_for_ytz_finetune(self,write_filename,write_Xfn,write_Xwindows,gt_fn):
    jpeg_07_dir = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/JPEGImages/'
    gt_mat = scipy.io.loadmat(gt_fn)

    all_windows_all_labels = {}
    for class_i in range(self.num_classes):
      print('loading all windows of class {}'.format(class_i))
      Xfn = write_Xfn[class_i]
      Xwindows = write_Xwindows[class_i]

      y = np.zeros(Xfn.shape[0])
      gtBoxes
      (is_pos,is_neg) = self.find_pos_and_neg_examples(Xwindows_buf,gtBoxes_buf,0.5,0.5)
      
      for (idx,myXfn) in enumerate(Xfn):
        import re
        im_number = re.search('Feats/(.+?)Feat',myXfn)
        feats_number = myXfn.split(':')[1]
        mykey = im_number+':'+feats_number

        mylabel = (class_i+1)*y[idx]
        try:
          all_windows_all_labels[mykey].append((class_i+1)*y[idx])
        except KeyError:
          all_windows_all_labels[mykey] = [Xwindows[idx]]
          all_windows_all_labels[mykey].append((class_i+1)*y[idx])

      ## add gtBoxes to all_windows_all_labels here
      for j in range(gt_mat['gtIms'].shape[0]):
        im_number = str(gt_mat['gtIms'][class_i,0][j,0][0])
        mykey = im_number+':gt'+str(j)
        all_windows_all_labels[mykey] = [ gt_mat['gtBoxes'][class_i,0][j,:].reshape(1,4) ]
        all_windows_all_labels[mykey].append(class_i+1)

    all_keys = all_windows_all_labels.keys()
    all_keys.sort()

    f_ytz_ft = open(write_filename,'w')
    print('start writing all labels file')
    for key in all_keys: # '<im_number>:<feat_number>'
      print(key),
      all_labels = np.array(all_windows_all_labels[key][1:])
      pos_labels = all_labels[all_labels>0]

      line_prefix = key.split(':')[0] + ' ' + str(all_windows_all_labels[key][0].astype(int))[1:-1] + ' '
      if pos_labels.shape[0] == 0:
        f_ytz_ft.write(line_prefix+'-1\n')
      else:
        for pos_label in pos_labels:
          f_ytz_ft.write(line_prefix+str(pos_label.astype(int))+'\n')
    f_ytz_ft.close()
  
  def train_selected_classes(self,classes_to_train=None):
    """
    classes_to_train : [1 0 0 1 ... 1] whether to train a classifier on a class, 
                     1 means TRAIN, 0 means DO NOT TRAIN
                     default is to train all classes
    """
    if classes_to_train == None:
      classes_to_train = [1]*self.num_classes
    
    self.mylogf = open('mylogf_{}.log'.format(str(int(time.time()))[-6:]),'w')

    for class_i in range(self.num_classes):
      print("training class "+str(class_i)+" classifier")
      if classes_to_train[class_i] == 1:
        self.train_iteratively(class_i)

    self.mylogf.close()

  def train_iteratively(self,class_i,max_iter=0):
    if max_iter == 0:
      max_iter = self.max_iter

    
    iter = 0
    while iter < max_iter:
      print("==========================ITERATION {}==============================".format(iter))
      # choose sample (random sample or negative mining) on self.Xwindows_tr[i]
      num_pos = min(np.sum(np.concatenate(self.y_tr[class_i])==1),self.max_X_rows)
      num_neg = num_pos*10
      print("number of positive examples: {}".format(num_pos))
      print("number of negative examples: {}".format(num_neg))
      if iter == 0:
        print("============Begin random sampling for training set====================")
        trandsample = time.time()
        (idx,X,y) = self.generate_train_set(class_i,num_pos,num_neg,'random')
        print("time took to random sample for a training set: {}".format(time.time() - trandsample))
      else:
        print("============Begin hard negative mining for training set================")
        thardnegm = time.time()
        (idx,X,y) = self.generate_train_set(class_i,num_pos,num_neg,'hardneg')
        print("time took to hard negative mining: {}".format(time.time() - thardnegm))

      # train SVM
      print("==============Fitting SVM to training set===============================")
      X_stacked = np.vstack(X)
      y_stacked = np.concatenate(y)
      print(X_stacked.shape)
      print(y_stacked.shape)
      #import pdb; pdb.set_trace()

      tfit = time.time()
      self.svms[class_i].fit(X_stacked,y_stacked)
      print("time took to fit: {}".format(time.time() - tfit))

      # get training score for mAP
      #self.evaluate_model(class_i,self.Xfn_tr,self.Xwindows_tr,self.gtBoxes_tr,'TRAINING')
      
      # get validation score for mAP
      if self.Xfn_va[class_i] != None:
        self.evaluate_model(class_i,self.Xfn_va,self.Xwindows_va,self.gtBoxes_va,'VALIDATION')

      iter = iter+1

  def evaluate_model(self,class_i,Xfn,Xwindows,gtBoxes,set_name='TRAINING',IoU_thresh_nms=None):
    if IoU_thresh_nms == None:
      IoU_thresh_nms = self.IoU_thresh_nms

    print("============Get prediction and AP score on {} set==============".format(set_name))
    t_eval = time.time()
    (pred, pred_lp) = self.get_pred_and_lp(class_i, Xfn[class_i])
    scores = [ pred_lp_one_im[:,1] for pred_lp_one_im in pred_lp]

    (windows_nms, scores_nms) = self.nms_all_image(Xwindows[class_i], scores, IoU_thresh_nms)
    #import pdb; pdb.set_trace()

    scores_nms_stacked = np.concatenate(scores_nms)
    scores_stacked = np.concatenate(scores)
    print("Eliminated {} windows, from {} to {}".format( scores_stacked.shape[0]-scores_nms_stacked.shape[0],\
      scores_stacked.shape[0], scores_nms_stacked.shape[0]) )

    (ap, PR_curve) = self.calc_ap_score(scores_nms, windows_nms, gtBoxes[class_i], PR_num_bins=30)
    print("time took to evaluate on {} set: {}".format(set_name,time.time() - t_eval))

    # print score
    print("===================================================")
    print("AP on class {} {}: {}".format(class_i,set_name,ap))
    print(PR_curve)
    print("===================================================")

    self.mylogf.write("===================================================")
    self.mylogf.write("AP on class {} {}: {}".format(class_i,set_name,ap))
    self.mylogf.write(repr(PR_curve))
    self.mylogf.write("===================================================")
    self.mylogf.flush()

  def calc_ap_score(self,scores,pred_windows,gtBoxes,PR_num_bins):
    """
    Essentially VOC2007 devkits implementation
    """

    IoU_with_gt_all_im = []
    for im_idx in range(len(pred_windows)):
      IoU_with_gt = []
      for gtBox in gtBoxes[im_idx]:
        IoU_with_gt.append(self.calc_IoU(pred_windows[im_idx],gtBox))
      IoU_with_gt = np.vstack(IoU_with_gt) # (number of gt boxes X number of prediction windows)
      max_IoU = np.max(IoU_with_gt,axis=0)
      max_IoU_gtidx = np.argmax(IoU_with_gt,axis=0)
      max_IoU_im_idx = np.array([im_idx]*pred_windows[im_idx].shape[0])

      IoU_with_gt_all_im.append(np.vstack([max_IoU,max_IoU_gtidx,max_IoU_im_idx]).transpose())
    IoU_with_gt_all_im = np.vstack(IoU_with_gt_all_im) # (3 X prediction windows over all images)

    scores_stacked = np.concatenate(scores)
    # descending index for scores (and also IoU_with_gt_all_im)
    d_idx = scores_stacked.argsort()[::-1]
    ns = d_idx.shape[0]

    tp = np.zeros(ns)
    fp = np.zeros(ns)
    detected = {}

    for i in range(ns):
      gtkey = tuple(IoU_with_gt_all_im[d_idx[i],1:]) # ground truth box with best IoU
      if gtkey in detected:
        fp[i] = 1
        detected[gtkey] = detected[gtkey]+1 # not useful for now
      else:
        if IoU_with_gt_all_im[d_idx[i],0] >= self.VOC_min_IoU:
          tp[i] = 1
          detected[gtkey] = 1
        else:
          fp[i] = 1

    num_pos = np.vstack(gtBoxes).shape[0]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = np.divide(tp,num_pos)
    precision = np.divide(tp,(fp+tp))

    # average precision and precision/recall curve
    PR_curve = np.zeros(PR_num_bins)
    for i in range(PR_num_bins):
      t = float(i)/(PR_num_bins-1)
      try:
        PR_curve[i] = np.max(precision[recall>=t])
      except ValueError: # when no recall is >= t
        PR_curve[i] = 0
    
    ap = np.mean(PR_curve)

    return (ap,PR_curve)

  def generate_train_set(self,class_i,num_pos,num_neg,negatives_selection_method='random'):
    if negatives_selection_method == 'random':
      posidx = self.random_sample_in_one_label(class_i,1,num_pos)
      negidx = self.random_sample_in_one_label(class_i,-1,num_neg)
    elif negatives_selection_method == 'hardneg':
      posidx = self.random_sample_in_one_label(class_i,1,num_pos)
      negidx = self.mine_hard_negatives(class_i,num_neg)
    else:
      raise Exception("invalid choice of negatives_selection_method")

    idx = [np.append(posidx[im_idx],negidx[im_idx]) for im_idx in range(len(posidx))]
    y = [np.array( posidx[im_idx].shape[0]*[1]+negidx[im_idx].shape[0]*[-1] ) for im_idx in range(len(posidx))]

    # Retrieve the features
    X = [None]*len(idx)
    for (im_idx,feats_idx_of_im) in enumerate(idx):
      fn = self.Xfn_tr[class_i][im_idx][0].split(':')[0]
      feats = pd.read_hdf(fn,'df')
      X[im_idx] = np.array(feats.ix[feats_idx_of_im,0:])
      # NOTE: I assumed feats has 'filename','ymin','xmin','ymax','xmax',0,1,..., as columns
      # that's why feats.ix[k,0:] works. It picks out columns, 0-4095.
  
    return (idx,X,y)
  
  def mine_hard_negatives(self,class_i,num_neg):
    neg_Xfn = [ Xfn[self.y_tr[class_i][im_idx] == -1] for im_idx,Xfn in enumerate(self.Xfn_tr[class_i])]
    (pred,pred_lp) = self.get_pred_and_lp(class_i,neg_Xfn)
    pred_lp_stacked = np.vstack(pred_lp)
    candidate_neg_idx = pred_lp_stacked[:,1].argsort()[::-1]
    # windows with highest confidence to be positives are at the beginning
    # NOTE: pos_prob = exp(pred_lp[:,1]) ; neg_prob = exp(pred_lp[:,0]) ; pos_prob+neg_prob==1
    
    # Get the indices of the negatives most confidently chosen as the positives
    all_y = np.concatenate(self.y_tr[class_i])
    negidx = candidate_neg_idx[:num_neg]

    #assert(num_neg <= np.sum(self.y_tr[class_i]==-1))
    # argh.. OCD. I am never going to use this feature, but just for symmetry with random_sample method.
    if num_neg > negidx.shape[0]:
      negidx = np.append(negidx, np.random.choice(negidx,num_neg-negidx.shape[0]) )

    negidx.sort()
    negidx = self.unstack_idx_to_each_im(negidx,self.Xfn_tr[class_i])

    return negidx

  def random_sample_in_one_label(self,class_i,wanted_label,num_wanted):
    # get all idx under wanted_label
    all_y_tr = np.concatenate(self.y_tr[class_i])
    widx = np.where(all_y_tr == wanted_label)[0]

    if num_wanted < widx.shape[0]:
      widx_c = np.random.choice(widx,size=num_wanted,replace=False)
    elif num_wanted == widx.shape[0]:
      widx_c = widx
    else:
      widx_c = np.random.choice(widx,size=num_wanted,replace=True)
    widx_c.sort()

    widx_c = self.unstack_idx_to_each_im(widx_c,self.Xfn_tr[class_i])

    return widx_c

  def unstack_idx_to_each_im(self,idx,reference_splitting):
    """
    Helper function
    idx indexes the feats/windows/gtboxes/y for all images of one class, after they are stacked together
    This function unstack (i.e. split) idx, so that it indexes the feats/windows/gtboxes/y for each image

    NOTE: idx does not have to be sorted
    """
    idx_unstacked = []
    zero_idx_offset = 0
    for im_idx in range(len(reference_splitting)):
      curr_im_num_feats = reference_splitting[im_idx].shape[0]
      is_in_range = np.logical_and(zero_idx_offset <= idx,idx < zero_idx_offset+curr_im_num_feats)
      idx_unstacked.append(idx[is_in_range] - zero_idx_offset)
      zero_idx_offset = zero_idx_offset + curr_im_num_feats

    return idx_unstacked

  def nms_all_image(self, all_boxes, all_scores, overlap=0.5):
    all_boxes_nms = []
    all_scores_nms = []

    assert(len(all_boxes) == len(all_scores))
    for im_idx in range(len(all_boxes)):
      (boxes_nms,scores_nms) = self.nms(all_boxes[im_idx],all_scores[im_idx],overlap)
      all_boxes_nms.append(boxes_nms)
      all_scores_nms.append(scores_nms)

    return (all_boxes_nms, all_scores_nms)

  def nms(self, boxes, scores, overlap=0.5):
    """
    Essentially the fast version of nms from Examplar-SVM
    https://github.com/quantombone/exemplarsvm/blob/master/internal/esvm_nms.m

    nms for boxes and scores of one image
    """
    if scores.shape[0] == 1:
      return (boxes,scores)

    pick_idx = np.zeros(scores.shape[0]) # index of the picked windows
    deleted = np.zeros(scores.shape[0]) # the ordering of scores and boxes
    didx = scores.argsort()[::-1] # index of scores sorted in descending order.
    # index of highest score is didx[0], index of lowest score is didx[1].
    # highest score means highest confidence on detection.

    pick_counter = 0
    for i in range(didx.shape[0]):
      if deleted[didx[i]] == 1: # already deleted/suppressed
        continue

      # not deleted/suppressed yet
      pick_idx[pick_counter] = didx[i]
      pick_counter = pick_counter + 1

      if i == didx.shape[0]-1:
        break

      #unprocessed_idx = didx[i+1:]
      IoU = self.calc_IoU(boxes[didx[i],:], boxes[didx[i+1:],:])
      deleted[didx[i+1:][IoU>overlap]] = 1

    pick_idx = pick_idx[:pick_counter]
    pick_idx.sort()
    pick_idx = pick_idx.astype(int)
    return (boxes[pick_idx,:],scores[pick_idx])

  def get_pred_and_lp(self,class_i,Xfn):
    pred = []
    pred_lp = []

    # make prediction once per image
    for im_idx in range(len(Xfn)):
      print(im_idx),
      fn = Xfn[im_idx][0].split(':')[0]

      feats = pd.read_hdf(fn,'df')
      #print("==WORKING ON "+Xfn[i].split(':')[0][-26:]+"=======")
      print("="+fn[-23:-15]+"="),
      
      if feats.shape[0] == Xfn[im_idx].shape[0]:
        #pred.append(self.svms[class_i].predict(feats.ix[:,0:]))
        pred_lp.append(self.svms[class_i].predict_log_proba(feats.ix[:,0:]))
      else:
        feats_idx = np.array([ int(Xfn[im_idx][Xfn_i].split(':')[1]) for Xfn_i in range(Xfn[im_idx].shape[0]) ])
        #pred.append(self.svms[class_i].predict(feats.ix[feats_idx,0:]))
        pred_lp.append(self.svms[class_i].predict_log_proba(feats.ix[feats_idx,0:]))

    return (pred,pred_lp)

  def setup_examples(self,gt_fn,feats_dir,feats_choice):
    gt_mat = scipy.io.loadmat(gt_fn)
    #num_classes = gt_mat['gtIms'].shape[0]
    #self.num_classes = num_classes
    num_classes = self.num_classes

    all_Xfn = [None]*num_classes
    all_Xwindows = [None]*num_classes
    all_y = [None]*num_classes
    all_gtBoxes = [None]*num_classes

    for i in range(num_classes):
      total_num_im = gt_mat['gtIms'][i,0].shape[0] # number of example images for this class
      
      #X = None
      Xfn = []
      Xwindows = []
      y = []
      gtBoxes = []

      gtBoxes_buf = np.array([]).reshape(0,4)
      #import pdb; pdb.set_trace()
      for j in range(total_num_im):
        print("i="+str(i)+" j="+str(j)+" total_num_im="+str(total_num_im))
        
        # Retrieve gt boxes
        im_number = str(gt_mat['gtIms'][i,0][j,0][0])
        gtBoxes_buf = np.append(gtBoxes_buf,gt_mat['gtBoxes'][i,0][j,:].reshape(1,4),axis=0)

        # Collect all gtBoxes of objects of same class in 1 image before comparing feats
        if j+1 < total_num_im:
          next_im_number = str(gt_mat['gtIms'][i,0][j+1,0][0])
          if im_number == next_im_number: # multiple objects of same class in 1 image
            continue

        # Retrieve feats for proposed candidate boxes
        im_fn = glob.glob(os.path.join(feats_dir,im_number+'Feat'+str(feats_choice)+'*'))
        if len(im_fn) > 1:
          raise Exception('more than one tr_im having tr_im_number and feats_choice')
        if len(im_fn) == 0:
          raise Exception('feats not found for image {}, feature {}'.format(im_number, feats_choice))
        im_fn = im_fn[0]
        feats = pd.read_hdf(im_fn,'df')
        if feats.shape[0] == 0:
          raise Exception('feats file contain 0 features for boxes')
        if self.feats_size == 0:
          self.feats_size = feats.ix[0,0:].shape[0]

        # load all windows
        Xwindows_buf = np.array(feats.ix[:,'ymin':'xmax'])
        num_windows = Xwindows_buf.shape[0]
        Xfn_buf = np.array([im_fn+':'+str(win_i) for win_i in range(num_windows)])
        y_buf = np.array([0]*num_windows) # 0 means do not use, 1 positive, -1 negative
        
        # Append all observations to Xfn Xwindows, y, gtBoxes
        y.append(y_buf)
        Xwindows.append(Xwindows_buf)
        Xfn.append(Xfn_buf)
        gtBoxes.append(gtBoxes_buf)
        
        # visualization for debugging
        #self.draw_Xwindows_gtBoxes(feats['filename'][0],Xwindows_buf,gtBoxes_buf,10)

        # clear gtBoxes buffer
        gtBoxes_buf = np.array([]).reshape(0,4)

      # get the assignment for the labels, according to find_pos_and_neg_examples method
      y = self.assign_labels(Xwindows,gtBoxes,y)

      # Add all processed examples of the i-th class
      all_Xfn[i] = Xfn
      all_Xwindows[i] = Xwindows
      all_y[i] = y
      all_gtBoxes[i] = gtBoxes

    return (all_Xfn, all_Xwindows, all_y, all_gtBoxes)

  def assign_labels(self,Xwindows,gtBoxes,y,pos_thresh=None,neg_thresh=None):
    """
    Decide whether each proposed candidate box is a positive or negative (or don't use) example of a class
    Xwindows, gtBoxes, y are lists. 
    Each element for Xwindows, gtBoxes, y of each image
    """
    for im_idx in range(len(Xwindows)):
      y[im_idx] = np.zeros(y[im_idx].shape) # clear previous labels
      (is_pos,is_neg) = self.find_pos_and_neg_examples( \
        Xwindows[im_idx],gtBoxes[im_idx],pos_thresh,neg_thresh)
      y[im_idx][is_pos] = 1
      y[im_idx][is_neg] = -1
    return y

  def draw_Xwindows_gtBoxes(self,im_filename,Xwindows,gtBoxes,max_number_windows_drawn=100):
    """
    Visualize ground truth boxes and proposed image windows

    max_number_windows_drawn: maximum number of Xwindows to be drawn
    Note: Always draw every gtBox, because gtBoxes assumed to be small (<10)
    """
    import matplotlib.pyplot as plt
    print("drawing image for debugging================")
    myimage =plt.imread(im_filename)
    plt.imshow(myimage)#,cmap=matplotlib.pylab.cm.color)

    best_IoUs = np.max(self.get_IoUs(Xwindows,gtBoxes),axis=1)
    best_IoUs_didx = best_IoUs.argsort()[::-1] # descending sort
    for (didx_i,didx) in enumerate(best_IoUs_didx):
      Xwindow = Xwindows[didx,:]
      plt.axhspan(Xwindow[0],Xwindow[2],float(Xwindow[1])/myimage.shape[1],float(Xwindow[3])/myimage.shape[1],color='b',Fill=False)
      print(best_IoUs[didx])
      print(Xwindows[didx,:])
      if didx_i >= max_number_windows_drawn-1:
        break

    for gtBox in gtBoxes:
      print(gtBox)
      plt.axhspan(gtBox[0],gtBox[2],float(gtBox[1])/myimage.shape[1],float(gtBox[3])/myimage.shape[1],color='y',Fill=False)
    plt.show()

  def find_pos_and_neg_examples(self,Xwindows,gtBoxes,pos_thresh=None,neg_thresh=None):
    """
    Return indexes to proposed candidate box, as positive and negative examples
    """
    IoU_mining = self.get_IoUs(Xwindows,gtBoxes)
    if pos_thresh == None:
      is_pos = np.max(IoU_mining,axis=1) > self.IoU_pos_example_thresh
    else:
      is_pos = np.max(IoU_mining,axis=1) > pos_thresh
    if neg_thresh == None:
      is_neg = np.max(IoU_mining,axis=1) < self.IoU_neg_example_thresh
    else:
      is_neg = np.max(IoU_mining,axis=1) < neg_thresh

    return (is_pos,is_neg)

  def get_IoUs(self,Xwindows,gtBoxes):
    """
    Return IoUs, a np.array of (# of Xwindows)X(# of gtBoxes)
    Each entry is the IoU between Xwindow (row) and gtBox (column)
    """
    all_gt_IoUs = []
    for gtBox in gtBoxes:
      IoUs = self.calc_IoU(Xwindows,gtBox)
      IoUs = IoUs.reshape(IoUs.shape[0],1)
      all_gt_IoUs.append(IoUs)
    
    return np.hstack(all_gt_IoUs)

  def calc_IoU(self,boxA,boxB):
    box1 = boxA
    box2 = boxB

    try:
      if len(box1.shape) == 1:
        box1 = box1.reshape(1,4)
      if len(box2.shape) == 1:
        box2 = box2.reshape(1,4)
    except ValueError:
      print("box1 and/or box2 is in an inappropriate format")
      return None
    assert(len(box1.shape) == 2 and len(box2.shape) == 2)
    
    if box1.shape[0] == 1 and box2.shape[0] > 1:
      box1 = np.tile(box1,[box2.shape[0],1])
    elif box1.shape[0] > 1 and box2.shape[0] == 1:
      box2 = np.tile(box2,[box1.shape[0],1])
    elif box1.shape[0] != box2.shape[0]:
      print("box1 and box2 doesn't have the same number of boxes, AND either one doesn't have one box")
      return None

    s_ymin = np.maximum(box1[:,0],box2[:,0])
    s_xmin = np.maximum(box1[:,1],box2[:,1])
    s_ymax = np.minimum(box1[:,2],box2[:,2])
    s_xmax = np.minimum(box1[:,3],box2[:,3])
    s_yoverlap = np.maximum(0,s_ymax-s_ymin+1)
    s_xoverlap = np.maximum(0,s_xmax-s_xmin+1)
    I_area = (s_yoverlap * s_xoverlap).astype(float)
    U_area = self.area(box1)+self.area(box2)-I_area

    return I_area/U_area

  def area(self,box):
    return (box[:,2]-box[:,0]+1)*(box[:,3]-box[:,1]+1)

###
#  def random_sample_train_set(self,class_i,num_pos,num_neg):
#    # get all positive/negative idx
#    posidx = np.array([])
#    negidx = np.array([])
#    for i in range(self.y_tr[class_i].shape[0]):
#      if self.y_tr[class_i][i] == 1: # positive example
#        posidx = np.append(posidx,i)
#      else: # negative example
#        negidx = np.append(negidx,i)
#    posidx_chosenidx = np.random.random_integers(0,posidx.shape[0],num_pos)
#    posidx_chosenidx.sort()
#    posidx_c = np.array([posidx[i] for i in posidx_chosenidx])
#    negidx_chosenidx = np.random.random_integers(0,negidx.shape[0],num_neg)
#    negidx_chosenidx.sort()
#    negidx_c = np.array([negidx[i] for i in negidx_chosenidx])
#
#    posX = None
#    # collect all feats for chosen positive examples
#    cache_fn = None
#    cache_feats = None
#    for i in posidx_c:
#      # checking whether the feats file is already open
#      if cache_fn == None:
#        cache_fn = Xfn_tr[class_i][i].split(':')[0]
#        cache_feats = pd.read_hdf(Xfn_tr[class_i][i].split(':')[0],'df')
#        posX = np.array([]).reshape(0,self.feats_size)
#      elif cache_fn != Xfn_tr[class_i][i].split(':')[0]:
#        cache_fn = Xfn_tr[class_i][i].split(':')[0]
#        cache_feats = pd.read_hdf(Xfn_tr[class_i][i].split(':')[0],'df')
#
#      feats_number = int(Xfn_tr[class_i][i].split(':')[1])
#      posX = np.append(posX,cache_feats.ix[feats_number,0:],axis=0)  
#    
#    negX = None
#    # collect all feats for chosen negative examples
#    cache_fn = None
#    cache_feats = None
#    for i in negidx_c:
#      # checking whether the feats file is already open
#      if cache_fn == None:
#        cache_fn = Xfn_tr[class_i][i].split(':')[0]
#        cache_feats = pd.read_hdf(Xfn_tr[class_i][i].split(':')[0],'df')
#        negX = np.array([]).reshape(0,self.feats_size)
#      elif cache_fn != Xfn_tr[class_i][i].split(':')[0]:
#        cache_fn = Xfn_tr[class_i][i].split(':')[0]
#        cache_feats = pd.read_hdf(Xfn_tr[class_i][i].split(':')[0],'df')
#
#      feats_number = int(Xfn_tr[class_i][i].split(':')[1])
#      negX = np.append(negX,cache_feats.ix[feats_number,0:],axis=0)
#
#    idx = np.append(posidx_c,negidx_c)
#    X = np.append(posX,negX,axis=0)
#    y = np.append([1]*(posX.shape[0]),[-1]*(negX.shape[0]))
#    return (idx,X,y)
###
###
#  def old_decide_pos_or_neg_examples(self,Xwindows,gtBoxes):
#    y_buf = np.array([])
#    IoU_mining = []
#
#    for Xwindow in Xwindows:
#      IoU_mining = []
#      for gtBox in gtBoxes:
#        IoU_mining.append(self.calc_IoU(Xwindow,gtBox))
#
#      if max(IoU_mining) >= self.IoU_thresh_mining:
#        y_buf = np.append(y_buf,1)
#      else:
#        y_buf = np.append(y_buf,-1)
#
#    return y_buf
###
###
#  def generate_files_for_ytz_finetune(self,write_filename,write_Xfn,write_Xwindows,write_y):
#    all_windows_all_labels = {}
#    for class_i in range(self.num_classes):
#      print('loading all windows of class {}'.format(class_i))
#      Xfn = write_Xfn[class_i]
#      Xwindows = write_Xwindows[class_i]
#      y = write_y[class_i]
#      
#      for (idx,myXfn) in enumerate(Xfn):
#        try:
#          all_windows_all_labels[myXfn].append((class_i+1)*y[idx])
#        except KeyError:
#          all_windows_all_labels[myXfn] = [Xwindows[idx]]
#          all_windows_all_labels[myXfn].append((class_i+1)*y[idx])
#
#    all_keys = all_windows_all_labels.keys()
#    all_keys.sort()
#
#    f_ytz_ft = open(write_filename,'w')
#    print('start writing all labels file')
#    for key in all_keys: # '<filename>:<feat_number>'
#      print(key)
#      all_labels = np.array(all_windows_all_labels[key][1:])
#      pos_labels = all_labels[all_labels>0]
#
#      line_prefix = key.split(':')[0] + ' ' + str(all_windows_all_labels[key][0].astype(int))[1:-1] + ' '
#      if pos_labels.shape[0] == 0:
#        f_ytz_ft.write(line_prefix+'-1\n')
#      else:
#        for pos_label in pos_labels:
#          f_ytz_ft.write(line_prefix+str(pos_label.astype(int))+'\n')
#    f_ytz_ft.close()
###
###
# Old code to stack features into a big matrix
# Retrieve the features in order, so to not open a feats file twice
#idx2widx = np.argsort(idx)
#widx = idx[idx2widx]
##wX = np.array([]).reshape(0,self.feats_size)
#wX = []
#cache_fn = None
#cache_feats = None
#for i in widx:
#  fn = self.Xfn_tr[class_i][i].split(':')[0]
#  # checking whether the feats file is already open
#  if cache_fn == None:
#    cache_fn = fn
#    cache_feats = pd.read_hdf(fn,'df')
#  elif cache_fn != fn:
#    cache_fn = fn
#    cache_feats = pd.read_hdf(fn,'df')
#  feats_number = int(self.Xfn_tr[class_i][i].split(':')[1])
#  #wX = np.append(wX, cache_feats.ix[feats_number,0:].reshape(1,self.feats_size), axis=0)
#  wX.append(cache_feats.ix[feats_number,0:].reshape(1,self.feats_size))
#wX = np.vstack(wX)
#X = np.zeros(wX.shape)
#X[idx2widx,:] = wX
###
###
#  def unstack_idx_and_y(self,idx,Xfn,y=None):
#    idx_unstacked = []
#    y_unstacked = []
#    zero_idx_offset = 0
#    for im_idx in range(len(Xfn)):
#      curr_im_num_feats = Xfn[im_idx].shape[0]
#      is_in_range = np.logical_and(zero_idx_offset <= idx,idx < zero_idx_offset+curr_im_num_feats)
#      idx_unstacked.append(idx[is_in_range] - zero_idx_offset)
#      if y != None:
#        y_unstacked.append(y[is_in_range])
#      zero_idx_offset = zero_idx_offset + curr_im_num_feats
#
#    return (idx_unstacked,y_unstacked)
###
