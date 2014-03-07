// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include<fstream>

using std::max;

namespace caffe {

template <typename Dtype>
void SiameseLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SiameseLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "SiameseLoss Layer takes no blob as output.";
  NUM_ = bottom[0]->num();
  CHECK_EQ(NUM_ % 2,0) << "SiameseLoss Layer accept only even number of samples";
  PAIR_NUM_ = NUM_ / 2;
  DIM_  = bottom[0]->count()/NUM_;
  Q_    = DIM_ * 2 * this->layer_param_.input_val_range();
  beta_ = this->layer_param_.imposter_power();
  alpha_= this->layer_param_.imposter_scale();
  // exp_neg_beta_ = (Dtype)std::exp(-beta_);
  regular_weight_= this->layer_param_.regular_weight();

  //LOG(INFO) << "Layer DIM " << DIM_;

  block_data_.reset( new SyncedMemory(DIM_*sizeof(Dtype)) );
  loss_data_.reset( new SyncedMemory(PAIR_NUM_*sizeof(Dtype)) );
  error_data_.reset( new SyncedMemory(PAIR_NUM_*sizeof(Dtype)) );

};

template <typename Dtype>
void SiameseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label  = bottom[1]->cpu_data();
	Dtype* loss  = reinterpret_cast<Dtype*>( loss_data_->mutable_cpu_data() );
	Dtype* block = reinterpret_cast<Dtype*>( block_data_->mutable_cpu_data() );
	Dtype* error = reinterpret_cast<Dtype*>( error_data_->mutable_cpu_data() );
	for ( int i=0; i<PAIR_NUM_; ++i ) {
		caffe_sub<Dtype>( DIM_, bottom_data + (i*2)*DIM_  ,
				bottom_data + (i*2+1)*DIM_, block );
		Dtype e = caffe_cpu_asum<Dtype>( DIM_, block );
		if ( label[i*2] == label[i*2+1] ) {
			error[i] = e;
			loss[i] = (Dtype)2./Q_*e*e;
		} else {
			error[i] = alpha_*(std::exp(-beta_/Q_*e));
			// loss[i] = (Dtype)2.*Q_*std::max( (Dtype)0., error[i] - exp_neg_beta_*e );
			loss[i] = (Dtype)2.*Q_*error[i];
		}
        loss[i] /= (Dtype)NUM_;
	}

    /*
    {
        std::ofstream log("bottom.txt");
        log.precision(10);
        for ( int i=0; i<NUM_; ++i ) {
            for ( int j=0; j<DIM_; ++j ) {
                if (j) log << ", ";
                log << bottom_data[i*DIM_+j];
            }
            log << std::endl;
        }
    }
    {
        std::ofstream log("loss.txt");
        log.precision(10);
        for ( int i=0; i<PAIR_NUM_; ++i ) {
            log << loss[i] << std::endl;
        }
    }
    */


}

template <typename Dtype>
void SiameseLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
	Forward_cpu( bottom, top );
}

template <typename Dtype>
Dtype SiameseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {

  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* loss  = reinterpret_cast<const Dtype*>( loss_data_->cpu_data() );
  const Dtype* error = reinterpret_cast<const Dtype*>( error_data_->cpu_data() );
  Dtype* x1_x2 = reinterpret_cast<Dtype*>( block_data_->mutable_cpu_data() );
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* label = (*bottom)[1]->cpu_data();

  for ( int i=0; i<PAIR_NUM_; ++i ) {
	const Dtype* x1 = bottom_data + (i*2)*DIM_;
	const Dtype* x2 = bottom_data + (i*2+1)*DIM_;
	Dtype* dx1 = bottom_diff + (i*2)*DIM_;
	Dtype* dx2 = bottom_diff + (i*2+1)*DIM_;


	//caffe_copy<Dtype>( DIM_, x1, x1_x2 );
	//caffe_axpy<Dtype>( DIM_, (Dtype)-1., x2, x1_x2 );

    caffe_sub<Dtype>(DIM_, x1, x2, x1_x2);
	for ( int j = 0; j<DIM_; ++j ) {
		dx1[j] = (x1_x2[j]>(Dtype)0.)?(Dtype)1.:
				((x1_x2[j]<(Dtype)0.)?(Dtype)-1.:(Dtype)0.);
	}

	if ( label[i*2] == label[i*2+1] ) {
		caffe_scal<Dtype>( DIM_, (Dtype)4./Q_*error[i]/(Dtype)NUM_, dx1 );
	} else {
		caffe_scal<Dtype>( DIM_, 
                (Dtype)-2*beta_*error[i]/(Dtype)NUM_, dx1 );
		//caffe_scal<Dtype>( DIM_, -(Dtype)4./Q_*error[i]/(Dtype)NUM_, dx1 );
	}
	caffe_copy<Dtype>( DIM_, dx1, dx2 );
	caffe_scal<Dtype>( DIM_, (Dtype)-1., dx2 );

    /// regularizor 
    const Dtype eps = (Dtype)1e-5;
	Dtype* block = reinterpret_cast<Dtype*>( block_data_->mutable_cpu_data() );
    for ( int j=0;j<DIM_;++j )
        block[j]=x1[j]*(1/(1-x1[j]*x1[j]+eps)-2) + x2[j]*(1/(1-x2[j]*x2[j]+eps)-2);

    caffe_axpy( DIM_, regular_weight_, block, dx1 );
    caffe_axpy( DIM_, regular_weight_, block, dx2 );

    // cap gradient
    const Dtype grad_bound = (Dtype)0.3;
    for ( int j=0;j<DIM_;++j ) {
        dx1[j] = std::min(std::max(dx1[j],-grad_bound),grad_bound);
        dx2[j] = std::min(std::max(dx2[j],-grad_bound),grad_bound);
    }

	//Debug
    // LOG(INFO) << "DX1: " << dx1[0] << " " << dx1[2] << " " << dx1[3];
	//memset( dx1, 0, sizeof(Dtype)*DIM_ );
	//memset( dx2, 0, sizeof(Dtype)*DIM_ );
  }
    /*
    {
        std::ofstream log("bottom_diff.txt");
        log.precision(10);
        for ( int i=0; i<NUM_; ++i ) {
            for ( int j=0; j<DIM_; ++j ) {
                if (j) log << ", ";
                log << bottom_diff[i*DIM_+j];
            }
            log << std::endl;
        }
    }
    */
  // compute the mean loss. As losses are positive, we can use asum to perform this
  Dtype mean_loss = caffe_cpu_asum<Dtype>( PAIR_NUM_, loss ) / PAIR_NUM_;

  return mean_loss;
}

template <typename Dtype>
Dtype SiameseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SiameseLossLayer);


}  // namespace caffe
