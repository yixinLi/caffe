// Copyright 2013 Yangqing Jia

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>

using std::max;

namespace caffe {

template <typename Dtype>
void TanhLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Tanh Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Tanh Layer takes a single blob as output.";
  if ( bottom[0] == (*top)[0] ) {
	  // in-place
	  buffer_.reset( new SyncedMemory( bottom[0]->count() * sizeof(Dtype) ) );
  } else {
	  // not in-place
	  int num      = bottom[0]->num();
	  int channels = bottom[0]->channels();
	  int height   = bottom[0]->height();
	  int width    = bottom[0]->width();
	  (*top)[0]->Reshape( num, channels, height, width );
  }
}

template <typename Dtype>
void TanhLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_cpu_tanh<Dtype>( count, bottom_data, top_data );
}

template <typename Dtype>
Dtype TanhLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		Dtype* buffer;
		if ((*bottom)[0] == top[0]) { // in-place
			buffer = reinterpret_cast<Dtype*>(buffer_->mutable_cpu_data());
		} else { // not in-place
			buffer = (*bottom)[0]->mutable_cpu_diff();
		}
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		caffe_cpu_sech2(count, bottom_data, buffer);
		caffe_mul(count, top_diff, buffer, bottom_diff);

        /*
        int dim = (*bottom)[0]->channels()*(*bottom)[0]->height()*(*bottom)[0]->width();
        for ( int i=0; i<(*bottom)[0]->num(); ++i ) {
            const Dtype* d = bottom_diff + i*dim;
            const Dtype* b = bottom_data + i*dim;
            LOG(INFO) << "Tanh-Bottom:     " << b[0] << " " << b[2] << " " << b[3];
            LOG(INFO) << "Tanh-BottomDiff: " << d[0] << " " << d[2] << " " << d[3];
        }
        */
	}
	return Dtype(0);
}

template <typename Dtype>
void TanhLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_gpu_tanh<Dtype>( count, bottom_data, top_data );
}

template <typename Dtype>
Dtype TanhLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		Dtype* buffer;
		if ((*bottom)[0] == top[0]) { // in-place
			buffer = reinterpret_cast<Dtype*>(buffer_->mutable_gpu_data());
		} else { // not in-place
			buffer = (*bottom)[0]->mutable_gpu_diff();
		}
		const Dtype* bottom_data = (*bottom)[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		const int count = (*bottom)[0]->count();
		caffe_gpu_sech2(count, bottom_data, buffer);
		caffe_gpu_mul(count, top_diff, buffer, bottom_diff);
	}
	return Dtype(0);
}

INSTANTIATE_CLASS(TanhLayer);


}  // namespace caffe
