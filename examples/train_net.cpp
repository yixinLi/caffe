// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
	    LOG(ERROR) << "Usage: train_net device_id solver_proto_file [resume_point_file]";
	    LOG(ERROR) << "Usage: train_net device_id solver_proto_file [pretrained_model_file] --finetune";
    return 0;
  }

  int device_id = std::atoi(argv[1]);
  caffe::Caffe::SetDevice( device_id );

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[2], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  if (argc >= 4) {
	  int resume_type = Solver<float>::RESUME_FROM_CHECKPOINT;
	  if ( argc >=5 && std::string(argv[4]) == "--finetune" ) {
		  resume_type = Solver<float>::FINETUNE_PRETRAINED;
		  LOG(INFO) << "Finetuning from " << argv[3];
	  } else
		  LOG(INFO) << "Resuming from " << argv[3];
	  solver.Solve(argv[3],resume_type);
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
