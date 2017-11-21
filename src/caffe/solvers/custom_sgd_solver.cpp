#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {


template <typename Dtype>
void CustomSGDSolver<Dtype>::Regularize(int param_id) {

	// if regularization should decay weights to specific target value X we need to subtract X before doing regularization
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	const vector<float>& net_params_weight_decay_target = this->net_->params_weight_decay_target();

	Dtype decay_target = net_params_weight_decay_target[param_id];

	if (decay_target != 0) {
		switch (Caffe::mode()) {
		  case Caffe::CPU: {
			  caffe_add_scalar(net_params[param_id]->count(),
					  				-1* decay_target,
									net_params[param_id]->mutable_cpu_data());
		    break;
		  }
		  case Caffe::GPU: {
		#ifndef CPU_ONLY
			  caffe_gpu_add_scalar(net_params[param_id]->count(),
			  					  	-1* decay_target,
			  			            net_params[param_id]->mutable_gpu_data());
		#else
		    NO_GPU;
		#endif
		    break;
		  }
		  default:
		    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		  }
	}

	SGDSolver<Dtype>::Regularize(param_id);

	// restore weights back to original value
	if (decay_target != 0) {
		switch (Caffe::mode()) {
		  case Caffe::CPU: {
			  caffe_add_scalar(net_params[param_id]->count(),
									decay_target,
									net_params[param_id]->mutable_cpu_data());
			break;
		  }
		  case Caffe::GPU: {
		#ifndef CPU_ONLY
			  caffe_gpu_add_scalar(net_params[param_id]->count(),
									decay_target,
									net_params[param_id]->mutable_gpu_data());
		#else
			NO_GPU;
		#endif
			break;
		  }
		  default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		  }
	}
}

INSTANTIATE_CLASS(CustomSGDSolver);
REGISTER_SOLVER_CLASS(CustomSGD);

}  // namespace caffe
