# Caffe support for Deep Compositional Networks with Gaussian Convolution Layer 

By constraining filter weights in CNN we can force deep networks to learn compositional hiearhices. Each composition is modeled with a Gaussian component which can also be implemnted as a seperable filter for speedup in forward pass (currently only supported experimantaly on CPU).

## Build notes

- Only CMake build supported
- CMake automatically enables C++11 support 
- Repository includes full Nvidia CUB library through git-submodule so do not forget to call `git submodule update`. 
- GuassianConv layer supports CPU and GPU, as well as CUDNN engine.

Please cite the following publication for Guassian Convolution Layer:

    @inproceedings{Tabernik2016ICPR,
      Author = {Tabernik, Domen and Kristan, Matej and Wyatt, Jeremy and Leonardis, Ale\v{s}}
      booktitle = {International Conference on Pattern Recognition},
      Title = {Towards Deep Compositional Networks},
      Year = {2016}
    }

## Usage 

You can follow the example at ```examples/cifar10/gauss-cnn/cifar10_gauss_train_test.prototxt```.

Enable Guassian Convolution Layer with `type: "GaussianConv"`. You can use following settings in your protobuf definition:
```
layer {  
  type: "GaussianConv"  
  ...
  # param for Gaussian weight
  param {
    lr_mult: 1
    decay_mult: 0
  }
  # param for Gaussian mean in X direction
  param {
    lr_mult: 50
    decay_mult: 0
  }
  # param for Gaussian mean in Y direction
  param {
    lr_mult: 50
    decay_mult: 0
  }
  # param for Gaussian variance
  param {
    lr_mult: 5
    decay_mult: 0
  }
  # dummy - old depricated paramater
  param {
    lr_mult: 0
    decay_mult: 0
  }
  # bias
  param {
    lr_mult: 1
  }
  
  convolution_param {
    engine: CUDNN # GaussianConv currently does not use CUDNN by default
    
    # standard Convolution layer parameters
    num_output: 96
    kernel_size: 7
    pad: 3
    stride: 1 # ONLY stride=1 curently supported !!
    
    bias_filler {
      type: "constant"
    }
    
    # GaussianConv layer specific parameters
    number_gauss: 2 # in each dimension (threfore 4 gauss components per kernel in this case)
    
    # Initialization of Gaussian weights
    weight_filler {
      type: "gaussian"
      std: 0.01
      mean: 0
    }
    # Initialization of Gaussian variance
    sigma_filler {
      type: "constant"
      value: 0.8     
    }    
    
    # Explicitly set other GaussianConv layer related parameters since they works better then default ones:
    gmm_weight_normalization: false 
    gmm_gauss_normalization: true
    gmm_mean_iteration_step: 1
    gmm_sigma_iteration_step: 1
    gmm_sigma_lower_bound: 0.5
    gmm_component_border_bound: 1.5
  }
}
```
Some other parameters can be found in caffe.proto:

```
  optional bool gmm_weight_normalization = 123 [default = true]; // Should weights of GMM be normalized (i.e. should sum to 1)
  optional bool gmm_gauss_normalization = 124 [default = true]; // Should gaussian GMM be normalized or not  (i.e. area under gauss should sum to 1)
  optional uint32 gmm_mean_iteration_step = 125 [default = 1]; // step between optimization iteration of mean in gaussian
  optional uint32 gmm_sigma_iteration_step = 126 [default = 1]; // step between optimization iteration  of sigma in gaussian
  optional float gmm_component_border_bound = 127 [default = 0]; // how close to kernel border is gaussian component allowed to be
  optional float gmm_sigma_lower_bound = 128 [default = 0.1]; // lower bound for sigma
  optional bool gmm_square_gauss_normalization = 129 [default = true]; // normalize using sum of squares 
  optional bool gmm_seperable_forward_pass = 130 [default = false]; // utilize seperable kernels to speed-up forward-pass
  optional bool gmm_use_old_cudnn = 131 [default = false]; // use old version of cudnn implmentation (uses a lot of memory)
```

---------------------------------------------------------------
# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
