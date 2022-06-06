#!/bin/bash

port=$1
if [[ "$port" == "" ]]; then
  port=6009
fi


singularity_images=""
if [[ -f /shared/sets/singularity/miniconda_pytorch_py39.sif ]]; then
  singularity_images=/shared/sets/singularity/miniconda_pytorch_py39.sif
  dir_results=/shared/results/struski/videoGAN
else
  singularity_images=$HOME/singularity_images/miniconda_pytorch_py39.sif
  dir_results=/local/results/videoGAN
fi


singularity exec -B $dir_results:/results $singularity_images tensorboard --logdir /results --port $port

# tensorboard --logdir_spec name1:/path/to/logs/1,name2:/path/to/logs/2
