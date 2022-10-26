#!/bin/bash

singularity_images=$HOME/singularity_images/miniconda_pytorch_py39.sif
datasets=/media/IN-167
dir_results=/local/results/videoGAN

mkdir -p "${dir_results}"

#singularity exec -B $datasets:/data -B $dir_results:/results "$singularity_images" python -u extract_sharp_images.py --sharp_images /results/images &>/dev/null

# test
singularity exec -B $datasets:/data -B $dir_results:/results "$singularity_images" python -u extract_sharp_images.py --sharp_images /results/images --test
