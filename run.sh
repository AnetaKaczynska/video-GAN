#!/bin/bash

#SBATCH --job-name=video_GAN
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=normal  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgxmatinf  # rtx2080 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Q os:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"

datasets=/shared/results/Skopia/noise_for_images
checkpoints=/shared/results/struski/videoGAN/AC-ProGAN_2022-10-14_141007
dir_results=/shared/results/z1188643/videoGAN

mkdir -p $dir_results
singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_clipping.py