#!/bin/bash

#SBATCH --job-name=video_GAN
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=normal  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=rtx2080  # rtx2080 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"


#datasets=/shared/results/Skopia/videos_8frames
datasets=/shared/results/Skopia/videos24frames
checkpoints=/shared/results/z1143165/jelito3d_batchsize8
dir_results=/shared/results/struski/videoGAN


mkdir -p $dir_results

if [[ "$1" == "resume" ]]; then

singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD2_alphaSimil0_seed5018_220602-102758/checkpoints/frame_seed_generator.pt --num_samples 5 --batch_size 10
singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD2_alphaSimil0_seed5018_220602-102758/checkpoints/frame_seed_generator_maxSimilarity.pt --num_samples 5 --batch_size 10 --name "minLOSS"
#singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD2_alphaSimil1_seed6878_220602-104237/checkpoints/frame_seed_generator.pt --num_samples 5 --batch_size 10
#singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD2_alphaSimil1_seed6878_220602-104237/checkpoints/frame_seed_generator_maxSimilarity.pt --num_samples 5 --batch_size 10 --name "minLOSS"
#singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD5_alphaSimil0_seed5114_220602-104144/checkpoints/frame_seed_generator.pt --num_samples 5 --batch_size 10
#singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path /results/clipping_frame24_iterD5_alphaSimil0_seed5114_220602-104144/checkpoints/frame_seed_generator_maxSimilarity.pt --num_samples 5 --batch_size 10 --name "minLOSS"

#singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_create_movie.py --resume_path $1

elif [[ "$1" == "1" ]]; then

singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_clipping.py

else

singularity exec --nv -B $datasets:/datasets -B $checkpoints:/checkpoints -B $dir_results:/results /shared/sets/singularity/miniconda_pytorch_py39.sif python training_gradient_penalty.py

fi
