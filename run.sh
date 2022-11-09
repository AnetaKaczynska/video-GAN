#!/usr/bin/env bash

#SBATCH --job-name=videoGAN
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --qos=big  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgxa100  # rtx2080 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"


current_date=$( date +"%Y-%m-%d_%H%M%S" )
add_prox="/AC-ProGAN_${current_date}"
#add_prox="/AC-ProGAN_2022-10-28_122557"
#add_prox=""

singularity_images=""
if [[ -f /shared/sets/singularity/miniconda_pytorch_py39.sif ]]; then
  singularity_images=/shared/sets/singularity/miniconda_pytorch_py39.sif
#  datasets=/shared/results/Skopia/images_split2classes
  datasets=/shared/results/Skopia
  dir_results=/shared/results/$USER/videoGAN${add_prox}
else
  echo "Singularity image does not exist!"
  exit 1
  singularity_images=$HOME/singularity_images/miniconda_pytorch_py39.sif
  datasets=/local/datasets
  dir_results=/local/results/partial_label_learning/katsura_CLL${add_prox}
  export CUDA_VISIBLE_DEVICES=$1
fi


#port=8113
port=8115



mkdir -p $dir_results

singularity exec "$singularity_images" python -m visdom.server -port $port &
#singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u train.py PGAN -c config_jelita.json --restart -n jelita --dir /results --visdom_port $port
singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u train.py PGAN -c config_jelita.json -n jelita --dir /results --visdom_port $port --save_iter 16000

#singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u train.py PGAN -c config_cifar10.json --restart -n cifar10 --dir /results --visdom_port $port

echo -e "\033[0;1;32mDone - bash script\033[0m"

exit 0




################################################################

dir_results=/shared/results/$USER/videoGAN/AC-ProGAN_2022-09-13

name=cifar10

#singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u eval.py visualization -n $name -m PGAN --showLabels --dir /results --no_vis


singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u eval.py visualization -n $name -m PGAN --dir /results --save_dataset /results/eval --size_dataset 100 --label 0 --np_vis







## 1. Pobierz i rozpakuj: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#
#path2dri="../datasets/"
#mkdir -p ${path2dri}
#wget -P ${path2dri} http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar -zxvf ${path2dri}/cifar-10-python.tar.gz -C ${path2dri} && rm ${path2dri}/cifar-10-python.tar.gz
#
## 2. Run:
#
#python datasets.py cifar10 ${path2dri}/cifar-10-batches-py -o ${path2dri}/cifar-10 && rm -r ${path2dri}/cifar-10-batches-py
#
##python -m visdom.server
##python train.py PGAN -c config_cifar10.json --restart -n cifar10



