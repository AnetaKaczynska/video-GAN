#!/usr/bin/env bash

#SBATCH --job-name=videoGAN
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --qos=normal  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgx  # rtx2080 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"


current_date=$( date +"%Y-%m-%d_%H%M%S" )
add_prox="/AC-ProGAN_${current_date}"
#add_prox=""

singularity_images=""
if [[ -f /shared/sets/singularity/miniconda_pytorch_py39.sif ]]; then
  singularity_images=/shared/sets/singularity/miniconda_pytorch_py39.sif
#  datasets=/shared/results/Skopia/images_split2classes
  datasets=/shared/results/Skopia
#  dir_results=/shared/results/$USER/videoGAN${add_prox}
else
  echo "Singularity image does not exist!"
  exit 1
  singularity_images=$HOME/singularity_images/miniconda_pytorch_py39.sif
  datasets=/local/datasets
#  dir_results=/local/results/partial_label_learning/katsura_CLL${add_prox}
  export CUDA_VISIBLE_DEVICES=$1
fi


#dir_results=/shared/results/$USER/videoGAN/AC-ProGAN_2022-10-14_141007
dir_results=/shared/results/$USER/videoGAN/AC-ProGAN_2022-10-28_122557
name=jelita

#singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u eval.py visualization -n $name -m PGAN --showLabels --dir /results --no_vis



mkdir -p "${dir_results}/EVALS"
for label in 0 1; do
singularity exec --nv -B $datasets:/datasets -B "${dir_results}":/results "$singularity_images" python -u eval.py visualization -n $name -m PGAN --dir /results --save_dataset /results/EVALS/eval_${label} --size_dataset 500 --label ${label} --np_vis --iter 48000
done

#for it in 16000 32000 48000 56000 64000 72000 80000 88000 96000 104000 112000 120000 128000 136000 144000 152000 160000 168000 176000 184000 192000 200000; do
#  singularity exec --nv -B $datasets:/datasets -B $dir_results:/results "$singularity_images" python -u eval.py visualization -n $name -m PGAN --dir /results --save_dataset /results/EVALS/eval_${it} --size_dataset 250 --label 1 --np_vis --iter $it
#done

echo -e "\033[0;1;32mDone - bash script\033[0m"

exit 0




################################################################

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



