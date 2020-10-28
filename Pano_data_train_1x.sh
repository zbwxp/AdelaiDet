#!/bin/bash
#SBATCH -M volta
#SBATCH -p batch
#SBATCH -A aiml
#SBATCH -n 1 # number of cores (here 2 cores requested)
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --mem=300GB # memory pool for all cores (here set to 8 GB)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=b.zhang@adelaide.edu.au


module load CUDA/9.0.176
module load CUDA/9.2.148.1
module load cuDNN/7.0-CUDA-9.0.176
module load cuDNN/7.3.1-CUDA-9.2.148.1
module load ncurses/6.0-GCCcore-5.4.0
module load libyaml/0.1.6-foss-2016uofa



python tools/train_net.py --config-file configs/CondInst/Panoptic/R_50_1x.yaml  --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 ))  --num-gpus 4   OUTPUT_DIR training_dir/Cond_Pano_instance_from_pano_R_50_1x_sem_loss_on 

