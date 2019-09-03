#!/bin/bash -l

#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --exclude=seskscpg020

conda activate reinvent-lite
trap "conda deactivate" EXIT

python /projects/mai/kpsg824/wlgan/encode.py \
-sf data/moses.smi \
-o data/moses.latent \
$@
