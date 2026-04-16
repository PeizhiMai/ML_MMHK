#!/bin/bash
#SBATCH -A cnms
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1
#SBATCH -J GUV
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cosdis123@gmail.com

cd /home/9pm/ED_qbasis/qbasis/bin/ML/HF/Gcal
export OMP_NUM_THREADS=32
time ./Square_Hubbard_tbc_G_calculation_UV.x 8 8 0 0 0 0 6 0 0
