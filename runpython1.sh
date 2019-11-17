#!/bin/csh
#PBS -N nike_numax
#PBS -o output_89.txt
#PBS -j oe
#PBS -q physics
#PBS -l nodes=node43:ppn=12
#PBS -l mem=20GB
#PBS -l walltime=20:00:00
#PBS -m ea
#PBS -M yali4742@uni.sydney.edu.au
#PBS -V

date
hostname
module load Anaconda3-5.1.0
python3 "/headnode2/yali4742/nike/lib/8-mr-mass.py"
python3 "/headnode2/yali4742/nike/lib/9-mr-radius.py"
date
exit
