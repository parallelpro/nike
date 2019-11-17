#!/bin/csh
#PBS -N nike_numax
#PBS -o output_56.txt
#PBS -j oe
#PBS -q physics
#PBS -l nodes=node45:ppn=12
#PBS -l mem=20GB
#PBS -l walltime=20:00:00
#PBS -m ea
#PBS -M yali4742@uni.sydney.edu.au
#PBS -V

date
hostname
module load Anaconda3-5.1.0
python3 "/headnode2/yali4742/nike/lib/5-tnu-numax.py"
python3 "/headnode2/yali4742/nike/lib/6-tnu-dnu.py"
date
exit
