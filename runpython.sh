#!/bin/bsh
#PBS -N nike_numax
#PBS -o output.txt
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
# module load Anaconda3-5.1.0
# '/headnode2/yali4742/nike'
# python3 "lib/pips/5-mist.py"
python3 'lib/heb_pips/1-parsec-fdnu.py'
python3 'lib/heb_pips/2-parsec-scaling.py'
python3 'lib/heb_pips/5-mist-fdnu.py'
# python3 'lib/rgb_pips/1-parsec-fdnu.py'
# python3 'lib/rgb_pips/2-parsec-scaling.py'
# python3 'lib/rgb_pips/5-mist-fdnu.py'
date
exit
