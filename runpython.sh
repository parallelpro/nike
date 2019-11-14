#!/bin/csh
#PBS -N nike_numax
#PBS -o lib/output.txt
#PBS -j oe
#PBS -q physics
#PBS -l nodes=1:ppn=12
#PBS -l mem=20GB
#PBS -l walltime=20:00:00
#PBS -m ea
#PBS -M yali4742@uni.sydney.edu.au
#PBS -V

# cd "$PBS_O_WORKDIR"
#your commands/programs start here, for example:
# cd "/headnode2/yali4742/nike/"
# rm "/headnode2/yali4742/nike/lib/output.txt"

date
hostname
module load Anaconda3-5.1.0
python3 "/headnode2/yali4742/nike/lib/5-1-tnu-numax-mass.py"
python3 "/headnode2/yali4742/nike/lib/5-2-tnu-numax-feh.py"
date
exit
