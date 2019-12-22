# !/bin/bsh
# PBS -N nike_numax
# PBS -o output.txt
# PBS -j oe
# PBS -q physics
# PBS -l nodes=node45:ppn=12
# PBS -l mem=20GB
# PBS -l walltime=20:00:00
# PBS -m ea
# PBS -M yali4742@uni.sydney.edu.au
# PBS -V

date
hostname
# module load Anaconda3-5.1.0
# '/headnode2/yali4742/nike'
python 'lib/5-tnu-numax.py'
python 'lib/6-tnu-dnu.py'
# python "/headnode2/yali4742/nike/lib/7-mr-mass.py"
# python "/headnode2/yali4742/nike/lib/8-mr-radius.py"
date
exit
