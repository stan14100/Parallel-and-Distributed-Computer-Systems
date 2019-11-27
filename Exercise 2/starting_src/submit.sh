#PBS -N game-of-life
#PBS -q pdlab
#PBS -j oe
#PBS -l nodes=1:ppn=8

module load mpi/mpich3-x86_64

cd $PBS_O_WORKDIR

echo "==== Run starts now ======= `date` "

mpiexec -ppn 1 -n $PBS_NUM_NODES ./game-of-life &> $PBS_JOBNAME.log

echo "==== Run ends now ======= `date` "
