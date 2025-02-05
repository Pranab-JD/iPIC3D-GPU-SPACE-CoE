#!/bin/bash
#SBATCH --account=IscrC_ipic3D
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --job-name=ipic
#SBATCH -o strong_1.out

echo "Loading modules"
module load nvhpc/24.3 cuda/12.3 openmpi/4.1.6--nvhpc--24.3 hdf5/1.14.3--openmpi--4.1.6--nvhpc--24.3


echo "--------------------------------------------------"
echo "nodes: $SLURM_NODELIST                            "
echo "total nodes: $SLURM_NNODES                        "
echo "tasks per node: $SLURM_TASKS_PER_NODE             "
echo "cpus per task: $SLURM_CPUS_PER_TASK               "
echo "gpus per node: $SLURM_GPUS_PER_NODE               "
echo "procid: $SLURM_PROCID                             "
echo "--------------------------------------------------"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}

echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
echo ""

echo "========================================================================"
echo " "
date
echo " "

mpirun -np 4 --map-by ppr:4:node:PE=8 ./iPIC3D ./strong_1_new.inp

echo " "
echo "========================================================================"
echo " "
date
