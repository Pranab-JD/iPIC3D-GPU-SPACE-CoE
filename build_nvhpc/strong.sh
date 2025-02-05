#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
###SBATCH --exclusive
#SBATCH -A IscrC_ipic3D
#SBATCH --time=00:03:00
#SBATCH --job-name=ipic
#SBATCH -o test.out

echo "Loading modules"
# module load nvhpc/24.3 cuda/12.3 openmpi/4.1.6--nvhpc--24.3 hdf5/1.14.3--openmpi--4.1.6--nvhpc--24.3
ml cmake/3.27.7 openmpi/4.1.6--gcc--12.2.0 cuda/12.3


# echo "--------------------------------------------------"
# echo "nodes: $SLURM_NODELIST                            "
# echo "total nodes: $SLURM_NNODES                        "
# echo "tasks per node: $SLURM_TASKS_PER_NODE             "
# echo "cpus per task: $SLURM_CPUS_PER_TASK               "
# echo "gpus per node: $SLURM_GPUS_PER_NODE               "
# echo "procid: $SLURM_PROCID                             "
# echo "--------------------------------------------------"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}

echo "[LOG] local rank $LOCAL_RANK: bind to $CUDA_VISIBLE_DEVICES"
echo ""

date

# IPIGPU=/leonardo_work/SPACE_bench/iPic3D_apps/iPIC3D-GPU-SPACE-CoE/build_nvhpc

start_time="$(date -u +%s.%N)"
# mpirun -np 4 --map-by ppr:4:node:PE=8 --report-bindings ./iPIC3D strong_1_new.inp

srun ./iPIC3D strong_1_new.inp

end_time="$(date -u +%s.%N)"
elapsed="$(bc <<<"$end_time-$start_time")"
echo "Total of $elapsed seconds elapsed for process"

date