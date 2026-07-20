#!/bin/ksh
#SBATCH --export=all
#SBATCH --job-name=test1    # create a short name for your job 
#SBATCH --ntasks=1   # cpu count 
#SBATCH --ntasks-per-node=1   # total number of tasks across all nodes  
#SBATCH --cpus-per-task=1    
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH -p kshcnormal
#SBATCH -o ./out_main.txt
export I_MPI_HYDRA_TOPOLIB=ipl 
srun --exclusive  python ./c2.4_Model_run_functions_sensitivity.py
