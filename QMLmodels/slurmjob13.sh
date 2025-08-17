#!/bin/bash
#SBATCH --job-name=PQOC13
#SBATCH --output=PQOC13_OUT.txt
#SBATCH --error=PQOC13_ERR.txt
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=50G
#SBATCH --time=23:00:00
#SBATCH --mail-user=jow243@pitt.edu    
#SBATCH --mail-type=END,FAIL

echo "JOB START"
echo "$(date)"

echo "WORKING DIRECTORY"
echo "$(pwd)"

module load gcc/8.2.0
module load python/ondemand-jupyter-python3.11
echo "Modules loaded."

# source activate /ihome/pgivi/jow243/.conda/envs/qml_env
source activate /ihome/pgivi/jow243/.conda/envs/mlenv
echo "Python env loaded."
echo "$(which python)"

# cd /ihome/pgivi/jow243/code/qml4pde/QMLmodels

echo "Calling training script..."
python -u train_pqoc_burgers.py 13

echo "JOB END"
echo "$(date)"
