#!/bin/bash
#SBATCH --job-name=GENERATE_BURGER
#SBATCH --output=GENERATE_BURGER_OUT.txt
#SBATCH --error=GENERATE_BURGER_ERR.txt
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --mail-user=jow243@pitt.edu    
#SBATCH --mail-type=END,FAIL

echo "JOB START"
echo "$(date)"

echo "WORKING DIRECTORY"
echo "$(pwd)"

module load gcc/8.2.0
module load python/ondemand-jupyter-python3.11
echo "Modules loaded."

source activate /ihome/pgivi/jow243/.conda/envs/mlenv
echo "Python env loaded."
echo "$(which python)"

# cd /ihome/pgivi/jow243/code/qml4pde/QMLmodels

echo "Calling burgers equation data script..."
python -u burgers_eqn.py 81   # 9
python -u burgers_eqn.py 101  # 11
python -u burgers_eqn.py 121  # 13
python -u burgers_eqn.py 141  # 15

echo "JOB END"
echo "$(date)"
