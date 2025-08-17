# QONet models
The training scripts in this folder are as follows:
```shell
# antiderivative
$ python train_pqoc_antiderivative.py
# heat eqn
$ python train_pqoc_heat.py
# burgers' eqn
$ python train_pqoc_burgers.py N  # where N in [9, 11, 13, 15]
```

These scripts can also be sent to a slurm cluster using the following:
```shell
# antiderivative
$ sbatch slurmantiderivative.sh
# heat eqn
$ sbatch slurmheat.sh
# burgers' eqn: choose one/any
$ sbatch slurmjob9.sh
$ sbatch slurmjob11.sh
$ sbatch slurmjob13.sh
$ sbatch slurmjob15.sh
```