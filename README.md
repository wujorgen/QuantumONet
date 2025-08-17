# Quantum O Net

This repository contains the code used to perform the experiments in: TODO: link [paper](/) once completed, along with abstract or contributions.

## Background
The Quantum Operator Network (Quantum O Net, or QONet) is a hybrid classical-quantum machine learning architecture which extends the classical DeepONet proposed in [CITATION].

# Setup
Using either conda or pip, install the required packages.
```shell
# conda
$ conda env create -f mlenv.yml
# pip
$ python -m venv venv
$ source ./venv/bin/activate
$ pip install -r Requirements.txt
```

# Generating Training Data
For specifics on generating training data, please see the ```README.md``` in each of the following data folders:
```shell
antiderivative_data/
burgers_data/
heat_eqn_data/
```

# Training the Quantum O Net
First, change directory into the ```QMLmodels``` directory.
```shell
$ cd QMLmodels
```
From there, call the appropriate training script. Each script can be called directly, or the job can be submitted to a slurm cluster. The slurm submission scripts can be changed as needed. 

(If you have a Pitt email, please change the email field in the submission scripts so I don't get emailed when your job finishes.)

## Anti-Derivative
$\frac{dy}{dx} = u(x), x\in [0,1]$

$y(0)=0$

```shell
# script
$ python train_pqoc_antiderivative.py
# slurm
$ sbatch slurmantiderivative.sh
```

## Heat Equation
$\frac{\partial u(x,t)}{\partial t} = \frac{\partial^2 u(x,t)}{\partial t^2}$

$u(-1, t)=u(-1, 0)$

$u(1, t)=u(1, 0)$

```shell
# script
$ python train_pqoc_heat.py
# slurm
$ sbatch slurmheat.sh
```

## Burgers' Equation
$\frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x}= \nu \frac{\partial^2 u(x,t)}{\partial t^2}$

$u(0,t)=u(1,t)$

There are four different levels of resolution used for the Burgers' Equation experiments. This was done because the solution operator being modelled is quite complex, and while it is trivial to make a large enough neural network to capture this, the resouces consumed by a simulation are very non-trivial. Muliple grid resolutions allows for the investigation of how accuracy improves as input space (and model/circuit size) increases.
```shell
# script: note that N can depend on the number of grid points you used to generate your data.
$ python train_pqoc_burgers.py N  # where N in [9, 11, 13, 15]
# slurm: choose one of the following
$ sbatch slurmjob9.sh
$ sbatch slurmjob11.sh
$ sbatch slurmjob13.sh
$ sbatch slurmjob15.sh
```