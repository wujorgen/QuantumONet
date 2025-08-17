# Burgers' Equation Data for QONet
To generate this data set, run one of the following commands:
```shell
# call the data generation shell script
$ ./generate_data.sh
# if you are on a slurm cluster:
$ sbatch generate_data.slurm
```

## burgers_eqn.py
The file ```burgers_eqn.py``` contains the logic for generating initial conditions and calculating transient solutions to the 1D Burgers' Equation. Seeding for initial conditions is done using the Pseudo-Random Number Generator functionality within JAX, to ensure reproducibility of results. Additionally, the usage of the PRNG allows for the solution of the same initial conditions over multiple datasets (which is needed due to downsampling - the number of grid points must be changed to ensure that the periodic boundary condition is preserved.)

A finite difference scheme is used. The second order (diffusion) term is discretized with a second order centered difference, while the first order (advection) term is discretized using a first order upwinding scheme. (Note that this does introduce some level of artificial diffusion, which is a numerical phenomemon. This does vary with grid size, but since all grids are relatively similar, the effect is barely noticeable.)

The usage is as follows, where the argument is the number of grid points.
```shell
$ python burgers_eqn.py 101
```