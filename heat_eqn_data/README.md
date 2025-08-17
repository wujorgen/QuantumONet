# Heat Equation Data for QONet
To generate this data set, run the following commands:
```shell
$ python heat_eqn.py
```

A simple finite difference scheme is used, along with a second order centered difference term.

TODO: initial condition generation should also be switch to use the PRNG capability in JAX. This isn't strictly needed but would be nice.