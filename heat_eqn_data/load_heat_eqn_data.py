import jax.numpy as jnp

def load_heat_eqn_data_deeponet(ngrid:int=-1):
    # load data
    mesh = jnp.load("heat_eqn_data/mesh.npy")
    tmesh = jnp.load("heat_eqn_data/tmesh.npy")
    solution_trajs = jnp.load("heat_eqn_data/solution_trajectories.npy")
    # X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  <- then x2 - 1 to scale between 0 and 1
    # X = X_scaled * (max - min) + min
    SCALE_MAX = 4
    SCALE_MIN = -4
    scaled_solutions = (solution_trajs - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 2 - 1
    #scaled_solutions.max().item(), scaled_solutions.min().item()

    dt = tmesh[1]
    tstep = 2
    jump = int(tstep / dt)

    t_interests = [0, 1, 2, 3]
    t_interests = [int(x/dt) for x in t_interests]
    branch_inputs_train = []
    branch_inputs_test = []
    trunk_inputs_train = mesh.reshape(-1,1)
    trunk_inputs_test = mesh.reshape(-1,1)
    outputs_train = []
    outputs_test = []
    for t_interest in t_interests:
        branch_inputs_train.append(scaled_solutions[:100, t_interest])
        #trunk_inputs_train = mesh.reshape(-1,1)
        outputs_train.append(scaled_solutions[:100, t_interest + jump])

        branch_inputs_test.append(scaled_solutions[100:, t_interest])
        #trunk_inputs_test = mesh.reshape(-1,1)
        outputs_test.append(scaled_solutions[100:, t_interest + jump])
    branch_inputs_train = jnp.vstack(branch_inputs_train)
    branch_inputs_test = jnp.vstack(branch_inputs_test)
    outputs_train = jnp.vstack(outputs_train)
    outputs_test = jnp.vstack(outputs_test)

    if ngrid > 0:
        DOWNSAMPLE = 1
        while branch_inputs_train[:, ::DOWNSAMPLE].shape[1] > ngrid:
            DOWNSAMPLE += 1
        print(f"{DOWNSAMPLE=} for {branch_inputs_train[:, ::DOWNSAMPLE].shape[1]} grid points")
    else:
        DOWNSAMPLE = 1
    
    branch_inputs_train_dwn = branch_inputs_train[:, ::DOWNSAMPLE]
    branch_inputs_test_dwn = branch_inputs_test[:, ::DOWNSAMPLE]
    trunk_inputs_train_dwn = trunk_inputs_train[::DOWNSAMPLE]
    trunk_inputs_test_dwn = trunk_inputs_test[::DOWNSAMPLE]
    outputs_train_dwn = outputs_train[:, ::DOWNSAMPLE]
    outputs_test_dwn = outputs_test[:, ::DOWNSAMPLE]

    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn
