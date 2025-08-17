import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def load_burgers_data_deeponet(NMESH:str, DIFF_QUERY:bool=False):
    """Dataloader for the 1D burgers equation data in deep-o-net format
    
    Args:
        ngrid (int): number of grid points for discretized initial condition input
        nquery (int): number of query locations for (un)aligned dataset
    """
    mesh_path = f"mesh_{NMESH}.npy"
    tmesh_path = f"tmesh_{NMESH}.npy"
    solution_trajs_path = f"solution_trajectories_{NMESH}.npy"
    print("Data paths:", mesh_path, tmesh_path, solution_trajs_path)
    mesh = jnp.load(mesh_path)
    tmesh = jnp.load(tmesh_path)
    solution_trajs = jnp.load(solution_trajs_path)

    numtrajs, trajsteps, trajpoints = solution_trajs.shape

    minmaxscale = lambda x, min, max: (x - min) / (max - min)

    absmax = jnp.abs(solution_trajs).max()

    scaled_solutions = solution_trajs / absmax
    # scaled_solutions = minmaxscale(solution_trajs, solution_trajs.min(), solution_trajs.max())

    # breakpoint()
    assert scaled_solutions.max().item() <= jnp.pi and scaled_solutions.min().item() >= -jnp.pi
    #assert scaled_solutions.max().item() <= 1 and scaled_solutions.min().item() >= -1

    tstep = 0.5
    dt = tmesh[1]
    jump = int(tstep / dt)

    t_interests = [0, 0.5]
    t_interests = [int(x/dt) for x in t_interests]
    CHOP = int(numtrajs * 0.6)
    branch_inputs_train = []
    branch_inputs_test = []
    trunk_inputs_train = mesh.reshape(-1,1)
    trunk_inputs_test = mesh.reshape(-1,1)
    outputs_train = []
    outputs_test = []
    for t_interest in t_interests:
        branch_inputs_train.append(scaled_solutions[:CHOP, t_interest])
        #trunk_inputs_train = mesh.reshape(-1,1)
        outputs_train.append(scaled_solutions[:CHOP, t_interest + jump])

        branch_inputs_test.append(scaled_solutions[CHOP:, t_interest])
        #trunk_inputs_test = mesh.reshape(-1,1)
        outputs_test.append(scaled_solutions[CHOP:, t_interest + jump])
    branch_inputs_train = jnp.vstack(branch_inputs_train)
    branch_inputs_test = jnp.vstack(branch_inputs_test)
    outputs_train = jnp.vstack(outputs_train)
    outputs_test = jnp.vstack(outputs_test)

    DOWNSAMPLE = 10
    print("DOWNSAMPLE has been hardcoded to 10. Make sure your dataset has been generated on your desired mesh.")
    #if ngrid > 0:
    #    while branch_inputs_train[:, ::DOWNSAMPLE].shape[1] > ngrid:
    #        DOWNSAMPLE += 1
    #    print(f"{DOWNSAMPLE=} for {branch_inputs_train[:, ::DOWNSAMPLE].shape[1]} grid points")
    
    if DIFF_QUERY:
        DOWNSAMPLE_QUERY = 5
    else:
        DOWNSAMPLE_QUERY = DOWNSAMPLE
    #if nquery > 0:
    #    while trunk_inputs_test[::DOWNSAMPLE_QUERY].shape[0] > nquery:
    #        DOWNSAMPLE_QUERY += 1

    branch_inputs_train_dwn = branch_inputs_train[:, ::DOWNSAMPLE]
    branch_inputs_test_dwn = branch_inputs_test[:, ::DOWNSAMPLE]
    trunk_inputs_train_dwn = trunk_inputs_train[::DOWNSAMPLE]
    trunk_inputs_test_dwn = trunk_inputs_test[::DOWNSAMPLE]
    trunk_inputs_branch = trunk_inputs_train[::DOWNSAMPLE]
    trunk_inputs_query = trunk_inputs_train[::DOWNSAMPLE_QUERY]
    outputs_train_dwn = outputs_train[:, ::DOWNSAMPLE_QUERY]
    outputs_test_dwn = outputs_test[:, ::DOWNSAMPLE_QUERY]
    # breakpoint()

    # return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn
    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_branch, trunk_inputs_query, outputs_train_dwn, outputs_test_dwn


def plot_trajectories(u, dt:float=0.01, mesh=None, t_interval:float=1.0, figuresize=(9,3)):
    plt.figure(figsize=figuresize)
    if mesh is None:
        plt.plot(u[0, :], label="initial")
    else:
        plt.plot(mesh, u[0, :], label="initial")
    walk = int(t_interval/dt)
    idx = walk
    while idx < u.shape[0]:
        if mesh is None:
            plt.plot(u[idx, :], label=f"{idx * dt:.2f} seconds")
        else:
            plt.plot(mesh, u[idx, :], label=f"{idx * dt:.2f} seconds")
        idx += walk
    plt.title(f"Solution Trajectories")
    plt.xlabel("Spatial Location")
    plt.ylabel("Value")
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2))

if __name__ == "__main__":
    (branch_inputs_train_dwn,
     branch_inputs_test_dwn,
     trunk_inputs_train_dwn,
     trunk_inputs_test_dwn,
     outputs_train_dwn,
     outputs_test_dwn) = load_burgers_data_deeponet(NMESH=101)
    breakpoint()

