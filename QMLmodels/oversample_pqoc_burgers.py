print("importing python packages")
import pennylane as qml
import jax
import jax.numpy as jnp
import jax.random as jr
# import numpy as np
# import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os
import sys
import sys
import json

from copy import deepcopy
from tqdm import tqdm, trange

from deeponet_dataloader import deeponet_dataloader
from ansatz import SU1Ansatz, SU2Ansatz, SU3Ansatz, SU1CRYAnsatz, CyclicAnsatz, RXXAnsatz
from feature_map import AngleVectorEmbedding, SinusoidalPositionEmbedding, CosineScalarEmbedding, RZZ_FeatureMap, RZZ_Layer, ChebyshevFeatureMap

# jax.config.update('jax_enable_x64', True)

if len(sys.argv) == 2:
    NQUBITS = int(sys.argv[1])
    print("Input detected for number of grid points.")
else:
    print("No input for number of grid points detected. Defaulting...")
    NQUBITS = 11  # 9, 11, 13, 15
print("==> NQUBITS has been set to", NQUBITS)
dev = qml.device("default.qubit", wires=NQUBITS)

diff_method = ["best", "adjoint", "backprop", "finite-diff"][2]
print("Diff method selected:", diff_method)
# adjoint is slow. specify and lock to backprop

@qml.qnode(dev, interface='jax', diff_method=diff_method)
def branch_circuit(params, inputs):
    """Parametrized Quantum Circuit for Branch Network
    Args:
        params (array):
        inputs (array):
    """
    # embedding / feature map for inputs
    AngleVectorEmbedding(inputs, nqubits=NQUBITS, rotation="Y", offset_and_invert=True)
    RZZ_Layer(inputs, nqubits=NQUBITS, mode="linear")
    qml.Barrier()
    # evaluate parametrized ansatz
    SU3Ansatz(params, wires=range(NQUBITS), entanglement_structure="linear", entanglement_gate="Z")
    # return expectation value of Pauli operator on qubits
    return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]


@qml.qnode(dev, interface='jax', diff_method=diff_method)
def trunk_circuit(params, input):
    """Parametrized Quantum Circuit for Trunk Network
    Args:
        params (array):
        input (float):
    """
    # single input embedding that increases complexity
    #ChebyshevFeatureMap(input, nqubits=NQUBITS)
    CosineScalarEmbedding(input, nqubits=NQUBITS)
    qml.Barrier()
    # evaluate parametrized ansatz
    SU1Ansatz(params, wires=range(NQUBITS), entanglement_structure="linear", entanglement_gate="Z")
    # return expectation value of Pauli operator on qubits
    return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]


def PQOC(params, branch_input, trunk_input):
    """Parametrized Quantum Operator Circuit
    Args:
        params (dict): fields are "branch" "trunk" "bias"
        branch_input (array):
        trunk_input (array):
    """
    # call branch circuit
    branch_output = branch_circuit(params["branch"], branch_input)
    # call trunk circuit
    trunk_output = trunk_circuit(params["trunk"], trunk_input)
    # compute and return inner product
    return jnp.dot(jnp.array(branch_output).ravel(), jnp.array(trunk_output).ravel()) + params["bias"]  # ravel seems janky but prevents vmap from destroying trunk output dimension


def normalized_l2_error(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm


def rmse_error(pred, ref):
    mse = jnp.mean((pred - ref)**2)
    return jnp.sqrt(mse)


@jax.jit
def loss_fn(params, branch_in, trunk_in, y):
    y_hat = jax.vmap(
        jax.vmap(
            PQOC, in_axes=(None, None, 0)
            ), 
        in_axes=(None, 0, None)
    )(params, branch_in, trunk_in)
    squared_errors = (y_hat - y) ** 2
    sse = jnp.sum(squared_errors)
    rmse = jnp.sqrt(jnp.mean(squared_errors))
    return rmse


def update_step(branch_input, trunk_input, target, params, opt, opt_state):
    """Update step
    Args:
        branch_input ():
        trunk_input ():
        target ():
        params (dict[jax.numpy.ndarray]):
        opt (optax function):
        opt_state (optax state):
    """
    loss_val, grads = jax.value_and_grad(loss_fn)(params, branch_input, trunk_input, target)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


def import_antiderivative_data(nqubits):
    rootdir = os.getcwd()
    os.chdir("../antiderivative_data")
    sys.path.append(os.getcwd())
    from load_antiderivative_data import load_antiderivative_data_deeponet
    branch_inputs_train, branch_inputs_test, trunk_inputs_train, trunk_inputs_test, outputs_train, outputs_test = load_antiderivative_data_deeponet(ngrid=nqubits)
    os.chdir(rootdir)
    return branch_inputs_train, branch_inputs_test, trunk_inputs_train, trunk_inputs_test, outputs_train, outputs_test


def import_heat_eqn_data(nqubits):
    rootdir = os.getcwd()
    os.chdir("../heat_eqn_data")
    sys.path.append(os.getcwd())
    from load_heat_eqn_data import load_heat_eqn_data_deeponet
    branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn = load_heat_eqn_data_deeponet(ngrid=nqubits)
    os.chdir(rootdir)
    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn


def import_burgers_data(nqubits):
    MM = (NQUBITS - 1) * 10 + 1
    rootdir = os.getcwd()
    os.chdir("../burgers_data")
    sys.path.append(os.getcwd())
    from load_burgers_data import load_burgers_data_deeponet
    branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = load_burgers_data_deeponet(NMESH=MM, DIFF_QUERY=True)
    os.chdir(rootdir)
    return branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test


def train_model():
    print("==> NQUBITS has been set to", NQUBITS)
    branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_burgers_data(NQUBITS)

    print("Training dataset: burgers")

    # set output file locations
    resultdir = f"burgers_{NQUBITS}_results"
    paramfile = f"{resultdir}/params.json"
    if not os.path.isdir(resultdir):
        os.mkdir(resultdir)

    # initialize parameters
    shape_of_branch_params = (NQUBITS, NQUBITS, 3)  # SU(3)
    shape_of_trunk_params = (NQUBITS, NQUBITS)  # SU(1)
    print(f"{shape_of_branch_params=}, {shape_of_trunk_params=}")
    ZERO_INIT = False
    if ZERO_INIT:
        params = {
            "branch": jnp.zeros(shape_of_branch_params, dtype=jnp.float32),
            "trunk": jnp.zeros(shape_of_trunk_params, dtype=jnp.float32),
            "bias": jnp.array(0, dtype=jnp.float32),
        }
    else:
        params = {
            "branch": jr.uniform(jr.PRNGKey(0), shape=shape_of_branch_params, dtype=jnp.float32),
            "trunk": jr.uniform(jr.PRNGKey(1), shape=shape_of_trunk_params, dtype=jnp.float32),
            "bias": jnp.array(0, dtype=jnp.float32),
        }

    RESTART = False
    if RESTART:
        loaded_params = json.load(open(paramfile, "r"))
        for key, item in loaded_params.items():
            params[key] = jnp.array(item, dtype=jnp.float32)

    def save_params():
        # json file can not handle non native python types?
        converted_params = dict.fromkeys(params.keys())
        for key in params.keys():
            converted_params[key] = params[key].tolist()
        json.dump(converted_params, open(paramfile, "w"))
    
    def save_history(history, fname):
        with open(fname, "w") as f:
            f.write("mean, std\n")
            for epoch in history:
                tmp = []
                for lss in epoch:
                    tmp.append(jnp.linalg.norm(lss))
                f.write(f"{jnp.mean(jnp.array(tmp))}, {jnp.std(jnp.array(tmp))}\n")

    opt = optax.adam(learning_rate=1e-3)
    opt_state = opt.init(params)

    loss_history = []
    grad_history = {
        "branch": [],
        "trunk": [],
        "bias": []
    }

    print("Begin training loop...")
    for epoch in range(1500):  # 1500
        loss_history.append([])
        for key in grad_history.keys():
            grad_history[key].append([])

        for branch_input, trunk_input, target in deeponet_dataloader(branch_inputs_train, trunk_inputs_query, outputs_train, batchsize=32, epoch=epoch):
            # params, opt_state, loss_val = update_step(branch_input, trunk_input, target, params, opt, opt_state)
            loss_val, grads = jax.value_and_grad(loss_fn)(params, branch_input, trunk_input, target)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            loss_history[epoch].append(loss_val)
            for key in grad_history.keys():
                grad_history[key][epoch].append(grads[key])
        # breakpoint()
        
        last_epoch_avg_loss = sum(loss_history[-1]) / len(loss_history[-1])

        if epoch % 10 == 0:
            print(f"End of epoch {epoch}, avg loss: {last_epoch_avg_loss:7.5f}")
            save_params()

    epoch_avg_loss = [sum(arr)/len(arr) for arr in loss_history]
    print("Last Epoch Loss:", epoch_avg_loss[-1])

    save_params()
    save_history(loss_history, f"{resultdir}/loss_history.txt")
    for key in grad_history.keys():
        save_history(grad_history[key], f"{resultdir}/grad_{key}.txt")

    plt.figure(figsize=(10,5))
    plt.semilogy(epoch_avg_loss)
    plt.grid()
    plt.ylabel("Training Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss vs Epoch")
    plt.savefig(f"{resultdir}/training_history.png")
    plt.close()    

    predictions_test = jax.vmap(
        jax.vmap(
            PQOC, in_axes=(None, None, 0)
            ), 
        in_axes=(None, 0, None)
    )(params, branch_inputs_test, trunk_inputs_query)  

    errs = jax.vmap(rmse_error)(predictions_test, outputs_test)

    with open(f"{resultdir}/test_report.txt", "w") as f:
        f.write(f"Mean RMSE over test set: {jnp.mean(errs)}\n")
        f.write(f"Standard Deviation of RMSE over test set: {jnp.std(errs)}\n")
        f.write(f"Median RMSE over test set: {jnp.median(errs)}\n")

    def viz(point):
        plt.figure(figsize=(10,5))
        plt.plot(trunk_inputs_branch[:,0], branch_inputs_test[point, :], marker="o", label="INPUT")
        plt.plot(trunk_inputs_query[:,0], outputs_test[point, :], marker="+", label="TARGET")
        plt.plot(trunk_inputs_query[:,0], predictions_test[point, :], marker="+", label="PREDICTION")
        rmse = jnp.sqrt(jnp.mean((outputs_test[point, :] - predictions_test[point, :])**2))
        plt.title(f"Parameterized Quantum Operator Circuit, Test Datapoint {point}, RMSE {rmse:.4f}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{resultdir}/test_{point}.png")
        plt.close()

    viz(13)
    viz(32)
    viz(42)
    viz(99)
    viz(117)
    viz(197)
    viz(222)
    viz(333)

    plt.figure()
    qml.draw_mpl(branch_circuit, style="pennylane", decimals=2)(params["branch"], branch_inputs_test[0,:])
    plt.savefig(f"{resultdir}/branch_circuit.png")
    plt.close()

    plt.figure()
    qml.draw_mpl(trunk_circuit, style="pennylane", decimals=2)(params["trunk"], trunk_inputs_query[0])
    plt.savefig(f"{resultdir}/trunk_circuit.png")
    plt.close()

    # breakpoint()


if __name__ == "__main__":
    train_model()

