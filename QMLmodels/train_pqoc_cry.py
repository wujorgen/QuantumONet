import pennylane as qml
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os
import sys
import json
from copy import deepcopy
from tqdm import tqdm, trange

from deeponet_dataloader import deeponet_dataloader
from ansatz import SU1CRYAnsatz, SU2CRYAnsatz
from feature_map import AngleVectorEmbedding, SinusoidalPositionEmbedding, RZZ_FeatureMap, ChebyshevFeatureMap


# jax.config.update("jax_enable_x64", True)  # this enables jax to use complex128

NQUBITS = 12  # 12, 18, or 24
dev = qml.device('default.qubit', wires=NQUBITS)


@qml.qnode(dev, interface='jax', diff_method="best")
def branch_circuit(params, inputs):
    """Parametrized Quantum Circuit for Branch Network
    Args:
        params (array):
        inputs (array):
    """
    # embedding / feature map for inputs
    AngleVectorEmbedding(inputs, nqubits=NQUBITS, rotation="Y", offset_and_invert=True)
    #RZZ_FeatureMap(inputs, nqubits=NQUBITS, mode="circular")
    # evaluate parametrized ansatz
    #SU1CRYAnsatz(params["theta"], params["cry"], repeats=5, nqubits=NQUBITS, entanglement_structure="circular")
    SU2CRYAnsatz(params["theta"], params["cry"], repeats=4, nqubits=NQUBITS, entanglement_structure="circular")
    #SU1Ansatz(params, repeats=3, nqubits=NQUBITS, entanglement_structure="circular", entanglement_neighbors=1, entanglement_gate="X")
    # return expectation value of Pauli operator on qubits
    return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]


@qml.qnode(dev, interface='jax', diff_method="best")
def trunk_circuit(params, input):
    """Parametrized Quantum Circuit for Trunk Network
    Args:
        params (array):
        input (float):
    """
    # single input embedding that increases complexity
    #SinusoidalPositionEmbedding(input, nqubits=NQUBITS)
    ChebyshevFeatureMap(input, nqubits=NQUBITS)
    # evaluate parametrized ansatz
    #SU1Ansatz(params, repeats=4, nqubits=NQUBITS, entanglement_structure="full", entanglement_neighbors=1, entanglement_gate="X")
    SU1CRYAnsatz(params["theta"], params["cry"], repeats=3, nqubits=NQUBITS, entanglement_structure="linear")
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


def import_burgers_data(nqubits, nquery):
    rootdir = os.getcwd()
    os.chdir("../burgers_data")
    sys.path.append(os.getcwd())
    from load_burgers_data import load_burgers_data_deeponet
    branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = load_burgers_data_deeponet(ngrid=nqubits, nquery=nquery)
    os.chdir(rootdir)
    return branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test


def train_model(mode:str):
    print("==> NQUBITS has been set to", NQUBITS)
    match mode.lower():
        case "antiderivative":
            branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_antiderivative_data(NQUBITS)
        case "burgers":
            branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_burgers_data(NQUBITS, nquery=NQUBITS*2)
        case _:
            sys.exit(1)
    resultdir = f"{mode.lower()}_results"
    paramfile = f"{resultdir}/params.json"
    if not os.path.isdir(resultdir):
        os.mkdir(resultdir)

    number_of_branch_params = 2 * NQUBITS * (4 + 1)
    number_of_branch_cry = 4 * (NQUBITS - 0)  #repeats * (nqubits - int(entanglement_structure!="circular"))
    number_of_trunk_params = 1 * NQUBITS * (3 + 1)
    number_of_trunk_cry = 3 * (NQUBITS - 1)
    print(f"{number_of_branch_params=}, {number_of_branch_cry=}, {number_of_trunk_params=}, {number_of_trunk_cry=}")
    params = {
        "branch": {
            "theta": jnp.zeros(number_of_branch_params, dtype=jnp.float32),
            "cry": jnp.zeros(number_of_branch_cry, dtype=jnp.float32) * jnp.pi,
        },
        "trunk": {
            "theta": jnp.zeros(number_of_trunk_params, dtype=jnp.float32),
            "cry": jnp.zeros(number_of_trunk_cry, dtype=jnp.float32) * jnp.pi,
        },
        "bias": jnp.array(0, dtype=jnp.float32),
    }

    RESTART = False
    if RESTART:
        loaded_params = json.load(open(paramfile, "r"))
        for key, item in loaded_params.items():
            if type(item) == dict:
                tmp = {}
                for subkey, subitem in loaded_params[key].items():
                    tmp[subkey] = jnp.array(subitem, dtype=jnp.float32)
                params[key] = deepcopy(tmp)
            else:
                params[key] = jnp.array(item, dtype=jnp.float32)
        # fix padding in case warm restarts are used, where a smaller ansatz is trained first before more layers are added
        # note that this will only work if the number of parameters read from the json file is smaller than the number specified
        params["branch"]["theta"] = jnp.pad(params["branch"]["theta"], (0, number_of_branch_params - params["branch"]["theta"].shape[0]))
        params["branch"]["cry"] = jnp.pad(params["branch"]["cry"], (0, number_of_branch_cry - params["branch"]["cry"].shape[0]))
        params["trunk"]["theta"] = jnp.pad(params["trunk"]["theta"], (0, number_of_trunk_params - params["trunk"]["theta"].shape[0]))
        params["trunk"]["cry"] = jnp.pad(params["trunk"]["cry"], (0, number_of_trunk_cry - params["trunk"]["cry"].shape[0]))
    breakpoint()

    opt = optax.adam(learning_rate=5e-3)
    opt_state = opt.init(params)

    loss_history = []
    for epoch in tqdm(range(150), position=0, desc="Epoch", ncols=80):
        loss_history.append([])
        if len(loss_history) <= 2:
            last_loss = "Last Epoch Loss: n/a"
        else:
            last_loss = f"Last Epoch Loss: {sum(loss_history[epoch-1])/len(loss_history[epoch-1]):.5f}"
        for branch_input, trunk_input, target in tqdm(deeponet_dataloader(branch_inputs_train, trunk_inputs_query, outputs_train, batchsize=16, epoch=epoch), position=1, desc=last_loss, ncols=80, leave=False):
            params, opt_state, loss_val = update_step(branch_input, trunk_input, target, params, opt, opt_state)
            loss_history[epoch].append(loss_val)

    epoch_avg_loss = [sum(arr)/len(arr) for arr in loss_history]
    print("Last Epoch Loss:", epoch_avg_loss[-1])

    # json file can not handle non native python types?
    converted_params = dict.fromkeys(params.keys())
    for key, item in params.items():
        if type(item) == dict:
            tmp = {}
            for subkey, subitem in params[key].items():
                tmp[subkey] = subitem.tolist()
            converted_params[key] = deepcopy(tmp)
        else:
            converted_params[key] = item.tolist()
    json.dump(converted_params, open(paramfile, "w"))

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
    viz(67)
    viz(99)
    viz(117)

    plt.figure()
    qml.draw_mpl(branch_circuit, style="pennylane", decimals=2)(params["branch"], branch_inputs_test[0,:])
    plt.savefig(f"{resultdir}/branch_circuit.png")
    plt.close()

    plt.figure()
    qml.draw_mpl(trunk_circuit, style="pennylane", decimals=2)(params["trunk"], trunk_inputs_query[0])
    plt.savefig(f"{resultdir}/trunk_circuit.png")
    plt.close()

    breakpoint()


if __name__ == "__main__":
    train_model(mode=["antiderivative", "burgers"][1])
