import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
import pennylane as qml
import sys
import qiskit
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2 as GBV2

from ansatz import SU1Ansatz, SU2Ansatz, SU3Ansatz, SU1CRYAnsatz, CyclicAnsatz, RXXAnsatz
from feature_map import AngleVectorEmbedding, SinusoidalPositionEmbedding, CosineScalarEmbedding, RZZ_FeatureMap, RZZ_Layer, ChebyshevFeatureMap

print("Qiskit Aer backends:")
print(Aer.backends())

NQUBITS = 11

backend = GBV2(num_qubits=NQUBITS, seed=42)
qk_noise_model = NoiseModel.from_backend(backend)

dev = qml.device("default.qubit", wires=NQUBITS)
NSHOTS = int(1e4)
noisy_dev = qml.device("qiskit.aer", wires=NQUBITS, shots=NSHOTS, noise_model=qk_noise_model)


def branch_routine(params, inputs):
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
    SU2Ansatz(params, wires=range(NQUBITS), entanglement_structure="linear", entanglement_gate="Z")
    #SU3Ansatz(params, wires=range(NQUBITS), entanglement_structure="linear", entanglement_gate="Z")
    # return expectation value of Pauli operator on qubits
    return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]


def trunk_routine(params, input):
    """Parametrized Quantum Circuit for Trunk Network
    Args:
        params (array):
        input (float):
    """
    # single input embedding that increases complexity
    ChebyshevFeatureMap(input, wires=range(NQUBITS))
    #CosineScalarEmbedding(input, nqubits=NQUBITS)
    qml.Barrier()
    # evaluate parametrized ansatz
    SU1Ansatz(params, wires=range(NQUBITS), entanglement_structure="linear", entanglement_gate="Z")
    # return expectation value of Pauli operator on qubits
    return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]


branch_circuit = qml.QNode(branch_routine, dev, interface="jax")
trunk_circuit = qml.QNode(trunk_routine, dev, interface="jax")
noisy_branch_circuit = qml.QNode(branch_routine, noisy_dev, interface="jax")
noisy_trunk_circuit = qml.QNode(trunk_routine, noisy_dev, interface="jax")


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
    # return jnp.dot(branch_output, trunk_output)[0] + params["bias"]


def noisy_PQOC(params, branch_input, trunk_input):
    """"""
    # call branch circuit
    branch_output = noisy_branch_circuit(params["branch"], branch_input)
    # call trunk circuit
    trunk_output = noisy_trunk_circuit(params["trunk"], trunk_input)
    # compute and return inner product
    return jnp.dot(jnp.array(branch_output).ravel(), jnp.array(trunk_output).ravel()) + params["bias"]  # ravel seems janky but prevents vmap from destroying trunk output dimension


def normalized_l2_error(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)
    return diff_norm / ref_norm


def rmse_error(pred, ref):
    mse = jnp.mean((pred - ref)**2)
    return jnp.sqrt(mse)


def import_heat_eqn_data(nqubits):
    rootdir = os.getcwd()
    os.chdir("../heat_eqn_data")
    sys.path.append(os.getcwd())
    from load_heat_eqn_data import load_heat_eqn_data_deeponet
    branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn = load_heat_eqn_data_deeponet(ngrid=nqubits)
    os.chdir(rootdir)
    return branch_inputs_train_dwn, branch_inputs_test_dwn, trunk_inputs_train_dwn, trunk_inputs_test_dwn, outputs_train_dwn, outputs_test_dwn


if __name__ == "__main__":
    print("==> NQUBITS has been set to", NQUBITS)
    assert NQUBITS == 11, "number of qubits must be 11"
    branch_inputs_train, branch_inputs_test, trunk_inputs_branch, trunk_inputs_query, outputs_train, outputs_test = import_heat_eqn_data(NQUBITS)

    resultdir = f"heat_results"
    noisyresultdir = f"{resultdir}/noisy"
    paramfile = f"{resultdir}/params.json"
    params = json.load(open(paramfile, "r"))

    if not os.path.isdir(resultdir):
        os.mkdir(resultdir)
    if not os.path.isdir(noisyresultdir):
        os.mkdir(noisyresultdir)

    for key, item in params.items():
        params[key] = np.array(item, dtype=np.float32)
        # params[key] = jnp.array(item, dtype=jnp.float32)

    assert params["branch"].shape == (NQUBITS, NQUBITS, 2)
    assert params["trunk"].shape == (NQUBITS, NQUBITS)

    predictions_test = jax.vmap(
        jax.vmap(
            PQOC, in_axes=(None, None, 0)
            ), 
        in_axes=(None, 0, None)
    )(params, branch_inputs_test, trunk_inputs_query)  
    #breakpoint()

    # we have to do this bc qiskit backends don't have jax support like pennylane...
    noisy_predictions_test = np.zeros_like(predictions_test)
    for i in range(branch_inputs_test.shape[0]):
        for j in range(trunk_inputs_query.shape[0]):
            noisy_predictions_test[i,j] = noisy_PQOC(params, branch_inputs_test[i, :], trunk_inputs_query[j])
        if i % 10 == 0:
            print("still making noisy data, don't fall asleep...", i)

    print(branch_inputs_train.shape, trunk_inputs_branch.shape)
    print(branch_inputs_test.shape, trunk_inputs_query.shape)
    print(predictions_test.shape, noisy_predictions_test.shape)

    errs = jax.vmap(rmse_error)(predictions_test, outputs_test)
    noisy_errs = jax.vmap(rmse_error)(noisy_predictions_test, outputs_test)

    with open(f"{resultdir}/test_report.txt", "w") as f:
        f.write(f"Mean RMSE over test set: {jnp.mean(errs)}\n")
        f.write(f"Standard Deviation of RMSE over test set: {jnp.std(errs)}\n")
        f.write(f"Median RMSE over test set: {jnp.median(errs)}\n")
    with open(f"{noisyresultdir}/test_report.txt", "w") as f:
        f.write(f"Mean RMSE over test set: {jnp.mean(noisy_errs)}\n")
        f.write(f"Standard Deviation of RMSE over test set: {jnp.std(noisy_errs)}\n")
        f.write(f"Median RMSE over test set: {jnp.median(noisy_errs)}\n")

    def viz(point):
        plt.figure(figsize=(10,5))
        plt.plot(trunk_inputs_branch[:,0], branch_inputs_test[point, :], marker="o", label="INPUT")
        plt.plot(trunk_inputs_query[:,0], outputs_test[point, :], marker="+", label="TARGET")
        plt.plot(trunk_inputs_query[:,0], predictions_test[point, :], marker="+", label="PREDICTION")
        rmse = jnp.sqrt(jnp.mean((outputs_test[point, :] - predictions_test[point, :])**2))
        plt.title(f"Quantum ONet, Test Datapoint {point}, RMSE {rmse:.4f}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{resultdir}/test_{point}.png")
        plt.close()

    viz(13)
    viz(32)
    viz(42)
    viz(55)
    viz(67)
    viz(80)
    
    def viz(point):
        plt.figure(figsize=(10,5))
        plt.plot(trunk_inputs_branch[:,0], branch_inputs_test[point, :], marker="o", label="INPUT")
        plt.plot(trunk_inputs_query[:,0], outputs_test[point, :], marker="+", label="TARGET")
        plt.plot(trunk_inputs_query[:,0], noisy_predictions_test[point, :], marker="+", label="PREDICTION")
        rmse = jnp.sqrt(jnp.mean((outputs_test[point, :] - noisy_predictions_test[point, :])**2))
        plt.title(f"Noisy Quantum ONet, Test Datapoint {point}, RMSE {rmse:.4f}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{noisyresultdir}/test_{point}.png")
        plt.close()

    viz(13)
    viz(32)
    viz(42)
    viz(55)
    viz(67)
    viz(80)

    breakpoint()
