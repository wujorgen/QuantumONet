import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from train_MI_pqoc import branch_circuit, multi_embed_branch_circuit, trunk_circuit, import_coupled_pde_data, NQUBITS


def draw_1dcoupled_circuit():
    folder = "1dcoupled_results"

    paramspath = os.path.join(folder, "params.json")

    params = json.load(open(paramspath, "r"))

    for key in params.keys():
        params[key] = jnp.array(params[key], dtype=jnp.float32)

    branch_inputs_train, branch_inputs_test, trunk_inputs_train, trunk_inputs_test, outputs_train, outputs_test = import_coupled_pde_data(NQUBITS)

    plt.figure()
    qml.draw_mpl(branch_circuit, style="pennylane", decimals=2)(params["branch_a"], branch_inputs_test[0,:,:])
    plt.savefig(f"{folder}/branch_a_circuit.png")
    plt.close()

    plt.figure()
    qml.draw_mpl(branch_circuit, style="pennylane", decimals=2)(params["branch_b"], branch_inputs_test[0,:,:])
    plt.savefig(f"{folder}/branch_b_circuit.png")
    plt.close()

    plt.figure()
    qml.draw_mpl(trunk_circuit, style="pennylane", decimals=2)(params["trunk"], trunk_inputs_test[0])
    plt.savefig(f"{folder}/trunk_circuit.png")
    plt.close()

    breakpoint()

if __name__ == "__main__":
    draw_1dcoupled_circuit()