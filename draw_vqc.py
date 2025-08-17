import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

NQ = 4

dev = qml.device("default.qubit", wires=NQ)


def entanglement_layer(wires, structure:str="linear", gate:str="Z"):
    """Dedicated layer for creating entanglementin variational circuits.
    Args:
        wires (list[int]): numbered wires to apply ansatz to
        structure (str): linear (default), circular, full, or diamond
    """
    match gate.upper():
        case "X":
            cgate = qml.CNOT
        case _:
            cgate = qml.CZ
    match structure.lower():
        case "circular":
            for w in range(len(wires)):
                cgate([wires[w], wires[(w+1)%len(wires)]])
        case "full":
            for i in range(len(wires)-1):
                for j in range(i+1, len(wires)):
                    cgate([wires[i], wires[j]])
        case "diamond":
            for w in range(1, len(wires)):
                cgate([0, wires[w]])
            for w in range(0, len(wires)-1):
                cgate([wires[w], wires[-1]])
        case _:  # default to linear
            for w in range(len(wires)-1):
                cgate([wires[w], wires[w+1]])


def SU3Ansatz(theta, wires, entanglement_structure:str="circular", entanglement_gate:str="Z"):
    """Defines ansatz using SU(2) group of RZ & RY.
    Args:
        theta (array-like): parameters, shape (R+1, NQUBITS, 3)
        wires (list[int]): numbered wires to apply ansatz to
        entanglement_structure (int|str): structure for entanglement. numeric means nearest n qubits.
        entanglement_gate (str): which controlled-operation gate to use in entanglement operations
    """
    for r in range(theta.shape[0] - 1):
        for w in wires:
            qml.RZ(theta[r,w,0], wires=w)
            qml.RY(theta[r,w,1], wires=w)
            qml.RZ(theta[r,w,2], wires=w)
        qml.Barrier()
        entanglement_layer(wires, structure=entanglement_structure, gate=entanglement_gate)
        qml.Barrier()
    for w in wires:
        qml.RZ(theta[-1,w,0], wires=w)
        qml.RY(theta[-1,w,1], wires=w)
        qml.RZ(theta[-1,w,2], wires=w)


@qml.qnode(dev)
def VQC(params, inputs):
    """Sample Variational Quantum Circuit."""
    qml.AngleEmbedding(inputs, wires=range(NQ), rotation="Y")
    qml.Barrier()
    SU3Ansatz(params, wires=range(NQ), entanglement_structure="linear", entanglement_gate="Z")
    qml.Barrier()
    return qml.state()


params = np.random.random((4,4,3))
inputs = np.array([0.1, 0.2, 0.3, 0.4])
qml.draw_mpl(VQC)(params, inputs)
plt.savefig("vqc.png")
plt.close()
breakpoint()
