import pennylane as qml


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


def SU2Ansatz(theta, wires, entanglement_structure:str="circular", entanglement_gate:str="Z"):
    """Defines ansatz using SU(2) group of RZ & RY.
    Args:
        theta (array-like): parameters, shape (R+1, NQUBITS, 2)
        wires (list[int]): numbered wires to apply ansatz to
        entanglement_structure (int|str): structure for entanglement. numeric means nearest n qubits.
        entanglement_gate (str): which controlled-operation gate to use in entanglement operations
    """
    for r in range(theta.shape[0] - 1):
        for w in wires:
            qml.RZ(theta[r,w,0], wires=w)
            qml.RY(theta[r,w,1], wires=w)
        qml.Barrier()
        entanglement_layer(wires, structure=entanglement_structure, gate=entanglement_gate)
        qml.Barrier()
    for w in wires:
        qml.RZ(theta[-1,w,0], wires=w)
        qml.RY(theta[-1,w,1], wires=w)


def SU1Ansatz(theta, wires, entanglement_structure:str="circular", entanglement_gate:str="Z"):
    """Defines ansatz using SU(1) group of RY.
    Args:
        theta (array-like): parameters, shape (R+1, NQUBITS)
        wires (list[int]): numbered wires to apply ansatz to
        entanglement_structure (int|str): structure for entanglement. numeric means nearest n qubits.
        entanglement_gate (str): which controlled-operation gate to use in entanglement operations
    """
    for r in range(theta.shape[0] - 1):
        for w in wires:
            qml.RY(theta[r,w], wires=w)
        qml.Barrier()
        entanglement_layer(wires, structure=entanglement_structure, gate=entanglement_gate)
        qml.Barrier()
    for w in wires:
        qml.RY(theta[-1,w], wires=w)


def SU1CRYAnsatz(theta, cry_theta, repeats:int=1, nqubits:int=4, entanglement_structure:str="linear"):
    """Defines ansatz using SU(1) group of RY and C-RY gates for entanglement.
    Args:
        theta (array-like): parameters. must have shape of NQUBITS * (repeats + 1)
        cry_theta (array-like): parameters for controlled rotations, shape of repeats * (nqubits - int(entanglement_structure!="circular"))
        repeats (int): number of times entanglement and RY are repeated.
        entanglement_structure (str): linear (default) or circular
        entanglement_gate (str): which controlled-operation gate to use in entanglement operations
    """
    for i in range(nqubits):
        qml.RY(theta[i], wires=i)
    for r in range(repeats):
        qml.Barrier()
        for i in range(nqubits - int(entanglement_structure.lower()!="circular")):
            qml.CRY(cry_theta[i+(r*(nqubits - int(entanglement_structure.lower()!="circular")))], wires=[i, (i+1)%nqubits])
        qml.Barrier()
        for i in range(nqubits):
            qml.RY(theta[i+(r+1)*nqubits], wires=i)


def SU2CRYAnsatz(theta, cry_theta, repeats:int=1, nqubits:int=4, entanglement_structure:str="linear"):
    """Defines ansatz using SU(2) group of RZ and RX, with C-RY gates for entanglement.
    Args:
        theta (array-like): parameters. must have shape of 2 * NQUBITS * (repeats + 1)
        cry_theta (array-like): parameters for controlled rotations, shape of repeats * (nqubits - int(entanglement_structure!="circular"))
        repeats (int): number of times entanglement and RY RZ are repeated.
        entanglement_structure (int|str): structure for entanglement. numeric means nearest n qubits.
        entanglement_gate (str): which controlled-operation gate to use in entanglement operations
    """
    for i in range(nqubits):
        qml.RZ(theta[i], wires=i)  # [0:n)
        qml.RX(theta[i+nqubits], wires=i)  # [n:2n)
    for r in range(repeats):
        qml.Barrier()
        for i in range(nqubits - int(entanglement_structure.lower()!="circular")):
            qml.CRY(cry_theta[i+(r*(nqubits - int(entanglement_structure.lower()!="circular")))], wires=[i, (i+1)%nqubits])
        qml.Barrier()
        for i in range(nqubits):
            qml.RZ(theta[i+2*(r+1)*nqubits], wires=i)      # r=0:2n->3n, r=1:4n->5n, ...
            qml.RX(theta[i+(2*(r+1)+1)*nqubits], wires=i)  # r=0:3n->4n, r=1:5n->6n, ...


# stealing the ansatz below from: https://pennylane.ai/qml/demos/tutorial_post-variational_quantum_neural_networks
def CyclicAnsatz(theta, nqubits:int=4, cgate:str="Z"):
    """Cyclic Ansatz
    Args:
        theta (array(float)): must have shape of (3, NQUBITS)
        nqubits (int):
        cgate (str):
    """
    # Apply RY rotations with the first set of parameters
    for i in range(nqubits):
        # qml.RY(theta[i], wires=i)
        qml.RY(theta[0, i], wires=i)

    # Apply CNOT gates with adjacent qubits (cyclically connected) to create entanglement
    for i in range(nqubits):
        if cgate == "Z":
            qml.CZ(wires=[(i - 1) % nqubits, (i) % nqubits])
        else:
            qml.CNOT(wires=[(i - 1) % nqubits, (i) % nqubits])

    # Apply RY rotations with the second set of parameters
    for i in range(nqubits):
        # qml.RY(theta[i + nqubits], wires=i)
        qml.RY(theta[1, i], wires=i)

    # Apply CNOT gates with qubits in reverse order (cyclically connected)
    # to create additional entanglement
    for i in range(nqubits):
        if cgate == "Z":
            qml.CZ(wires=[(nqubits - 2 - i) % nqubits, (nqubits - i - 1) % nqubits])
        else:
            qml.CNOT(wires=[(nqubits - 2 - i) % nqubits, (nqubits - i - 1) % nqubits])

    # Apply RY rotations with the third set of parameters
    for i in range(nqubits):
        # qml.RY(theta[i + 2*nqubits], wires=i)
        qml.RY(theta[2, i], wires=i)


def RXXAnsatz(theta, wires:list[int]):
    """RXX Variational Ansatz
    Args:
        theta (array-like): parameters, shape (R, NQUBITS, 3)
        wires (list[int]):
    """
    for r in range(theta.shape[0]):
        qml.Barrier()
        # RXX can be implemented in Pennylane via:
        for w in range(0, len(wires)-1, 2):
            qml.PauliRot(theta[r, w, 0], "XX", wires=[wires[w], wires[w+1]])
        for w in range(1, len(wires)-1, 2):
            qml.PauliRot(theta[r, w, 1], "XX", wires=[wires[w], wires[w+1]])
        qml.Barrier()
        for w in wires:
            qml.RY(theta[r, w, 2], wires=w)


def SU_SHARED():
    """TODO: Parameter sharing ansatz. need ones that share "vertically" and "horizontally" (along wire) per repeat group"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy
    NQUBITS = 6

    @qml.qnode(qml.device("default.qubit", wires=NQUBITS))
    def test_circuit():
        #SU1CRYAnsatz(range(15), range(10), repeats=2, nqubits=NQUBITS, entanglement_structure="circular")
        SU3Ansatz(numpy.random.random((3,NQUBITS,3)), wires=range(NQUBITS), entanglement_gate="X", entanglement_structure="diamond")
        #SU2Ansatz(numpy.random.random((3,NQUBITS,2)), wires=range(NQUBITS))
        #RXXAnsatz(numpy.random.random((2,NQUBITS,3)), wires=range(NQUBITS))
        #entanglement_layer(range(NQUBITS), ["linear", "circular", "full", "diamond"][3])
        return qml.state()  # , [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]
    
    #state = test_circuit()
    qml.draw_mpl(test_circuit, style="pennylane", decimals=3)()
    plt.savefig("ansatz_test_circuit.png")
    plt.close()

    breakpoint()
