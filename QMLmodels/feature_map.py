import pennylane as qml
import jax.numpy as jnp


def AngleVectorEmbedding(inputs, nqubits:int=4, rotation:str="Y", offset_and_invert:bool=True):
    """Sinusoidal Positional Embedding"""
    for wire in range(nqubits):
        match rotation.upper():
            case "X":
                qml.RX(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert) * inputs[wire], wires=wire)
            case _:  # default to RY
                qml.RY(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert) * inputs[wire], wires=wire)


def SinusoidalPositionEmbedding(input, nqubits:int=4, rotation:str="Y", offset_and_invert:bool=True):
    """Sinusoidal Positional Embedding"""
    for wire in range(nqubits):
        match rotation.upper():
            case "X":
                qml.RX(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert) * jnp.sin((2**wire) * input / nqubits), wires=wire)
            case _:  # default to RY
                qml.RY(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert) * jnp.sin((2**wire) * input / nqubits), wires=wire)


def CosineScalarEmbedding(input, nqubits:int=4, rotation:str="Y", offset_and_invert:bool=False):
    """Cosine Scalar Embedding
    Expectation values of PauliZ should represent a Fourier (cosine) series
    If offset_and_invert, sine series(?)
    """
    for wire in range(nqubits):
        match rotation.upper():
            case "X":
                qml.RX(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert) * (2**wire) * input, wires=wire)
            case _:  # default to RY
                #print(wire)
                #print(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert + 1 - offset_and_invert) * (2**wire) * input)
                # qml.RY(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert + 1 - offset_and_invert) * (wire+1) * input * jnp.pi, wires=wire)
                qml.RY(jnp.pi/2*offset_and_invert + (-1 * offset_and_invert + 1 - offset_and_invert) * input * ((wire+1)*2-1) * jnp.pi, wires=wire)


def DataReuploadEmbedding(input, nqubits:int=4, n_layers:int=2, rotation:str="Y"):
    """Data Reuploading Scheme Embedding"""
    for _ in range(n_layers):
        for wire in range(nqubits):
            match rotation.upper():
                case "X":
                    qml.RX((wire+1)*input, wires=wire)
                    qml.CNOT(wires=[wire, (wire + 1) % nqubits])
                case _:  # default to RY
                    qml.RY(jnp.pi/2 - (wire+1) * input, wires=wire)
                    qml.CNOT(wires=[wire, (wire + 1) % nqubits])

# the feature map below is inspired by: https://pennylane.ai/qml/demos/tutorial_post-variational_quantum_neural_networks
def DoubleVectorEmbedding(inputs, nqubits:int=4):
    """Embedding for two 1D vector inputs.
    Args:
        inputs (array): shape of (2, nqubits)
        nqubits (int): number of qubits in circuit
    """
    for wire in range(nqubits):
        qml.Hadamard(wire)
    qml.AngleEmbedding(inputs[0, :], wires=range(nqubits), rotation="Z")
    qml.AngleEmbedding(inputs[1, :], wires=range(nqubits), rotation="X")


def ChebyshevFeatureMap(input, wires):
    """Chebyshev Feature Map, based off of the one in [Self-Adaptive Physics-Informed Quantum Machine Learning for Solving Differential Equations](https://arxiv.org/pdf/2312.09215)
    Args:
        input (float): 
        wires (array):
    """
    for wire in wires:
        # qml.RY(2 * (wire + 1) * jnp.arccos(input), wires=wire)
        coef = 2**(wire - wires[0]) / len(wires)
        qml.RY(coef * jnp.arccos(input), wires=wire)


def RZZ_FeatureMap(inputs, nqubits:int=4, mode:str="full"):
    """Feature Map for a vector using RZ and RZZ gates. See A.1 in: [The power of quantum neural networks](https://arxiv.org/pdf/2011.00027) 

    First, the feature map applies Hadamard gates on each of the S := s_in qubits, 
        followed by a layer of RZ-gates, whereby the angle of the Pauli rotation on qubit i 
        depends on the i-th feature x_i of the data vector ~x, normalised between [-1, 1]

    Then, RZZ-gates are implemented on qubits i, i + j for i in [1, ..., S - 1] and j in [i + 1, ..., S] 
        using a decomposition into two CNOT-gates and one RZ-gate 
        with a rotation angle (pi - x_i) (pi - x_{i+j} ).

    Args:
        inputs (array):
        nqubits (int):
        mode (str): full (default), next, circular
    """
    for wire in range(nqubits):
        qml.Hadamard(wire)
        qml.RZ(inputs[wire], wires=wire)
    RZZ_Layer(inputs, nqubits=nqubits, mode=mode)


def RZZ_Layer(inputs, nqubits:int=4, mode:str="full"):
    match mode.lower():
        case "linear":
            for i in range(0, nqubits-1):
                j = i + 1
                qml.CNOT(wires=[i,j])
                # qml.RZ((jnp.pi - inputs[i]) * (jnp.pi - inputs[j]), wires=j)
                qml.RZ((inputs[i]) * (inputs[j]), wires=j)
                qml.CNOT(wires=[i,j])
        case "circular":
            for i in range(0, nqubits):
                j = i + 1
                qml.CNOT(wires=[i,j%nqubits])
                # qml.RZ((jnp.pi - inputs[i]) * (jnp.pi - inputs[j%nqubits]), wires=j%nqubits)  # TODO is pi - input the best way? does that even matter?
                qml.RZ((inputs[i]) * (inputs[j%nqubits]), wires=j%nqubits)
                qml.CNOT(wires=[i,j%nqubits])
        case _:  # default to full
            for i in range(0, nqubits-1):
                for j in range(i+1, nqubits):
                    qml.CNOT(wires=[i,j])
                    # qml.RZ((jnp.pi - inputs[i]) * (jnp.pi - inputs[j]), wires=j)
                    qml.RZ((inputs[i]) * (inputs[j]), wires=j)
                    qml.CNOT(wires=[i,j])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    NQUBITS = 11

    @qml.qnode(qml.device("default.qubit", wires=NQUBITS))
    def test_circuit(x):
        #RZZ_FeatureMap([1,2,3,4,5], mode="circular", nqubits=NQUBITS)
        #CosineScalarEmbedding(x, nqubits=NQUBITS, rotation="Y", offset_and_invert=False)
        ChebyshevFeatureMap(x, wires=range(NQUBITS))
        return [qml.expval(qml.PauliZ(n)) for n in range(NQUBITS)]
    
    output = test_circuit(1)
    qml.draw_mpl(test_circuit, style="pennylane", decimals=2)(0)
    plt.savefig("feature_map.png")
    plt.close()
    
    x_domain = jnp.linspace(0, 1, NQUBITS)
    asdf = [test_circuit(x) for x in x_domain]
    plt.figure()
    for i in range(NQUBITS):
        plt.plot(x_domain, [x[i] for x in asdf], label=str(i))
    plt.legend()
    plt.savefig("featurebasis.png")
    plt.close()
    breakpoint()
