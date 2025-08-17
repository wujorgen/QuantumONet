import pennylane as qml
import qiskit
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel

print(Aer.backends())

from qiskit.providers.fake_provider import GenericBackendV2

NQUBITS = 4

backend = GenericBackendV2(num_qubits=NQUBITS, seed=42)

qk_noise_model = NoiseModel.from_backend(backend)
print(qk_noise_model)
