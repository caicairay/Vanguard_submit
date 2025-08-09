import numpy as np
import pennylane as qml
from pennylane import qaoa
from functools import partial

from pennylane.qaoa import cost_layer
from pennylane.qaoa import mixer_layer
from pennylane.qaoa.mixers import x_mixer


def qubo_obj_to_cost_H(qubo_obj):
    """Convert the QUBO objective matrix to the cost Hamiltonian

    Parameter:
    - qubo_obj: an upper triangle matrix

    Return:
    - cost_H: ZZ Hamiltonian (Pennylane representation)
    """

    n = qubo_obj.shape[0]
    coeffs = []
    ops = []
    for i in range(n):
        # Diagonal: single Z
        if qubo_obj[i, i] != 0:
            coeffs.append(qubo_obj[i, i])
            ops.append(qml.PauliZ(i))
        # Off-diagonal: ZZ
        for j in range(i + 1, n):
            if qubo_obj[i, j] != 0:
                coeffs.append(qubo_obj[i, j])
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)


def generate_random_cost_and_mixer_layers(num_qubits, n_QAOA_layer, H_cost):
    cost_layers = []
    mixer_layers = []

    for _ in range(n_QAOA_layer):
        # --- Cost Layer ---
        gamma = np.random.uniform(0, 2 * np.pi)
        cost = cost_layer(gamma, H_cost)
        # cost = qml.evolve(H_cost, coeff=gamma)  # not compatible with lightning.qubit

        # --- Mixer Layer ---
        beta = np.random.uniform(0, 2 * np.pi)

        # mixer_h = x_mixer(range(num_qubits))
        coeffs = 2 * np.random.rand(num_qubits) - 1  # Random coeffs in [-1, 1]
        ops = [qml.PauliX(w) for w in range(num_qubits)]
        mixer_h = qml.Hamiltonian(coeffs, ops)

        mixer = mixer_layer(beta, mixer_h)
        cost_layers.append(cost)
        mixer_layers.append(mixer)

    return cost_layers, mixer_layers


def random_Y_layer(n_qubits, angle=np.pi):
    """Applies a unitary evolution e^{-i * angle * H} with H = sum_i Y_i on random subset of qubits."""
    ops = []
    coeffs = []

    for i in range(n_qubits):
        if np.random.rand() < 0.5:
            ops.append(qml.PauliY(i))
            coeffs.append(1.0)

    if not ops:
        # ensure at least one PauliY term
        i = np.random.randint(0, n_qubits)
        ops.append(qml.PauliY(i))
        coeffs.append(1.0)

    H = qml.Hamiltonian(coeffs, ops)
    return mixer_layer(angle, H)


class Operators:
    def __init__(
        self,
        num_max_b=4,
        num_qubits=6,
        num_shots=30,
        n_StrongEnt_layer=10,  # number of repeated strongly Ent layers.
        n_QAOA_layer=5,
        n_RY=100,
        init_state_func=qml.qchem.hf_state,
        device="default.qubit",
    ):
        self.num_max_b = num_max_b
        self.num_qubits = num_qubits
        self.num_shots = num_shots
        self.init_state = init_state_func(num_max_b, num_qubits)

        if device == "qiskit.aer":
            from qiskit_aer import AerSimulator

            self.simulator = AerSimulator(method="matrix_product_state")
            self.dev = qml.device(
                device,
                wires=num_qubits,
                backend=self.simulator,
#                 backend_options={"method": ["matrix_product_state"]},
            )
            self.dev.backend.set_options(device='GPU')

        elif "default.tensor" in device:
            device_kwargs_mps = {
                "max_bond_dim": 100,
                "cutoff": np.finfo(np.complex128).eps,
                "contract": "auto-mps",
            }
            self.dev = qml.device(
                device, wires=num_qubits, method="mps", **device_kwargs_mps
            )
        else:
            self.dev = qml.device(device, wires=num_qubits)

        self.n_StrongEnt_layer = n_StrongEnt_layer
        self.wshape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_StrongEnt_layer, n_wires=num_qubits
        )
        self.weights = np.random.random(size=self.wshape)

        self.n_QAOA_layer = n_QAOA_layer

        self.n_RY = n_RY

    def generate_op_pool(self, qubo_obj=None, op_types=None):
        """Generate operator pool.

        Parameter:
        - qubo_obj: QUBO objective matrix ~ cost Hamiltonian (required if QAOA ops are included)
        - op_types: list of strings, e.g. ["DoubleExcitation", "SingleExcitation", "Identity", "StronglyEntanglingLayers", "QAOA"]

        Return:
        - operator_pool as a numpy.ndarry
        """
        op_times = np.sort(
            np.array([-(2**k) for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160
        )
        singles, doubles = qml.qchem.excitations(self.num_max_b, self.num_qubits)
        operator_pool = []

        if op_types is None:
            op_types = [
                "DoubleExcitation",
                "SingleExcitation",
                "Identity",
                "StronglyEntanglingLayers",
                "QAOA",
                "RY",
            ]

        for op_name in op_types:
            if op_name == "DoubleExcitation":
                operator_pool += [
                    qml.DoubleExcitation(time, wires=double)
                    for double in doubles
                    for time in op_times
                ]
            elif op_name == "SingleExcitation":
                operator_pool += [
                    qml.SingleExcitation(time, wires=single)
                    for single in singles
                    for time in op_times
                ]
            elif op_name == "Identity":
                operator_pool += [
                    qml.exp(qml.I(range(self.num_qubits)), 1j * time)
                    for time in op_times
                ]
            elif op_name == "StronglyEntanglingLayers":
                operator_pool.append(
                    qml.StronglyEntanglingLayers(
                        weights=self.weights,
                        ranges=[1] * self.n_StrongEnt_layer,
                        wires=range(self.num_qubits),
                    )
                )
            elif op_name == "QAOA":
                if qubo_obj is None:
                    raise ValueError(
                        "THe cost Hamiltonian must be provided for QAOA operators."
                    )

                H_cost = qubo_obj_to_cost_H(qubo_obj)
                qaoa_c, qaoa_m = generate_random_cost_and_mixer_layers(
                    self.num_qubits, self.n_QAOA_layer, H_cost
                )
                operator_pool += qaoa_c + qaoa_m

            elif op_name == "RY":
                operator_pool += [
                    random_Y_layer(self.num_qubits) for _ in range(self.n_RY)
                ]
                
            elif op_name == 'PauliY':
                operator_pool += [qml.PauliY(i) for i in range(self.num_qubits)]
            elif op_name == 'PauliX':
                operator_pool += [qml.PauliX(i) for i in range(self.num_qubits)]

        return np.array(operator_pool)
    
    def get_distribution(self, op_seq):
        @partial(qml.set_shots, shots=self.num_shots)
        @qml.qnode(self.dev)
        def distribution_circuit(ops):
            # apply all randomly chosen ops from the pool.
            for layer in ops:
                qml.apply(layer)
            # return sampled results as dict.
            return qml.counts()
        # Collates the energies of each subsequence for a batch of sequences
        distributions = []
        for ops in op_seq:
            distributions.append(distribution_circuit(ops))
        return np.array(distributions)