# ============================================================
# HHL Algorithm – Qiskit Implementation
# ============================================================

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# Output Directory Creation (auto-create if not exists)
# ─────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Returns the full path to save the output file."""
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────
# Global Parameters
# ─────────────────────────────────────────────
n_qubits = 1          # Number of qubits in the system
phase_qubits = 4      # Number of qubits for phase estimation (QPE)
ancilla_qubit = 1     # Number of ancilla qubits
tot_qubits = n_qubits + phase_qubits + ancilla_qubit

# Hermitian Matrix A
A_matrix = np.array([[-5, 0.5],
                     [0.5, -4]], dtype=np.complex128)
A_matrix = -A_matrix  # Using -A as in PennyLane

# Vector b
b_vector = np.array([0.03, -0.02])
norm_b = b_vector / np.linalg.norm(b_vector)

# Printing problem parameters
print("=== Problem Parameters ===")
print(f"A =\n{A_matrix}")
print(f"b = {b_vector}")
print(f"||b|| = {np.linalg.norm(b_vector):.6f}")
print(f"Normalized b̂ = {norm_b}")

# ─────────────────────────────────────────────
# Compute U = exp(iA)
# ─────────────────────────────────────────────
t = 0.2
U_matrix = expm(1j * A_matrix * t)
assert np.allclose(U_matrix.conj().T @ U_matrix, np.eye(2**n_qubits)), "U is not unitary!"
print(f"\nU = exp(iA) =\n{np.round(U_matrix, 4)}")

# ─────────────────────────────────────────────
# Function to apply controlled-U^power
# ─────────────────────────────────────────────
def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    """Applies controlled-U^power to the target qubits with the control qubit."""
    U_power = np.linalg.matrix_power(U, power)
    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(U_power, list(range(len(target_qubits))))
    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [control_qubit] + list(target_qubits))

# ─────────────────────────────────────────────
# Quantum Phase Estimation (QPE)
# ─────────────────────────────────────────────
def qpe_circuit(qc, phase_reg, target_reg):
    """Performs the Quantum Phase Estimation (QPE) procedure."""
    # Apply Hadamard on all phase qubits
    for w in phase_reg:
        qc.h(w)

    # Apply controlled-U^{2^k}
    for i, ctrl in enumerate(reversed(phase_reg)):
        power = 2 ** i
        controlled_unitary_power(qc, U_matrix, power, ctrl, target_reg)

    # Apply Inverse QFT
    iqft = QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=False)
    qc.append(iqft, phase_reg)

# ─────────────────────────────────────────────
# Controlled RY Rotation (Eigenvalue Encoding)
# ─────────────────────────────────────────────
def control_rotation_gate(qc, control_reg, target_qubit, t, C=10):
    """Performs controlled RY rotation for eigenvalue encoding."""
    n = len(control_reg)

    for d in range(1, 2**n):
        bin_str = f"{d:0{n}b}"  # MSB -> LSB

        # Calculate phase from bitstring
        phi = sum(int(b) * 2**(-(j+1)) for j, b in enumerate(bin_str))
        
        # QPE returns phase in [0, 1). If phi > 0.5, it's actually a negative angle.
        if phi > 0.5:
            phi = phi - 1.0
            
        lam = (2 * np.pi / t) * phi
        
        if lam == 0:
            continue

        # Compute amplitude
        amp = C / lam
        
        if abs(amp) > 1:
            continue

        # Compute rotation angle
        theta = 2 * np.arcsin(amp)
        ctrl_state = bin_str  

        # Apply controlled RY gate
        from qiskit.circuit.library import RYGate
        cry = RYGate(theta).control(n, ctrl_state=ctrl_state)
        qc.append(cry, list(control_reg) + [target_qubit])

# ─────────────────────────────────────────────
# Full HHL Circuit Implementation
# ─────────────────────────────────────────────

phase_reg = QuantumRegister(phase_qubits, name='phase')
target_reg = QuantumRegister(n_qubits, name='target')
ancilla_reg = QuantumRegister(ancilla_qubit, name='ancilla')

qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

# Absolute qubit indices
phase_indices = list(range(phase_qubits))
target_indices = list(range(phase_qubits, phase_qubits + n_qubits))
ancilla_indices = list(range(phase_qubits + n_qubits, tot_qubits))

# ── Step 1: Initialize |b⟩ on target qubits ──
qc_hhl.initialize(norm_b, target_indices)

# ── Step 2: QPE ──
qc_hhl.barrier(label="QPE")
qpe_circuit(qc_hhl, phase_indices, target_indices)

# ── Step 3: Controlled RY ──
qc_hhl.barrier(label="AQE")
control_rotation_gate(qc_hhl, phase_indices, ancilla_indices[0], t=t, C=1.0)

# ── Step 4: Inverse QPE (adjoint) ──
qc_hhl.barrier(label="QPE†")
# Construct QPE circuit and take inverse
qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
qpe_circuit(qc_qpe_only, list(range(phase_qubits)), list(range(phase_qubits, phase_qubits + n_qubits)))
inv_qpe = qc_qpe_only.inverse()
inv_qpe.name = "QPE†"
qc_hhl.append(inv_qpe, phase_indices + target_indices)

qc_hhl.barrier()

# ─────────────────────────────────────────────
# Draw the Circuit
# ─────────────────────────────────────────────
fig_main, ax_main = plt.subplots(figsize=(20, 6))
qc_hhl.decompose().draw(output='mpl', ax=ax_main,
                         style={'backgroundcolor': '#EEEEEE'},
                         fold=60)
ax_main.set_title("HHL Circuit (Decomposed Level 1)", fontsize=14)
plt.tight_layout()
plt.savefig(out("hhl_circuit_main.png"), dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Simulate using Statevector
# ─────────────────────────────────────────────
print("\n=== Running Statevector Simulation ===")
sv = Statevector.from_instruction(qc_hhl)

# Extract probabilities from (target + ancilla)
n_meas = n_qubits + ancilla_qubit  # = 2
probs_ta = sv.probabilities(qargs=target_indices + ancilla_indices)

print(f"Probabilities (target ⊗ ancilla) = {probs_ta}")

# Bit order in probs_ta (little-endian Qiskit):
# index 0 = |ancilla=0, target=0⟩  -> |00⟩
# index 1 = |ancilla=0, target=1⟩  -> |01⟩  (ancilla bit=0)
# index 2 = |ancilla=1, target=0⟩  -> |10⟩  (ancilla bit=1)
# index 3 = |ancilla=1, target=1⟩  -> |11⟩

# Filter states where ancilla = 0 (high bit = 0, i.e., index 0 and 1)
ancilla_zero_mask = [(i >> n_qubits) & 1 == 1 for i in range(len(probs_ta))]
probs_ancilla1 = probs_ta[ancilla_zero_mask]
print(f"Probabilities for ancilla=|1⟩: {probs_ancilla1}")

# Normalize
probs_ancilla1_norm = probs_ancilla1 / np.sum(probs_ancilla1)
print(f"Normalized Probabilities (Quantum): {probs_ancilla1_norm}")

# ─────────────────────────────────────────────
# Classical Result Comparison
# ─────────────────────────────────────────────
A_inv = np.linalg.inv(A_matrix)
x_exact = A_inv @ b_vector
c_probs = (x_exact / np.linalg.norm(x_exact)) ** 2

print(f"\n=== Result Comparison ===")
print(f"Classical solution x     = {x_exact}")
print(f"|x_n|^2 (classical)  = {c_probs}")
print(f"|<x|n>|^2 (quantum)  = {probs_ancilla1_norm}")