# ============================================================
#  HHL Algorithm – Qiskit Implementation
#  Chuyển đổi từ PennyLane sang Qiskit
# ============================================================

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Thư mục lưu kết quả (tự động tạo nếu chưa có)
# ─────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────
#  1. Tham số toàn cục
# ─────────────────────────────────────────────
n_qubits      = 1          # Số qubit hệ thống
phase_qubits  = 8          # Số qubit ước lượng pha (QPE)
ancilla_qubit = 1          # Số qubit ancilla
tot_qubits    = n_qubits + phase_qubits + ancilla_qubit

# Ma trận A (Hermitian)
A_matrix = np.array([[-1.5, 0.5],
                      [ 0.5,-1.5]], dtype=np.complex128)
A_matrix = -A_matrix          # lấy -A như trong PennyLane

# Vector b
b_vector = np.array([-2, 1])
norm_b   = b_vector / np.linalg.norm(b_vector)

print("=== Thông số bài toán ===")
print(f"A =\n{A_matrix}")
print(f"b = {b_vector}")
print(f"||b|| = {np.linalg.norm(b_vector):.6f}")
print(f"b̂ (chuẩn hoá) = {norm_b}")

# ─────────────────────────────────────────────
#  2. U = exp(iA)
# ─────────────────────────────────────────────
U_matrix = expm(1j * A_matrix)
assert np.allclose(U_matrix.conj().T @ U_matrix, np.eye(2**n_qubits)), \
    "U không phải unitary!"
print(f"\nU = exp(iA) =\n{np.round(U_matrix, 4)}")

# ─────────────────────────────────────────────
#  3. Controlled-U^(2^k)
# ─────────────────────────────────────────────
def controlled_unitary_power(qc, U, power, control_qubit, target_qubits):
    """Áp dụng controlled-U^power lên target_qubits với control_qubit."""
    U_power = np.linalg.matrix_power(U, power)
    gate = QuantumCircuit(len(target_qubits), name=f"U^{power}")
    gate.unitary(U_power, list(range(len(target_qubits))))
    controlled_gate = gate.to_gate().control(1)
    qc.append(controlled_gate, [control_qubit] + list(target_qubits))

# ─────────────────────────────────────────────
#  4. QPE (Quantum Phase Estimation)
# ─────────────────────────────────────────────
def qpe_circuit(qc, phase_reg, target_reg):
    # Hadamard lên tất cả qubit pha
    for w in phase_reg:
        qc.h(w)

    # Controlled-U^{2^k} — lưu ý: Qiskit đánh số LSB ngược
    # phase_reg[0] là qubit pha thứ 0 (MSB trong QPE PennyLane -> đảo)
    for i, ctrl in enumerate(reversed(phase_reg)):
        power = 2 ** i
        controlled_unitary_power(qc, U_matrix, power, ctrl, target_reg)

    # Inverse QFT
    iqft = QFT(num_qubits=len(phase_reg), inverse=True, do_swaps=True)
    qc.append(iqft, phase_reg)

# ─────────────────────────────────────────────
#  5. Controlled RY rotation (eigenvalue encoding)
# ─────────────────────────────────────────────
def control_rotation_gate(qc, control_reg, target_qubit):
    n = len(control_reg)
    for d in range(1, 2**n):            # bỏ qua d=0
        bin_str  = f"{d:0{n}b}"
        bit_list = [int(b) for b in bin_str]
        # Giá trị phân số: sum x_j * 2^{-(j+1)}
        frac_d = sum(int(b) * 2**(-(j+1)) for j, b in enumerate(bin_str))
        inv    = 1.0 / frac_d
        inv    = inv / (2 ** n)         # chuẩn hoá về [0,1]
        theta  = 2 * np.arccos(inv)     # RY(theta)

        # Xây dựng cổng multi-controlled RY
        ctrl_state = "".join(str(b) for b in bit_list)
        # Qiskit ctrl_state: chuỗi bit tương ứng với thứ tự control_reg
        mcry = QuantumCircuit(n + 1, name=f"RY|{ctrl_state}⟩")
        from qiskit.circuit.library import RYGate
        cry = RYGate(theta).control(n, ctrl_state=ctrl_state)
        mcry.append(cry, list(range(n + 1)))
        qc.append(mcry, list(control_reg) + [target_qubit])

# ─────────────────────────────────────────────
#  6. Mạch HHL đầy đủ
# ─────────────────────────────────────────────

phase_reg   = QuantumRegister(phase_qubits,  name='phase')
target_reg  = QuantumRegister(n_qubits,      name='target')
ancilla_reg = QuantumRegister(ancilla_qubit, name='ancilla')

qc_hhl = QuantumCircuit(phase_reg, target_reg, ancilla_reg, name="HHL")

# Chỉ số qubit tuyệt đối
phase_indices   = list(range(phase_qubits))
target_indices  = list(range(phase_qubits, phase_qubits + n_qubits))
ancilla_indices = list(range(phase_qubits + n_qubits, tot_qubits))

# ── Bước 1: Khởi tạo |b⟩ trên target qubit ──
qc_hhl.initialize(norm_b, target_indices)

# ── Bước 2: QPE ──
qc_hhl.barrier(label="QPE")
qpe_circuit(qc_hhl, phase_indices, target_indices)

# ── Bước 3: Controlled RY ──
qc_hhl.barrier(label="AQE")
control_rotation_gate(qc_hhl, phase_indices, ancilla_indices[0])

# ── Bước 4: Inverse QPE (adjoint) ──
qc_hhl.barrier(label="QPE†")
# Xây dựng QPE riêng rồi lấy inverse
qc_qpe_only = QuantumCircuit(phase_qubits + n_qubits, name="QPE")
qpe_circuit(qc_qpe_only, list(range(phase_qubits)), list(range(phase_qubits, phase_qubits + n_qubits)))
inv_qpe = qc_qpe_only.inverse()
inv_qpe.name = "QPE†"
qc_hhl.append(inv_qpe, phase_indices + target_indices)

qc_hhl.barrier()


# ─────────────────────────────────────────────
#  7. Vẽ mạch
# ─────────────────────────────────────────────

fig_main, ax_main = plt.subplots(figsize=(20, 6))
qc_hhl.decompose().draw(output='mpl', ax=ax_main,
                         style={'backgroundcolor': '#EEEEEE'},
                         fold=60)
ax_main.set_title("HHL Circuit (decompose level 1)", fontsize=14)
plt.tight_layout()
plt.savefig(out("hhl_circuit_main.png"), dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
#  8. Mô phỏng bằng Statevector
# ─────────────────────────────────────────────
print("\n=== Chạy mô phỏng Statevector ===")
sv = Statevector.from_instruction(qc_hhl)

# Lấy xác suất trên (target + ancilla)
# Qiskit đánh số qubit ngược: qubit thấp nhất ở bên phải chuỗi bit
# Thứ tự: [phase(3..0), target(0), ancilla(0)] -> tot_qubits bits
# Ta chỉ lấy target + ancilla, tức là 2 qubit cuối cùng trong qargs
# Số qubit cho target + ancilla
n_meas = n_qubits + ancilla_qubit           # = 2
# Qubit index trong sv: ancilla = 0, target = 1 (Qiskit little-endian)
# Trace out phase register bằng cách sum over phase states
probs_ta = sv.probabilities(qargs=target_indices + ancilla_indices)

print(f"Xác suất (target ⊗ ancilla) = {probs_ta}")

# Thứ tự bit trong probs_ta (little-endian Qiskit):
# index 0 = |ancilla=0, target=0⟩  -> |00⟩
# index 1 = |ancilla=0, target=1⟩  -> |01⟩  (ancilla bit=0)
# index 2 = |ancilla=1, target=0⟩  -> |10⟩  (ancilla bit=1)
# index 3 = |ancilla=1, target=1⟩  -> |11⟩

# Lọc những trạng thái có ancilla = 0 (bit cao = 0, tức index 0 và 1)
# Trong little-endian: ancilla là qubit đầu tiên trong qargs
# -> ancilla_bit = (index >> 1) & 1
ancilla_zero_mask = [(i >> n_qubits) & 1 == 0 for i in range(len(probs_ta))]
probs_ancilla0 = probs_ta[ancilla_zero_mask]
print(f"Xác suất khi ancilla=|0⟩: {probs_ancilla0}")

# Chuẩn hoá
probs_ancilla0_norm = probs_ancilla0 / np.sum(probs_ancilla0)
print(f"Xác suất chuẩn hoá (quantum): {probs_ancilla0_norm}")

# ─────────────────────────────────────────────
#  9. Giải cổ điển để so sánh
# ─────────────────────────────────────────────
A_inv   = np.linalg.inv(A_matrix)
x_exact = A_inv @ b_vector
c_probs = (x_exact / np.linalg.norm(x_exact)) ** 2

print(f"\n=== So sánh kết quả ===")
print(f"Nghiệm cổ điển x     = {x_exact}")
print(f"|x_n|^2 (classical)  = {c_probs}")
print(f"|<x|n>|^2 (quantum)  = {probs_ancilla0_norm}")

