
# HHL Algorithm – Qiskit Implementation

This repository provides an implementation of the **HHL (Harrow-Hassidim-Lloyd) Algorithm** for solving linear systems using quantum computing, specifically with **Qiskit**. The algorithm is implemented by transforming the original PennyLane-based approach into Qiskit.

## Problem Overview

The HHL algorithm solves a system of linear equations \( Ax = b \), where \( A \) is a Hermitian matrix, and \( b \) is a known vector. The algorithm uses quantum circuits to approximate the solution \( x \) through quantum phase estimation and controlled rotations.

## Requirements

To run the code, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Key Dependencies:

- **Qiskit**: Quantum computing framework
- **NumPy**: For numerical operations
- **SciPy**: For matrix exponentiation and other scientific computations
- **Matplotlib**: For plotting the quantum circuit and results
- **Qiskit-Aer**: For simulating quantum circuits using statevector simulations

## Running the Code

To execute the algorithm and simulate the quantum circuit, run the following command:

```bash
python hhl_algorithm.py
```

The script will output the quantum circuit, simulate it using Qiskit’s Statevector simulator, and print the probabilities of different states. The result is compared to the classical solution obtained by directly solving the linear system.

## Output

- **Quantum Circuit Visualization**: The quantum circuit for the HHL algorithm, showing the steps of quantum phase estimation, controlled rotations, and inverse QPE.
- **Quantum Probabilities**: The probabilities of different states from the quantum simulation.
- **Classical Solution**: The classical solution \( x = A^{-1}b \) for comparison.

### Example Output:

- **Quantum Probabilities (Ancilla = 1)**: The probability distribution for the quantum state corresponding to the ancilla qubit being in state |1⟩.
- **Classical Solution**: The exact classical solution to the linear system.

## File Structure

The repository contains the following files:

- `hhl_algorithm.py`: The main Python script implementing the HHL algorithm using Qiskit.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file, providing an overview of the project.

## Quantum Circuit Description

The HHL algorithm is implemented using the following steps:

1. **Quantum Phase Estimation (QPE)**: This step estimates the phase of the eigenvalues of the matrix \( A \).
2. **Controlled RY Rotations**: Eigenvalue encoding is performed using controlled RY rotations, based on the result of the phase estimation.
3. **Inverse QPE**: The inverse of the QPE step is applied to retrieve the solution from the quantum state.

## Classical vs Quantum Comparison

The quantum solution is compared against the classical solution, which is computed by directly solving \( Ax = b \) using the matrix inverse. The comparison includes the absolute and relative error between the two solutions.

## Future Improvements

- **Error Mitigation**: Implementing error mitigation techniques to address noise in quantum circuits.
- **Scalability**: Extending the algorithm to work with larger systems and more qubits.
- **Optimization**: Investigating other quantum circuits or algorithms to improve the efficiency of solving linear systems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
