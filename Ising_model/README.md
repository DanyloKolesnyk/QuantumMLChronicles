# Ising Model Experiments: Classical vs Quantum

Welcome to the **Ising Model** folder. Here, we explore the behavior of the **Ising spin model** through both classical and quantum approaches. This folder contains Jupyter notebooks that analyze the magnetic properties, phase transitions, and entanglement patterns of the Ising model, which is a fundamental system in statistical mechanics and condensed matter physics.

The objective is to compare classical simulations with quantum methods to determine whether leveraging quantum properties can provide an advantage over classical techniques.

## Overview

In this folder, you will find two primary notebooks:

1. **Classical Ising Model Simulation**: This notebook addresses the problem using **mean-field approximations** and **exact diagonalization** methods. The focus is on the energy spectrum, magnetization, and correlations between spins. The classical approach is reliable and provides a strong foundation for understanding the model.

2. **Quantum Ising Model Simulation**: This notebook employs **quantum computing** techniques, specifically the **Variational Quantum Eigensolver (VQE)**, to solve the Ising Hamiltonian using quantum circuits. We compare the eigenvalues obtained from quantum simulations with classical results to highlight any notable differences.

### Contents

- **Classical Ising Model**:
  - Utilizes mean-field theory for approximations.
  - Employs exact diagonalization to obtain the energy spectrum and eigenstates.
  - Provides visualizations of spin configurations, their evolution, and magnetization curves.

- **Quantum Ising Model**:
  - Uses Qiskit to simulate the Ising Hamiltonian on quantum hardware.
  - Implements the **VQE algorithm** to estimate the ground state energy and compare it with results from exact diagonalization.
  - Aims to bridge insights from both classical and quantum perspectives.

---

## Current Results

| **Method**                   | **Ground State Energy**    |
|----------------------------- |----------------------------|
| Classical Approach           | Matches analytical approx. |
| Quantum VQE (TwoLocal Ansatz)| Close to classical, with minor deviations due to noise and limitations. |

The quantum approach, while not always outperforming classical methods in terms of precision, provides insights into how quantum computation may address physics problems more effectively in the future. It also helps gauge the current capabilities of quantum hardware when applied to moderately complex problems.

---

## Motivation for Comparing Classical and Quantum Methods

Classical models are well-established and powerful, and for this situation, exact diagonalization is possible and provides reliable results. However, as the system size increases, classical diagonalization becomes computationally infeasible due to the exponential growth in the Hilbert space. Quantum simulations, in contrast, have the potential to provide efficient solutions for larger systems by leveraging quantum parallelism and entanglement. Although current quantum hardware may not yet surpass classical methods in accuracy, exploring these quantum approaches helps us understand the potential for **quantum advantage** as hardware continues to improve.

## Future Work

- **Larger Spin Chains**: Extend experiments to larger spin chains to evaluate the scaling behavior of classical and quantum techniques.
- **VQE Experimentation**: Continue experimenting with different VQE optimizers and ansatzes to determine their impact on performance and accuracy.
- **Entanglement Analysis**: Analyze how entanglement entropy changes across different phases and explore quantum indicators of critical behavior in the Ising model.

---

## Updates

This folder will be updated regularly with new experiments and findings. The focus is on expanding both the depth and breadth of our exploration of quantum-classical methods.

If you have suggestions or are interested in collaborating, please reach out.
