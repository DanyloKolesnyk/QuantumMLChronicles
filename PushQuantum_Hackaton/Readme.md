
# Rodeo Algorithm for Molecular Spectra Estimation Using Classiq

This repository contains our winning solution for the Classiq Challenge at the Push Quantum Hackathon 2024. We present a **general solution** for the Rodeo Algorithm using the Classiq SDK, and our approach has been applied to estimate the energy eigenvalues of both the H₂ molecule and a reduced model of the H₂O molecule.

## Overview

Our solution features two key ideas:

### 1. General Implementation with the Classiq SDK

- **Modular Quantum Program:**  
  We implemented the Rodeo Algorithm using high-level abstractions provided by the Classiq SDK. Our code is organized into modular functions that encapsulate the key steps of the algorithm:
  
  - **State Preparation:**  
    Initially, we prepare the quantum state. In our early experiments, we used random initialization to form the initial superposition. This step is crucial because the quality of the initial state can directly affect the algorithm’s performance.
  
  - **Controlled Trotter Evolution:**  
    We apply the Suzuki-Trotter decomposition as a way to approximate the time evolution under a given Hamiltonian. This evolution is controlled by ancillary qubits and is essential for selectively amplifying the contributions from specific energy eigenstates.
  
  - **Phase Application:**  
    After the controlled evolution, we apply phase gates to impose energy-dependent shifts on the ancilla. This process helps to filter out unwanted components and enhance the signal corresponding to the true eigenvalues.
  
- **Quick Explanation of the Rodeo Algorithm:**  
  The Rodeo Algorithm is a quantum procedure designed to estimate the energy eigenvalues of a Hamiltonian. It works by iteratively “sweeping” through an energy interval: in each iteration, controlled time evolutions and phase shifts are applied, and after many repetitions, the statistical data reveals peaks in the probability distribution that correspond to the eigenvalues of the system.

### 2. Efficient Eigenvalue Estimation Approach

- **Initial Approach – Random State Initialization:**  
  At the outset, we initialized the quantum states randomly and applied the Rodeo Algorithm directly. While this method can work for smaller or simpler Hamiltonians, we observed that for larger systems (e.g., the H₂O molecule), random initialization leads to noisy probability distributions and less distinct energy peaks, thereby degrading the performance.

- **Refined Strategy for Complex Systems:**
  
  1. **Approximate Estimation Using a Polynomial-Scaling Algorithm:**  
     To overcome the limitations of random initialization, we first derive an approximate eigenvalue solution using an algorithm that scales polynomially with the system size. This initial approximation provides a rough—but computationally efficient—estimate of the energy landscape.
  
  2. **Search Space Reduction via the Gershgorin Circle Theorem:**  
     Next, we apply the Gershgorin Circle Theorem to compute tight bounds for the eigenvalues of the Hamiltonian. By evaluating the diagonal elements and the sum of the off-diagonal elements for each row, we obtain a reduced interval where the true eigenvalues must lie. This significantly narrows the search space.
  
  3. **Refined Energy Estimation with the Rodeo Algorithm:**  
     Finally, with the energy interval narrowed down by the Gershgorin bounds and the approximate method, we reapply the Rodeo Algorithm. Operating within this reduced and more focused interval, the algorithm accurately resolves the eigenvalues even for larger, more complex molecular systems like H₂O.
