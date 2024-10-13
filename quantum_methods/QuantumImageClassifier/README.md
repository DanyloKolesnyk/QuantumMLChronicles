# QuantumClassifier 🌀

The **QuantumClassifier** project is focused on bridging the gap between quantum and classical machine learning methods. We are exploring how quantum computing can complement classical approaches to solve machine learning problems more efficiently.

## 🎯 Current Status: Significant Progress

At this stage, the classifier is functional and has achieved a **95% accuracy** on the test set, leveraging **Qiskit** for quantum data encoding and a simple parameterized ansatz. This improvement is a huge leap from our initial attempts.

### Test Accuracy: **95%**

Currently, we're achieving **95%** test accuracy, which is a promising indicator of the classifier's capability. This accuracy was achieved by improving the feature map and making careful adjustments to the quantum circuit parameters. Initially, we were working with heavily downgraded data and a basic feature map, which led to poor performance. However, with the latest optimizations, the model has made substantial progress.

The feature map we’re using for now is as follows:

```python
def parameterized_scaled_angle_encoding(num_qubits, scale_factor=np.pi):
    """
    Parameterized Angle Encoding with Feature Scaling.

    Parameters:
    - num_qubits (int): Number of qubits in the circuit.
    - scale_factor (float): Scaling factor to adjust rotation angles.

    Returns:
    - QuantumCircuit: Parameterized quantum circuit implementing scaled angle encoding.
    """
    qr = QuantumRegister(num_qubits, name='q')
    circuit = QuantumCircuit(qr)
    data_parameters = ParameterVector('x', num_qubits)
    for i in range(num_qubits):
        angle = data_parameters[i] * scale_factor
        circuit.ry(angle, qr[i])
    return circuit
```

### Challenges We're Facing

The initial challenges boiled down to a combination of **poor data quality** (downscaled images) and a **basic feature map**. As we progressed, we managed to mitigate these issues by enhancing the feature map and increasing the **dimensionality of the data representation**. The current version utilizes more qubits and a more refined encoding, allowing the classifier to extract more meaningful patterns from the data.

The ansatz used for parameterized quantum operations is shown below:

```python
def parameterized_ansatz(num_qubits, parameters):
    """
    Parameterized Ansatz.

    Constructs a Parameterized Quantum Circuit (PQC) with RY and RZ gates and entangling CX gates.

    Parameters:
    - num_qubits (int): Number of qubits in the circuit.
    - parameters (ParameterVector): List of parameters for rotation gates.

    Returns:
    - QuantumCircuit: Parameterized quantum circuit implementing the ansatz.
    """
    qr = QuantumRegister(num_qubits, name='q')
    circuit = QuantumCircuit(qr)
    param_idx = 0
    for i in range(num_qubits):
        circuit.ry(parameters[param_idx], qr[i])
        param_idx += 1
        circuit.rz(parameters[param_idx], qr[i])
        param_idx += 1
    # Entangling layer
    for i in range(num_qubits - 1):
        circuit.cx(qr[i], qr[i + 1])
    return circuit
```

### Why Amplitude Encoding?

For the current implementation, we chose **amplitude encoding** primarily because it allows us to represent high-dimensional data with a relatively small number of qubits. Using **amplitude encoding**, we can efficiently represent our data using only **10 qubits**, which significantly reduces the computational load. 
However, as we move towards running on **real quantum hardware**, we need to rethink our approach to the feature map. **Amplitude encoding** requires highly entangled states, which can be challenging for **NISQ (Noisy Intermediate-Scale Quantum)** devices to handle efficiently. Thus, our next step will be to experiment with feature maps that are more hardware-friendly, allowing for better performance under the noise constraints of current quantum hardware.

### Why SVM for Classification?

The choice of an **SVM (Support Vector Machine)** was motivated by its ability to work well with **kernel methods**, which we implemented using a **quantum kernel**. The SVM can leverage the quantum kernel to implicitly map the data into a higher-dimensional feature space, providing a non-linear decision boundary that might be harder to achieve with classical models. Given our current focus on kernel methods and quantum state overlaps, an SVM was the most straightforward tool for exploring how quantum-enhanced kernels affect classification performance.

Other methods like **neural networks** or **decision trees** might offer different strengths, but integrating them with quantum kernel methods is a more complex and experimental process. For now, the SVM provides a clear and interpretable approach to understanding the quantum advantage in machine learning.

## 🚀 The Plan for the Future

Moving forward, we plan to focus on:

- **Using Matrix Product States (MPS) for Data Representation**: We intend to leverage **MPS construction** to efficiently fit data into any number of qubits, making our quantum circuits more adaptable to the available quantum hardware.
- **Testing on Real Quantum Hardware**: Running these quantum circuits on actual quantum devices will be a significant milestone. We expect differences between simulations and real hardware due to noise and other quantum effects, so adapting the circuits for practical use will be a key focus.
