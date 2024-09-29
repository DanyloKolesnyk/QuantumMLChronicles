# QuantumClassifier 🌀

Welcome to the **QuantumClassifier**, where we attempt to push the boundaries of machine learning on quantum hardware, only to discover that sometimes those boundaries push back... hard. Let’s just say, the journey from classical to quantum isn’t always smooth, but hey, we’re learning!

## 🎯 Current Status: A Very Humble Beginning

So, here’s where we’re at right now. The classifier is up and running, using **Qiskit** for quantum data encoding and a simple parameterized ansatz, but there’s a bit of a catch... We’ve had to **downgrade the MNIST images from 28x28 pixels to a whopping 3x3 pixels** (yup, you read that right 😅). If you’re thinking "How can anything work with 3x3 pixel images?"—well, you’d be correct. It doesn’t work so great.

### Test Accuracy: **12.50%**

That’s right, folks—**12.50%**. You can flip a coin and probably do better, but we’re running quantum circuits, so that’s kind of cool, right? The poor accuracy is mainly due to the harsh data downgrading and the fact that our **feature map** isn’t doing us any favors either. Here’s the basic feature map we’re using:

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

### What’s Going Wrong?

Well, we’ve got **bad data** (3x3 pixels is barely recognizable as an image) and a **less-than-optimal feature map** that essentially just rotates the vector before passing it into a simple ansatz:

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

### Why So Humble?

Quantum hardware is **super powerful**, but right now, we’re **limited to simulations** on classical hardware, and simulating more than a handful of qubits is like trying to squeeze an elephant into a phone booth—just not gonna happen. That’s why we’re working with only **9 qubits** and an **extremely simplified feature map**, which obviously leaves room for improvement.

## 🚀 The Plan for the Future

But don’t worry, we’ve got big plans! This is just a **starting point**—and yes, it’s probably the **worst possible starting point**—but it’s important to lay down the groundwork:

- **Improved Feature Maps**: We’ll try more sophisticated encodings to see if we can squeeze out more accuracy.
- **Better Ansatz Designs**: The current ansatz is pretty basic. We’ll experiment with more complex and hardware-efficient versions.
- **Testing on Real Quantum Hardware**: Simulations are one thing, but once we run these circuits on **actual quantum computers**, things could get a lot more interesting. 

One thing to note: **hardware-efficient ansätze** are likely going to differ significantly from the ones that show the best results during simulations, so we’ll be testing and refining everything to ensure the circuits work well on real quantum devices.
