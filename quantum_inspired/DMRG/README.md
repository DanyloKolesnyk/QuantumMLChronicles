# Generative AI with Density Matrix Renormalization Group (DMRG) and Matrix Product States (MPS)

This project leverages Density Matrix Renormalization Group (DMRG) techniques combined with Matrix Product States (MPS) for generative modeling using the Bars and Stripes dataset. The approach focuses on efficiently modeling high-dimensional data by integrating a compression layer that reduces input dimensionality, thereby enhancing computational efficiency.

## Overview

In this project, we aim to optimize a **compression layer** to minimize the input physical dimension for the MPS model. The primary workflow includes:

1. **Data Generation and Visualization**: Generate all unique Bars and Stripes patterns for a specified grid size and visualize them.
2. **Data Preprocessing**: Preprocess the generated images by replacing zero values with small random values and applying feature maps using cosine and sine functions.
3. **MPS Initialization and Orthonormalization**: Initialize the MPS tensors with random values and orthonormalize them using QR decomposition to ensure numerical stability.
4. **Training Functions**: Implement functions for contracting the MPS with input data, computing norms, and calculating gradients essential for training.
5. **Compression Layer Training**: Train the compression matrices to optimally reduce the input dimensionality using the Procrustes problem, solved via Singular Value Decomposition (SVD).
6. **Training Loop with DMRG Sweeps**: Perform iterative sweeps (left-to-right and right-to-left) to optimize the MPS tensors, incorporating gradient clipping to prevent exploding gradients.
7. **Result Visualization**: Plot the loss values over training iterations to monitor convergence and performance.

The integration of a compression layer significantly reduces computational costs by decreasing feature map dimensions, making MPS optimization more practical and efficient.

## Key Components

### 1. Data Generation and Visualization

- **Bars and Stripes Dataset**: Generates all unique binary configurations for a specified grid size, focusing on bars (row-wise patterns) and stripes (column-wise patterns).
- **Visualization**: Displays the generated patterns to provide a clear understanding of the dataset's structure.

### 2. Data Preprocessing

- **Replacement of Zeros**: Replaces zero values in the images with small random values to stabilize training and prevent issues related to zero gradients.

- **Feature Mapping**: Applies cosine and sine transformations to the preprocessed images to enhance feature representation.

  $$
  \phi(x) = \begin{bmatrix}
    \cos(\pi x) \\
    \sin(\pi x)
  \end{bmatrix}
  $$

### 3. Matrix Product States (MPS) Initialization and Orthonormalization

- **MPS Tensors Initialization**: Initializes the MPS tensors with random values scaled by the inverse of the total number of sites to ensure appropriate scaling.

- **Orthonormalization**: Utilizes QR decomposition to orthonormalize the MPS tensors, ensuring that the MPS maintains numerical stability during training.

  - **Local Orthonormalization**: Right-orthonormalizes individual tensors.
  - **Global Orthonormalization**: Applies local orthonormalization across all tensors in the MPS.

### 4. Training Functions

- **Contraction Functions**: Implements functions to contract the MPS with input data, excluding specific sites as needed to compute vectors representing the MPS-contracted embeddings.
- **Norm Computation**: Calculates the norm of the MPS to ensure proper scaling during loss computation.

- **Gradient Calculation**: Defines the loss function based on negative log-likelihood and computes gradients for MPS tensors to facilitate optimization.

### 5. Compression Layer Training

- **Compression Matrices ($U_i$)**: Trains the compression matrices to optimally reduce the input dimensionality while preserving essential information.
  
  **Algorithm Overview:**
  
  1. For each site $i$, compute $u_{i,j}$ (input feature vector) and $v_{i,j}$ (MPS-contracted embedding excluding site $i$).
  2. Calculate the inner product $c_j = u_{i,j}^T U_i v_{i,j}$.
  3. Determine the magnitude $p_j = |c_j|$ and phase $\phi_j = \frac{c_j}{p_j}$.
  4. Accumulate the negative log-likelihood (NLL) loss over all samples.
  5. Update the compression matrix $U_i$ by solving the Procrustes problem using Singular Value Decomposition (SVD).
  
  $$
  U_i = \text{argmin}_{U} \sum_j -\log\left(\frac{|c_j|^2}{\| \text{MPS} \|^2}\right)
  $$

### 6. Training Loop with DMRG Sweeps

- **DMRG Sweeps**: Executes iterative left-to-right and right-to-left sweeps to optimize the MPS tensors.
- **Gradient Clipping**: Applies gradient clipping techniques to prevent exploding gradients during the optimization process.

### 7. Result Visualization

- **Loss Plotting**: Visualizes the loss values over training iterations to monitor convergence and assess the model's performance.

![output](https://github.com/user-attachments/assets/3850f9be-4948-4c37-8c8f-d5b9dc664a1f)

  
  *Figure: Loss over DMRG Iterations.*

