# Matrix Product State (MPS) with Compression Layer

This notebook implements a training procedure for a Matrix Product State (MPS) model with a compression layer, following the algorithm described in the research paper [Generative Learning of Continuous Data by Tensor Networks](https://arxiv.org/abs/2310.20498). The MPS efficiently models high-dimensional data, and the compression layer reduces the dimensionality of the input data before feeding it into the MPS, improving computational efficiency.

## Overview

In this notebook, we focus on optimizing a **compression layer** to reduce the input physical dimension for MPS. The main steps include:

1. **Data Loading and Preprocessing**: Load an artificial dataset, normalize it, and reshape it for MPS representation.
2. **Apply MPS Construction**: Construct the MPS from the flattened input vectors. Each input feature gets its own tensor representation, forming a network of tensors.
3. **Compression Layer Training**: Train the compression matrices using the Procrustes problem to find an optimal transformation that reduces the physical dimension.

The compression layer helps in significantly reducing the computational cost by decreasing the feature map dimensions, making MPS optimization more practical.

## Key Components

- contract_mps_except_i Function: Contracts the MPS with compressed input data, excluding a specified site. This function computes the vector $v_{i,j}$ for a given sample $x^{(j)}$ and site $i$ by contracting the MPS with compressed input data, excluding site $i$.

Process:

Compress input data for each site:

$n \neq i$, compress the input data $x_n^{(j)}$ using the compression matrix $U_n$:
   
   $\tilde{x}_n^{(j)} = U_n^T x_n^{(j)}$

Sequentially contract the MPS tensors over the bond dimensions, skipping the contraction over the physical dimension at site $i$.

The result is a vector $v_{i,j}$, representing the MPS-contracted embedding excluding site $i$.

- **`train_compression_layer` Function**: Trains the compression matrices $U_i$ using the input data and
  the MPS. The compression matrices are updated to minimize the negative log-likelihood loss using the Procrustes problem, which is solved with **singular value decomposition (SVD)**.

  **Algorithm Overview:**
  - For each site $i$, compute $u_{i,j}$ and $v_{i,j}$, where $u_{i,j}$ is the input feature vector at site $i$ and $v_{i,j}$ is the MPS-contracted embedding excluding site $i$.
  - Use these values to compute the inner product $c_j$, calculate magnitude and phase, accumulate the negative log-likelihood loss, and update the compression matrix $U_i$ using SVD.

Future Plans

This folder will eventually contain the latest DMRG implementation techniques. The compression layer presented here is the first step, and more techniques will be added later to further enhance the efficiency and performance of the MPS and DMRG algorithms.
## Results

In our experiments with an artificial, easy dataset, we observed that the compression matrices $U_i$ initially started as random 2x2 matrices and, after a few epochs, transformed into near-identity 

$$\begin{bmatrix}
 0.9998 &  -0.0188 \\
0.0188 & 0.9998 
\end{bmatrix}$$

This demonstrates that the compression layer successfully learned an optimal transformation that preserves the input features while reducing dimensionality.

## Motivation

MPS is a powerful tool for capturing correlations in data, but large physical dimensions can make computations infeasible. The compression layer provides an effective way to make MPS more practical by reducing the input size without sacrificing too much information, allowing us to perform efficient optimization.

This notebook presents a ready solution for optimizing the input compression for MPS, showing promising results even with minimal training epochs.

Feel free to explore and experiment with the provided code!

