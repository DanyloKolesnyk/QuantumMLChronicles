# MPS Image Classifier

This project implements an **Image Classifier** based on **Matrix Product States (MPS)**, a type of tensor network, to efficiently perform image classification on the MNIST dataset. By leveraging tensor networks, this approach reduces the computational complexity typically required for processing high-dimensional data.

## Overview

The **MPS Image Classifier** applies various feature mapping techniques to transform pixel data from images and then uses the MPS architecture to classify these images. The project explores different feature mappings, evaluates their impact on classification performance, and visualizes the results.

## Features

- **Feature Mapping Techniques**:
  - **Trigonometric Mapping**: Uses sine and cosine transformations to capture periodic patterns in the data.
  - **Polynomial Mapping**: Applies polynomial expansions to enrich the feature space.
  - **Original-Zero Mapping**: Maps each pixel value to a two-dimensional space including the pixel value and zero, preserving simplicity.
  
- **MPS Architecture**: The MPS model processes the feature-mapped data, allowing for efficient image classification.

- **Training and Evaluation**:
  - Implements training loops with cross-entropy loss.
  - Provides evaluation metrics such as accuracy to gauge performance across feature mappings.
  
- **Visualization**: Generates performance plots to compare the effectiveness of different feature mappings.

## Installation

To set up the environment and run the code:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DanyloKolesnyk/MPSImageClassifier.git
   cd MPSImageClassifier
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Classifier

To train the MPS classifier with different feature mappings, run the following script:

```bash
python src/main.py
```

You can configure parameters like the feature mapping type, number of epochs, batch size, and bond dimension within the script.

### Evaluating and Visualizing Results

Once training is complete, the model evaluates its performance on the test dataset and generates performance metrics. Results and visualizations are saved as image files in the root directory.

```bash
python src/main.py
```

## Results

The project compares the classifier's performance across feature mappings, showcasing the ability of MPS to achieve competitive accuracy on the MNIST dataset. Polynomial mapping provides the best accuracy, illustrating the benefits of richer feature spaces.

| **Feature Mapping** | **Accuracy** |
|---------------------|--------------|
| Trigonometric (TRIG)| 94.00%       |
| Original-Zero (OZ)  | 93.50%       |
| Polynomial (POLY)   | 96.20%       |

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by research on tensor networks in machine learning.
- Uses TensorFlow and Scikit-learn for implementation.
- The MNIST dataset is provided by [Yann LeCun's MNIST database](http://yann.lecun.com/exdb/mnist/).
