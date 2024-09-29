import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for clarity
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
def load_and_preprocess_mnist(num_train_per_digit=1000, num_test=1000):
    """
    Loads the MNIST dataset, downsamples images to 14x14, normalizes pixel values,
    and selects a subset of the data for training and testing.

    Args:
        num_train_per_digit (int): Number of training samples per digit (0-9).
        num_test (int): Total number of test samples.

    Returns:
        train_images_subset (np.ndarray): Training images, shape (num_train_samples, 14, 14).
        train_labels_subset (np.ndarray): Training labels, shape (num_train_samples,).
        test_images_subset (np.ndarray): Test images, shape (num_test, 14, 14).
        test_labels_subset (np.ndarray): Test labels, shape (num_test,).
    """
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Downscale images to 14x14
    train_images = downscale_images(train_images)
    test_images = downscale_images(test_images)
    
    # Normalize pixel values to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Select a subset of training data with equal samples per digit
    train_images_subset = []
    train_labels_subset = []
    for digit in range(10):
        idx = np.where(train_labels == digit)[0][:num_train_per_digit]
        train_images_subset.append(train_images[idx])
        train_labels_subset.append(train_labels[idx])
    train_images_subset = np.concatenate(train_images_subset, axis=0)
    train_labels_subset = np.concatenate(train_labels_subset, axis=0)
    
    # Select a subset of test data
    test_images_subset = test_images[:num_test]
    test_labels_subset = test_labels[:num_test]
    
    print(f"Loaded {len(train_images_subset)} training images and {len(test_images_subset)} test images.")
    return train_images_subset, train_labels_subset, test_images_subset, test_labels_subset

def downscale_images(images):
    """
    Downscales images from 28x28 to 14x14 by averaging over each 2x2 pixel block.

    Args:
        images (np.ndarray): Original images, shape (num_samples, 28, 28).

    Returns:
        downscaled_images (np.ndarray): Downscaled images, shape (num_samples, 14, 14).
    """
    # Reshape to separate 2x2 blocks and compute the mean
    downscaled_images = images.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return downscaled_images  # Shape: (num_samples, 14, 14)

# ----------------------------
# 2. Feature Mapping Function
# ----------------------------
def map_pixels_to_phi(images):
    """
    Maps pixel values to quantum spin states using φ_sj(x_j) = [cos(π x_j / 2), sin(π x_j / 2)].

    Args:
        images (np.ndarray): Normalized images, shape (num_samples, 14, 14).

    Returns:
        phi_images (np.ndarray): Mapped spin states, shape (num_samples, 14, 14, 2).
    """
    x_j = images  # Shape: (num_samples, 14, 14)
    phi_images = np.stack([np.cos(np.pi * x_j / 2), np.sin(np.pi * x_j / 2)], axis=-1)  # Shape: (num_samples, 14, 14, 2)
    return phi_images

# ----------------------------
# 3. Wavefunction Construction
# ----------------------------
def construct_wavefunctions(train_phi_images, train_labels):
    """
    Constructs and normalizes wavefunctions for each digit by summing the spin states
    of all training images corresponding to that digit.

    Args:
        train_phi_images (np.ndarray): Training spin states, shape (num_train_samples, 14, 14, 2).
        train_labels (np.ndarray): Training labels, shape (num_train_samples,).

    Returns:
        wavefunctions (dict): Dictionary mapping each digit to its normalized wavefunction vector.
    """
    wavefunctions = {}
    for digit in range(10):
        # Extract all training images corresponding to the current digit
        digit_phi = train_phi_images[train_labels == digit]
        if digit_phi.size == 0:
            print(f"No training samples found for digit {digit}.")
            wavefunctions[digit] = np.zeros((14*14*2,))
            continue
        # Sum the spin states across all training images for the digit
        digit_phi_flat = digit_phi.reshape(digit_phi.shape[0], -1)  # Shape: (num_samples, 392)
        summed_wavefunction = np.sum(digit_phi_flat, axis=0)  # Shape: (392,)
        # Normalize the wavefunction to unit length
        norm = np.linalg.norm(summed_wavefunction)
        if norm != 0:
            summed_wavefunction /= norm
        wavefunctions[digit] = summed_wavefunction
        print(f"Constructed wavefunction for digit {digit} with norm {np.linalg.norm(wavefunctions[digit]):.4f}")
    return wavefunctions

# ----------------------------
# 5. Classification Function
# ----------------------------
def classify_test_images(wavefunctions, test_phi_images):
    """
    Classifies test images based on the overlap with the wavefunction of each digit.

    Args:
        wavefunctions (dict): Dictionary mapping each digit to its normalized wavefunction vector.
        test_phi_images (np.ndarray): Test spin states, shape (num_test_samples, 14, 14, 2).

    Returns:
        predictions (list): Predicted digit labels for the test images.
    """
    num_test = test_phi_images.shape[0]
    # Flatten test images to vectors
    test_vectors = test_phi_images.reshape(num_test, -1)  # Shape: (num_test, 392)
    # Prepare wavefunction matrix: shape (10, 392)
    wavefunction_matrix = np.array([wavefunctions[digit] for digit in range(10)])  # Shape: (10, 392)
    # Compute overlaps: dot product between test vectors and each wavefunction
    overlaps = np.dot(test_vectors, wavefunction_matrix.T)  # Shape: (num_test, 10)
    # Assign each test image to the digit with the highest overlap
    predicted_digits = np.argmax(overlaps, axis=1)
    predictions = predicted_digits.tolist()
    return predictions

# ----------------------------
# 6. Evaluation Function
# ----------------------------
def evaluate_predictions(true_labels, predictions):
    """
    Evaluates the performance of the classifier.

    Args:
        true_labels (np.ndarray): True labels of test images.
        predictions (list): Predicted labels of test images.

    Returns:
        accuracy (float): Accuracy of the classifier.
        precision (float): Weighted precision of the classifier.
        recall (float): Weighted recall of the classifier.
        f1 (float): Weighted F1 score of the classifier.
        conf_mat (np.ndarray): Confusion matrix.
    """
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(true_labels, predictions)
    return accuracy, precision, recall, f1, conf_mat

# ----------------------------
# 7. Main Function
# ----------------------------
def main():
    # Load and preprocess data
    num_train_per_digit = 15000  # Number of training samples per digit
    num_test = 1000             # Number of test samples
    train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist(num_train_per_digit, num_test)
    
    # Map images to quantum spin states
    train_phi_images = map_pixels_to_phi(train_images)  # Shape: (num_train_samples, 14, 14, 2)
    test_phi_images = map_pixels_to_phi(test_images)    # Shape: (num_test, 14, 14, 2)
    
    # Construct wavefunctions for each digit
    wavefunctions = construct_wavefunctions(train_phi_images, train_labels)
    
    # Enforce sparsity on wavefunctions (Optional)
    # Uncomment the following line to apply sparsity
    # wavefunctions = enforce_sparsity(wavefunctions, sparsity_level=0.05)
    
    # Classify test images
    predictions = classify_test_images(wavefunctions, test_phi_images)
    
    # Evaluate performance
    accuracy, precision, recall, f1, conf_mat = evaluate_predictions(test_labels, predictions)
    print("\n=== Classification Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()

# ----------------------------
# 8. Run the Main Function
# ----------------------------
if __name__ == "__main__":
    main()

