import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings

# Suppress TensorFlow warnings for clarity
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def load_mnist(fraction=1.0):
    """
    Loads and preprocesses the MNIST dataset.

    Args:
        fraction (float): Fraction of the dataset to load (default is 1.0 for full dataset).

    Returns:
        tuple: Tuple containing training images and labels.
    """
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    total_train = int(fraction * len(train_images))
    train_images = tf.convert_to_tensor(train_images[:total_train], dtype=tf.float32)
    # Normalize to [-1, 1]
    train_images = (train_images / 255.0 - 0.5) * 2
    train_labels = tf.convert_to_tensor(train_labels[:total_train], dtype=tf.int32)
    print(f"Loaded {total_train} training images with shape {train_images.shape}")
    return train_images, train_labels


def split_train_test(train_images, train_labels, test_size=0.2):
    """
    Splits the training data into training and testing sets.

    Args:
        train_images (tf.Tensor): Training images with shape [num_train, height, width].
        train_labels (tf.Tensor): Training labels with shape [num_train].
        test_size (float): Fraction of data to use for testing (default is 0.2).

    Returns:
        tuple: Tuple containing training and testing images and labels.
    """
    train_images_np = train_images.numpy()
    train_labels_np = train_labels.numpy()
    train_images_np, test_images_np, train_labels_np, test_labels_np = train_test_split(
        train_images_np, train_labels_np, test_size=test_size, random_state=42, shuffle=True
    )
    train_images_tf = tf.convert_to_tensor(train_images_np, dtype=tf.float32)
    test_images_tf = tf.convert_to_tensor(test_images_np, dtype=tf.float32)
    train_labels_tf = tf.convert_to_tensor(train_labels_np, dtype=tf.int32)
    test_labels_tf = tf.convert_to_tensor(test_labels_np, dtype=tf.int32)
    print(f"Split data into {len(train_images_tf)} training and {len(test_images_tf)} testing samples.")
    return train_images_tf, test_images_tf, train_labels_tf, test_labels_tf
