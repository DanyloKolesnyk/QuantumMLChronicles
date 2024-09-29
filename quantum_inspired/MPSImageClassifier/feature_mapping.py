import tensorflow as tf
import numpy as np


def feature_map(batch_x, feature_map_type):
    """
    Applies the selected feature map to a batch of images.

    Args:
        batch_x (tf.Tensor): Batch of images with shape [batch_size, height, width].
        feature_map_type (str): Type of feature mapping ('trig', 'polynomial', 'original_zero').

    Returns:
        tf.Tensor: Feature-mapped images with shape [batch_size, n_pixels, feature_dim].
    """
    if feature_map_type == 'trig':
        # Trigonometric Feature Mapping
        normalized = (batch_x + 1) / 2  # Scale from [-1,1] to [0,1]
        cos_component = tf.cos(np.pi * normalized)
        sin_component = tf.sin(np.pi * normalized)
        feature_mapped = tf.stack([cos_component, sin_component], axis=-1)  # [batch_size, height, width, 2]
        feature_mapped = tf.reshape(feature_mapped, [batch_x.shape[0], -1, 2])  # [batch_size,784,2]
        return feature_mapped

    elif feature_map_type == 'polynomial':
        # Polynomial Feature Mapping (e.g., degree 2)
        flattened = tf.reshape(batch_x, [batch_x.shape[0], -1])  # [batch_size, 784]
        poly_features = tf.concat([flattened, tf.square(flattened)], axis=-1)  # [batch_size, 784*2]
        feature_dim = 2  # Original and squared terms
        feature_mapped = tf.reshape(poly_features, [batch_x.shape[0], -1, feature_dim])  # [batch_size,784,2]
        return feature_mapped

    elif feature_map_type == 'original_zero':
        # Original Value and Zero Feature Mapping
        # Each pixel is mapped to [x, 0], where x is the original normalized value
        flattened = tf.reshape(batch_x, [batch_x.shape[0], -1])  # [batch_size, 784]
        zeros = tf.zeros_like(flattened)  # [batch_size, 784]
        feature_mapped = tf.stack([flattened, zeros], axis=-1)  # [batch_size, 784, 2]
        return feature_mapped

    else:
        raise ValueError(f"Unsupported feature_map_type: {feature_map_type}")


def prepare_feature_mappings(train_images, feature_map_type):
    """
    Prepares feature mappings based on the selected type.

    Args:
        train_images (tf.Tensor): Training images with shape [num_train, height, width].
        feature_map_type (str): Type of feature mapping ('trig', 'polynomial', 'original_zero').

    Returns:
        tuple: Empty tuple, as no initialization is needed for the current feature mappings.
    """
    # Since PCA is excluded, no initialization is needed
    print(f"No initialization needed for {feature_map_type.upper()} feature mapping.")
    return ()
