import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score
from feature_mapping import feature_map


def predict(model, image_mps):
    """
    Predicts the class label for a single image.

    Args:
        model (MPSClassifier): The MPS Classifier model.
        image_mps (tf.Tensor): MPS tensor of the input image with shape [n_pixels, feature_dim].

    Returns:
        int: Predicted class label.
    """
    # Expand dimensions to form a batch of size 1
    image_mps_batch = tf.expand_dims(image_mps, axis=0)  # [1, n_pixels, feature_dim]
    logits = model(image_mps_batch, training=False)  # [1, num_classes]
    prediction = tf.argmax(logits, axis=-1)
    return prediction.numpy()[0]


def evaluate(model, test_images, test_labels, feature_map_type, batch_size=32):
    """
    Evaluates the model on the test dataset.

    Args:
        model (MPSClassifier): The MPS Classifier model.
        test_images (tf.Tensor): Test images with shape [num_test, height, width].
        test_labels (tf.Tensor): True labels for test images with shape [num_test].
        feature_map_type (str): Type of feature mapping used.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (accuracy, precision)
    """
    num_samples = len(test_images)
    predictions = []
    true_labels = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        # Apply feature mapping
        feature_mapped_batch = feature_map(
            batch_images, feature_map_type
        )  # [batch_size, n_pixels, feature_dim]
        # Forward pass: [batch_size, num_classes]
        logits = model(feature_mapped_batch, training=False)  # [batch_size, num_classes]
        # Predictions: [batch_size]
        preds = tf.argmax(logits, axis=-1).numpy()
        predictions.extend(preds)
        true_labels.extend(batch_labels.numpy())
    # Compute performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision
