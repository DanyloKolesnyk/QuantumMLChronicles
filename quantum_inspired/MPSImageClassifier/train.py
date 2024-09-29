import tensorflow as tf


def train_step(model, image_mps_batch, labels, optimizer):
    """
    Performs a single training step.

    Args:
        model (MPSClassifier): The MPS Classifier model.
        image_mps_batch (tf.Tensor): Batch of image MPS with shape [batch_size, n_pixels, feature_dim].
        labels (tf.Tensor): True labels for the batch with shape [batch_size].
        optimizer (tf.optimizers.Optimizer): Optimizer for training.

    Returns:
        tf.Tensor: The computed loss for the batch.
    """
    with tf.GradientTape() as tape:
        # Forward pass: [batch_size, num_classes]
        logits = model(image_mps_batch, training=True)  # [batch_size, num_classes]
        # Compute cross-entropy loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update parameters
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
