import tensorflow as tf


def initialize_classifier_mps(num_classes, n_pixels, feature_dim, bond_dim=10):
    """
    Initializes a Classifier MPS with a specified bond dimension.

    Args:
        num_classes (int): Number of output classes.
        n_pixels (int): Number of pixels in each image.
        feature_dim (int): Dimension of the feature mapping.
        bond_dim (int): Bond dimension.

    Returns:
        tf.Variable: Classifier MPS tensor with shape [num_classes, n_pixels, feature_dim, bond_dim].
    """
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.Variable(initializer([num_classes, n_pixels, feature_dim, bond_dim]), trainable=True)


class MPSClassifier(tf.Module):
    def __init__(self, num_classes, n_pixels, feature_dim, bond_dim=10):
        """
        Initializes the MPS Classifier.

        Args:
            num_classes (int): Number of output classes (default is 10 for MNIST).
            n_pixels (int): Number of pixels in each image (28*28=784 for MNIST).
            feature_dim (int): Dimension of the feature mapping.
            bond_dim (int): Bond dimension.
        """
        super(MPSClassifier, self).__init__()
        self.num_classes = num_classes
        self.n_pixels = n_pixels
        self.feature_dim = feature_dim
        self.bond_dim = bond_dim
        # Initialize Classifier MPS: [num_classes, n_pixels, feature_dim, bond_dim]
        self.classifier_mps = initialize_classifier_mps(num_classes, n_pixels, feature_dim, bond_dim)

    def __call__(self, image_mps_batch, training=False):
        """
        Performs the forward pass by contracting the image MPS with the Classifier MPS.

        Args:
            image_mps_batch (tf.Tensor): Batch of image MPS with shape [batch_size, n_pixels, feature_dim].
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            tf.Tensor: Logits for each class with shape [batch_size, num_classes].
        """
        # Expand dimensions for broadcasting
        # classifier_mps: [num_classes, n_pixels, feature_dim, bond_dim]
        # Expand to [1, num_classes, n_pixels, feature_dim, bond_dim]
        classifier_mps_exp = tf.expand_dims(self.classifier_mps, axis=0)  # [1, C, P, D, BD]

        # image_mps_batch: [batch_size, n_pixels, feature_dim]
        # Expand to [batch_size,1,n_pixels,feature_dim,1] to align with bond_dim=10
        image_mps_exp = tf.expand_dims(image_mps_batch, axis=1)  # [B,1,P,D]
        image_mps_exp = tf.expand_dims(image_mps_exp, axis=-1)  # [B,1,P,D,1]

        # Element-wise multiplication: [B, C, P, D, BD]
        multiplied = classifier_mps_exp * image_mps_exp  # Broadcasting

        # Sum over P, D, BD to get logits: [B, C]
        logits = tf.reduce_sum(multiplied, axis=[2, 3, 4])  # [B, C]

        return logits  # [B, C]
