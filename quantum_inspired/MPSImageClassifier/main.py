import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data import load_mnist, split_train_test
from feature_mapping import feature_map, prepare_feature_mappings
from model import MPSClassifier
from train import train_step
from evaluate import evaluate

def main():
    # Define the feature mapping types to evaluate (PCA excluded)
    feature_map_types = ['trig', 'original_zero', 'polynomial']

    # Initialize a dictionary to store performance metrics
    performance_metrics = {
        'Feature Map': [],
        'Accuracy': [],
        'Precision': []
    }

    # Load and preprocess data once
    fraction = 1.0  # Use the full dataset
    train_images, train_labels = load_mnist(fraction=fraction)
    train_images, test_images, train_labels, test_labels = split_train_test(train_images, train_labels, test_size=0.2)

    # Iterate over each feature mapping type
    for fmap_type in feature_map_types:
        print(f"\n=== Feature Mapping: {fmap_type.upper()} ===")

        # Prepare feature mappings (no initialization needed since PCA is excluded)
        prepare_feature_mappings(train_images, fmap_type)

        # Determine feature dimension and n_pixels based on feature mapping
        if fmap_type in ['trig', 'polynomial', 'original_zero']:
            n_pixels = 784  # 28x28
            if fmap_type == 'trig':
                feature_dim = 2  # 'trig' mapping
            elif fmap_type == 'polynomial':
                feature_dim = 2  # Original and squared terms
            elif fmap_type == 'original_zero':
                feature_dim = 2  # Original value and zero
            else:
                feature_dim = 2  # Default to 2
        else:
            raise ValueError(f"Unsupported feature_map_type: {fmap_type}")

        # Initialize the MPS Classifier
        num_classes = 10  # Number of classes for MNIST
        bond_dim = 10      # Increased bond dimension for better expressiveness
        model = MPSClassifier(num_classes=num_classes, n_pixels=n_pixels, feature_dim=feature_dim, bond_dim=bond_dim)

        # Define the optimizer
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Training parameters
        epochs = 5  # Reduced to 5 for brevity; increase as needed
        batch_size = 32

        # Training loop for the current feature mapping
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0
            num_batches = 0
            for start_idx in range(0, len(train_images), batch_size):
                end_idx = min(start_idx + batch_size, len(train_images))
                batch_images = train_images[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                # Apply feature mapping
                feature_mapped_batch = feature_map(
                    batch_images, fmap_type
                )  # [batch_size, n_pixels, feature_dim]
                # Perform a training step
                loss = train_step(model, feature_mapped_batch, batch_labels, optimizer)
                total_loss += loss.numpy()
                num_batches += 1
                # Print loss intermittently
                if (start_idx // batch_size + 1) % 100 == 0:
                    print(f"Batch {start_idx // batch_size + 1}/{len(train_images) // batch_size}: Loss = {loss.numpy():.4f}")
            # Compute and print average loss for the epoch
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Evaluate the model on the test set
        accuracy, precision = evaluate(
            model, test_images, test_labels, fmap_type, batch_size=batch_size
        )
        print(f"Test Set Performance for {fmap_type.upper()}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

        # Store the metrics
        performance_metrics['Feature Map'].append(fmap_type.upper())
        performance_metrics['Accuracy'].append(accuracy)
        performance_metrics['Precision'].append(precision)

    # ----------------------------
    # Visualization
    # ----------------------------
    # Convert performance metrics to a DataFrame for easy plotting
    df_metrics = pd.DataFrame(performance_metrics)

    # Set Seaborn style for aesthetics
    sns.set(style="whitegrid")

    # Create a figure with two subplots: Accuracy and Precision
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot Accuracy
    sns.barplot(x='Feature Map', y='Accuracy', data=df_metrics, palette='viridis', ax=axes[0])
    axes[0].set_title('Accuracy by Feature Mapping', fontsize=20)
    axes[0].set_ylim(0, 1)  # Since accuracy ranges from 0 to 1
    axes[0].set_xlabel('Feature Mapping', fontsize=16)
    axes[0].set_ylabel('Accuracy', fontsize=16)
    for index, row in df_metrics.iterrows():
        axes[0].text(index, row['Accuracy'] + 0.01, f"{row['Accuracy']*100:.2f}%", color='black', ha="center", fontsize=14)

    # Plot Precision
    sns.barplot(x='Feature Map', y='Precision', data=df_metrics, palette='magma', ax=axes[1])
    axes[1].set_title('Precision by Feature Mapping', fontsize=20)
    axes[1].set_ylim(0, 1)  # Since precision ranges from 0 to 1
    axes[1].set_xlabel('Feature Mapping', fontsize=16)
    axes[1].set_ylabel('Precision', fontsize=16)
    for index, row in df_metrics.iterrows():
        axes[1].text(index, row['Precision'] + 0.01, f"{row['Precision']*100:.2f}%", color='black', ha="center", fontsize=14)

    plt.tight_layout()
    plt.savefig('performance_plots.png')
    plt.show()


if __name__ == "__main__":
    main()
