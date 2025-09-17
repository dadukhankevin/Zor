import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets.fashion_mnist import load_data

# Configuration
SAMPLES_PER_CLASS = 1  # Set this to control how many examples per class

class FashionMLPAutoencoder:
    def __init__(self, input_size=784, hidden_size=128, lr=1e-3):
        # Build autoencoder: 784 -> 128 -> 784 (matching Zor structure)
        self.autoencoder = keras.Sequential([
            layers.Input(shape=(input_size,)),
            layers.Dense(hidden_size, activation='relu'),
            layers.Dense(input_size, activation='sigmoid')
        ])
        
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )
    
    def forward(self, x_np):
        return self.autoencoder.predict(x_np, verbose=0)
    
    def train_step(self, x_batch, y_batch):
        loss = self.autoencoder.train_on_batch(x_batch, y_batch)
        return loss

if __name__ == "__main__":
    # Load and prepare data (same as Zor version)
    (X_train, y_train), (X_test, _) = load_data()
    
    # Select SAMPLES_PER_CLASS examples from each of the 10 Fashion-MNIST classes
    train_indices = []
    for class_id in range(10):
        class_indices = np.where(y_train == class_id)[0][:SAMPLES_PER_CLASS]
        train_indices.extend(class_indices.tolist())
    
    X_train_subset = X_train[train_indices].reshape(-1, 784) / 255.0
    X_val = X_train[2000:3000].reshape(-1, 784) / 255.0  # Validation set (1000 samples)
    
    # Create MLP autoencoder
    mlp = FashionMLPAutoencoder()
    
    # Initialize per-sample error tracking (same curriculum learning as Zor)
    sample_errors = np.ones(len(X_train_subset))
    
    # Train the autoencoder
    print("Training MLP autoencoder on Fashion-MNIST...")
    start_time = time.time()
    
    for epoch in range(200):
        # Curriculum-based batch sampling - select the worst performing samples
        indices = np.argsort(sample_errors)[-min(50, len(X_train_subset)):]
        batch = X_train_subset[indices]
        
        # Train step
        loss = mlp.train_step(batch, batch)
        
        # Forward pass and compute error (same metric as Zor)
        outputs = mlp.forward(batch)
        errors = batch - outputs
        accuracy = 100.0 * (1.0 - np.mean(np.abs(errors)))
        
        # Update per-sample error tracking
        batch_errors = np.mean(np.abs(errors), axis=1)
        for i, idx in enumerate(indices):
            # Exponential moving average of sample errors (higher = harder)
            sample_errors[idx] = 0.9 * sample_errors[idx] + 0.1 * batch_errors[i]
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: {accuracy:.1f}% accuracy")
    
    training_time = time.time() - start_time
    print(f"Training complete! Final accuracy: {accuracy:.1f}%")
    print(f"Training time: {training_time:.1f} seconds")
    
    # Validation on unseen data (same as Zor)
    val_outputs = mlp.forward(X_val)
    val_errors = X_val - val_outputs
    val_accuracy = 100.0 * (1.0 - np.mean(np.abs(val_errors)))
    print(f"Validation accuracy: {val_accuracy:.1f}% (on unseen samples)")
    
    # Show reconstruction examples - 2 per class from validation data
    (_, y_val_full), _ = load_data()
    y_val = y_val_full[2000:3000]  # Validation labels
    
    test_indices = []
    for class_id in range(10):
        class_indices = np.where(y_val == class_id)[0][:2]  # 2 per class
        test_indices.extend(class_indices.tolist())
    
    test_batch = X_val[test_indices]  # 20 samples total (2 per class)
    reconstructions = mlp.forward(test_batch)
    
    plt.figure(figsize=(20, 4))
    plt.suptitle(f'MLP Autoencoder Results - {SAMPLES_PER_CLASS} Examples Per Class (All Unseen)\nTrained on {len(X_train_subset)} samples in {training_time:.1f}s | Validation: {val_accuracy:.1f}% on unseen data', 
                 fontsize=14, y=0.98)
    
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    for i in range(20):
        class_id = i // 2
        sample_num = (i % 2) + 1
        
        # Original images
        plt.subplot(2, 20, i + 1)
        plt.imshow(test_batch[i].reshape(28, 28), cmap='gray')
        plt.title(f'{class_names[class_id]} {sample_num}', fontsize=8)
        plt.axis('off')
        
        # Reconstructed images  
        plt.subplot(2, 20, i + 21)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        plt.title(f'Recon {sample_num}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
