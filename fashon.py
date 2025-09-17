import matplotlib.pyplot as plt
import torch
import time
from activation_functions import *
from zor import Zor, Layer
from keras.datasets.fashion_mnist import load_data

# Configuration
SAMPLES_PER_CLASS = 1 # Set this to control how many examples per class

# Create a simple 3-layer autoencoder
snn = Zor([
    Layer(784, target_activation=1, learning_range=.7, activation_rate=.1, novelty_factor=.8, activation_function=leaky_relu),
    Layer(54, target_activation=1, learning_range=.7, activation_rate=0.1, novelty_factor=0, activation_function=leaky_relu), 
    Layer(784, target_activation=.5, activation_function=sigmoid, learning_range=1, activation_rate=.1, novelty_factor=0)
])

if __name__ == "__main__":
    # Load and prepare data
    (X_train, y_train), (X_test, _) = load_data()
    
    # Select SAMPLES_PER_CLASS examples from each of the 10 Fashion-MNIST classes
    train_indices = []
    for class_id in range(10):
        class_indices = torch.where(torch.tensor(y_train) == class_id)[0][:SAMPLES_PER_CLASS]
        train_indices.extend(class_indices.tolist())
    
    X_train_subset = torch.tensor(X_train[train_indices].reshape(-1, 784) / 255.0, dtype=torch.float32)
    X_val = torch.tensor(X_train[2000:3000].reshape(-1, 784) / 255.0, dtype=torch.float32)  # Validation set (1000 samples)
    
    
    # Train the autoencoder
    print("Training Zor autoencoder on Fashion-MNIST...")
    start_time = time.time()
    for epoch in range(500):
        # Simple random batch selection
        indices = torch.randperm(len(X_train_subset))[:48]
        batch = X_train_subset[indices]
        
        # Forward pass and compute error
        outputs = snn.forward(batch)
        errors = batch - outputs
        accuracy = 1.0 - torch.mean(torch.abs(errors)).item()
        
        # Learn from errors with accuracy scaling
        snn.reinforce(errors, accuracy)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: {accuracy:.1f}% accuracy")
    
    training_time = time.time() - start_time
    print(f"Training complete! Final accuracy: {accuracy:.1f}%")
    print(f"Training time: {training_time:.1f} seconds")
    
    # Validation on unseen data
    val_outputs = snn.forward(X_val, train=False)
    val_errors = X_val - val_outputs
    val_accuracy = 100.0 * (1.0 - torch.mean(torch.abs(val_errors)))
    print(f"Validation accuracy: {val_accuracy:.1f}% (on unseen samples)")
    
    # Show reconstruction examples - 2 per class from validation data
    (_, y_val_full), _ = load_data()  # Get labels for validation selection
    y_val = y_val_full[2000:3000]  # Validation labels
    
    test_indices = []
    for class_id in range(10):
        class_indices = torch.where(torch.tensor(y_val) == class_id)[0][:2]  # 2 per class
        test_indices.extend(class_indices.tolist())
    
    test_batch = X_val[test_indices]  # 20 samples total (2 per class)
    reconstructions = snn.forward(test_batch, train=False)
    
    plt.figure(figsize=(20, 4))
    plt.suptitle(f'Zor Autoencoder Results - {SAMPLES_PER_CLASS} Examples Per Class (All Unseen)\nTrained on {len(X_train_subset)} samples in {training_time:.1f}s | Validation: {val_accuracy:.1f}% on unseen data', 
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