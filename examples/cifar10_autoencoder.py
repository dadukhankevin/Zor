import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zor import Zor, Layer
from activation_functions import *
import torch
import torch.optim as optim
from keras.datasets import cifar10
import time
import matplotlib.pyplot as plt
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Set device to MPS if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def psnr(y_true, y_pred, eps=1e-8):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse + eps)
    return float(20.0 * np.log10(1.0 / (rmse + eps)))

(X_train, _), (X_test, _) = cifar10.load_data()
X_train_full = torch.tensor(X_train.reshape(-1, 3072) / 255.0, dtype=torch.float16, device=device)
X_test_full = torch.tensor(X_test.reshape(-1, 3072) / 255.0, dtype=torch.float16, device=device)

POOL_SIZE = 500
EVAL_SIZE = 2000
BATCH_SIZE = 512

EPOCHS = 1000
VALIDATION_INTERVAL = 25
PRINT_INTERVAL = 50

X_pool = X_train_full[:POOL_SIZE]
X_eval = X_test_full[:EVAL_SIZE]

snn = Zor([
    Layer(3072, device=device, update_vectors_every=2),
    Layer(728, device=device, update_vectors_every=2),
    Layer(3072, device=device, update_vectors_every=1000)
], optimizer_class=optim.Adam, optimizer_kwargs={'lr': 0.001})

validation_psnr_history = []
validation_mae_history = []
validation_accuracy_history = []

def _param_count(layers):
    return sum(layers[i].input_size * layers[i+1].input_size for i in range(len(layers) - 1))

print(f"Total parameters: {_param_count(snn.layers):,}")

start_time = time.time()

for step in range(EPOCHS):
    start_idx = (step * BATCH_SIZE) % POOL_SIZE
    end_idx = start_idx + BATCH_SIZE
    
    if end_idx <= POOL_SIZE:
        batch = X_pool[start_idx:end_idx]
    else:
        # Wrap around if needed
        batch = torch.cat([X_pool[start_idx:], X_pool[:end_idx - POOL_SIZE]], dim=0)
    
    errors = snn.train_batch(batch, batch)
    snn.accuracy_history.append(100.0 * (1.0 - float(torch.mean(torch.abs(errors)))))
    
    if step % VALIDATION_INTERVAL == 0:
        val_accuracy = snn.evaluate(X_eval)
        validation_accuracy_history.append(val_accuracy)
        val_outputs = snn.forward(X_eval)
        validation_psnr_history.append(psnr(X_eval.cpu().numpy(), val_outputs.cpu().numpy()))
        validation_mae_history.append(float(torch.mean(torch.abs(X_eval - val_outputs))))
        
    if step % PRINT_INTERVAL == 0:
        val_acc = validation_accuracy_history[-1] if validation_accuracy_history else 0
        val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
        val_mae = validation_mae_history[-1] if validation_mae_history else 0
        print(f"Step {step}, Train: {snn.accuracy_history[-1]:.1f}%, Val: {val_acc:.1f}%, MAE: {val_mae:.4f}, PSNR: {val_psnr:.2f}dB LR: {snn.layers[0].learning_range:.3f}")

elapsed_time = time.time() - start_time
print(f"\nTraining completed in {elapsed_time:.1f}s!")
print(f"Final train reconstruction: {snn.accuracy_history[-1]:.1f}%")
print(f"Final validation reconstruction: {validation_accuracy_history[-1]:.1f}%")
print(f"Final validation MAE: {validation_mae_history[-1]:.4f}")
print(f"Final validation PSNR: {validation_psnr_history[-1]:.2f} dB")
for j, layer in enumerate(snn.layers):
    print(f"Layer {j} final activation %: {layer.get_activation():.3f}")

snn.plot_accuracy()
# Plot 6 validation inputs and outputs
with torch.no_grad():
    idx = torch.randperm(EVAL_SIZE)[:6]
    inputs = X_eval[idx]
    outputs = snn.forward(inputs, train=False)
    imgs = torch.cat([inputs, outputs]).cpu().numpy().reshape(12, 32, 32, 3)

fig, axes = plt.subplots(2, 6, figsize=(10, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.clip(imgs[i], 0, 1))
    ax.axis('off')
    if i == 0:
        ax.set_title("Input")
    if i == 6:
        ax.set_title("Output")
plt.tight_layout()
plt.show()
