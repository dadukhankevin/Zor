"""
Zor CIFAR-10 autoencoder demo (tutorial-style)

Sections
- Imports & helpers
- Data loading
- Model definition
- Training loop with curriculum
- Metrics & plotting (incl. forgotten-image tracking)
- Final visuals
"""

from zor import Zor, Layer
from activation_functions import sigmoid
import numpy as np
import torch
from keras.datasets import cifar10  # type: ignore
import time
import matplotlib.pyplot as plt

# Setup device for MPS (Metal Performance Shaders) on Mac
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def psnr(y_true, y_pred, eps=1e-8):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse + eps)
    return float(20.0 * np.log10(1.0 / (rmse + eps)))

def recon_accuracy(y_true, y_pred):
    return 100.0 * (1.0 - float(np.mean(np.abs(y_true - y_pred))))

def simple_plot(snn, X_val_samples=None):
    if not hasattr(simple_plot, 'fig'):
        simple_plot.fig = plt.figure(figsize=(15, 15))
        plt.ion()
    
    plt.figure(simple_plot.fig.number)
    plt.clf()
    
    # Input/Output images
    plt.subplot(3, 3, 1)
    if hasattr(snn, 'last_batch'):
        img = snn.last_batch[0].cpu().numpy().reshape(32, 32, 3)
        plt.imshow(img)
        plt.title('Train Input')
        plt.axis('off')
    
    plt.subplot(3, 3, 2)
    if hasattr(snn, 'last_outputs'):
        img = np.clip(snn.last_outputs[0].cpu().numpy().reshape(32, 32, 3), 0, 1)
        plt.imshow(img)
        plt.title('Train Output')
        plt.axis('off')
    
    # Validation samples
    plt.subplot(3, 3, 3)
    if X_val_samples is not None:
        val_outputs = snn.forward(X_val_samples)
        img = np.clip(val_outputs[0].cpu().numpy().reshape(32, 32, 3), 0, 1)
        plt.imshow(img)
        plt.title('Val Output')
        plt.axis('off')
    
    # Reconstruction score
    plt.subplot(3, 3, 4)
    if snn.accuracy_history:
        plt.plot([acc.item() for acc in snn.accuracy_history], 'g-', label='Train')
    if snn.validation_accuracy_history:
        val_x = [i * validation_interval for i in range(len(snn.validation_accuracy_history))]
        plt.plot(val_x, snn.validation_accuracy_history, 'r-', label='Val')
    if forgotten_accuracy_history:
        forgot_x = [i * validation_interval for i in range(len(forgotten_accuracy_history))]
        plt.plot(forgot_x, forgotten_accuracy_history, 'orange', label='Forgotten')
    # Per-forgotten-image curves
    if 'forgotten_per_image_history' in globals():
        for i, hist in enumerate(forgotten_per_image_history):
            if hist:
                plt.plot(forgotten_steps_history[i], hist, '--', label=f'F{i+1}')
    plt.title('Reconstruction (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PSNR
    plt.subplot(3, 3, 5)
    if validation_psnr_history:
        val_x = [i * validation_interval for i in range(len(validation_psnr_history))]
        plt.plot(val_x, validation_psnr_history, 'b-', label='Val PSNR')
        plt.title('PSNR (dB)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Validation MAE
    plt.subplot(3, 3, 6)
    if validation_mae_history:
        val_x = [i * validation_interval for i in range(len(validation_mae_history))]
        plt.plot(val_x, validation_mae_history, 'm-', label='Val MAE')
        plt.title('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Validation samples grid (2x10 - ground truth and reconstructions)
    if X_val_samples is not None:
        # Reuse the same forward results if sizes match; otherwise compute for first 10
        if 'val_outputs' in locals() and len(val_outputs) >= 10:
            val_outputs_10 = val_outputs[:10]
        else:
            val_outputs_10 = snn.forward(X_val_samples[:10])
        # Ground truth row
        for i in range(10):
            plt.subplot(4, 10, 21 + i)
            img = X_val_samples[i].cpu().numpy().reshape(32, 32, 3)
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title('Val Ground Truth', fontsize=8)
        
        # Reconstruction row
        for i in range(10):
            plt.subplot(4, 10, 31 + i)
            img = np.clip(val_outputs_10[i].cpu().numpy().reshape(32, 32, 3), 0, 1)
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title('Val Reconstructions', fontsize=8)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.draw()
    plt.pause(0.01)
# Load and prepare data
(X_train, _), (X_test, _) = cifar10.load_data()
pool_size = 300
eval_size = 500
active_size = 50

# Use train set for training pool, test set for evaluation
X_train_full = torch.tensor(X_train.reshape(-1, 3072) / 255.0, dtype=torch.float32, device=device)
X_test_full = torch.tensor(X_test.reshape(-1, 3072) / 255.0, dtype=torch.float32, device=device)

X_pool = X_train_full[:pool_size]
X_eval = X_test_full[:eval_size]  # Use test set for evaluation
X_val_samples = X_test_full[:10]  # Fixed 10 validation samples for plotting
X_candidates = X_pool[active_size:]  # Rest of pool are candidates
if len(X_candidates) > 0:
    perm = torch.randperm(len(X_candidates), device=device)
    X_candidates = X_candidates[perm]

# Initialize network

snn = Zor([
    Layer(
        3072,
        target_activation=.9,
        activation_function=None,
        learning_range=.05,
        activation_rate=0.2,
        novelty_factor=-.2,
        universal_rolling_factor=.9,
        device=device
    ),
    Layer(
        512,
        target_activation=.5,
        activation_function=None,
        learning_range=.05,  
        activation_rate=0.2,
        novelty_factor=-.2,
        universal_rolling_factor=.9,
        device=device
    ),
    Layer(
        3072,
        target_activation=.5,
        activation_function=sigmoid,
        learning_range=.05,
        activation_rate=0.2,
        novelty_factor=-.2,
        universal_rolling_factor=.9,
        device=device
    )
])

# Initialize histories (metrics and forgetting)
validation_psnr_history = []
validation_mae_history = []
forgotten_images = []  # Store up to 3 removed images (copies)
forgotten_accuracy_history = []  # Mean accuracy of all forgotten images
forgotten_per_image_history = [[], [], []]  # Per-image accuracy curves
forgotten_steps_history = [[], [], []]  # Steps corresponding to per-image points

# Parameter count (weighted connections only)
def _param_count(layers):
    return sum(layers[i].input_size * layers[i+1].input_size for i in range(len(layers) - 1))

print(f"Total parameters: {_param_count(snn.layers):,}")


# Training setup
batch_size = 10
total_steps = 10000
validation_interval = 25
plot_interval = 10
print_interval = 50

X_shuffled = X_pool[:active_size][torch.randperm(active_size, device=device)]
cand_pos = 0
current_pos = 0
start_time = time.time()

for step in range(total_steps):
    # Get next batch using circular buffer
    end_pos = current_pos + batch_size
    if end_pos > len(X_shuffled):
        tail_idx = torch.arange(current_pos, len(X_shuffled), device=device)
        head_idx = torch.arange(0, end_pos - len(X_shuffled), device=device)
        batch_indices = torch.cat([tail_idx, head_idx])
        batch = torch.cat([X_shuffled[current_pos:], X_shuffled[:end_pos - len(X_shuffled)]])
        current_pos = end_pos - len(X_shuffled)
    else:
        batch_indices = torch.arange(current_pos, end_pos, device=device)
        batch = X_shuffled[current_pos:end_pos]
        current_pos = end_pos
        
    # Forward pass and compute errors
    outputs = snn.forward(batch)
    errors = batch - outputs
    train_accuracy = 100.0 * (1.0 - torch.mean(torch.abs(errors)))
    
    # Backward pass
    snn.reinforce(errors)
    
    # Track metrics
    snn.accuracy_history.append(train_accuracy)
    for j, layer in enumerate(snn.layers):
        snn.activation_history[j].append(float(torch.mean(layer.spikes)))
    snn.last_batch = batch
    snn.last_outputs = outputs
    
    # Curriculum: replace only the most accurate image
    if len(X_candidates) > 0:
        per_sample_acc = torch.clamp(1.0 - torch.mean(torch.abs(errors), dim=1), 0, 1)
        best_idx = torch.argmax(per_sample_acc)
        if per_sample_acc[best_idx] > 0.91:  # Only replace if accuracy is high enough
            # Track first 3 forgotten images
            if len(forgotten_images) < 3:
                forgotten_images.append(X_shuffled[batch_indices[best_idx]].clone().cpu())
                print(f"Tracking forgotten image {len(forgotten_images)} at step {step}")
            # print(f"Replacing sample {batch_indices[best_idx]} with candidate {cand_pos}")
            X_shuffled[batch_indices[best_idx]] = X_candidates[cand_pos]
            cand_pos = (cand_pos + 1) % len(X_candidates)
    
    # Validation evaluation
    if step % validation_interval == 0:
        val_accuracy = snn.evaluate(X_eval)
        snn.validation_accuracy_history.append(val_accuracy)
        val_outputs = snn.forward(X_eval)
        val_psnr = psnr(X_eval.cpu().numpy(), val_outputs.cpu().numpy())
        validation_psnr_history.append(val_psnr)
        val_mae = float(torch.mean(torch.abs(X_eval - val_outputs)))
        validation_mae_history.append(val_mae)
        
        # Track forgotten images accuracy
        if forgotten_images:
            forgotten_batch = torch.stack(forgotten_images).to(device)
            forgotten_outputs = snn.forward(forgotten_batch)
            # Mean across all forgotten images
            forgotten_acc = recon_accuracy(forgotten_batch.cpu().numpy(), forgotten_outputs.cpu().numpy())
            forgotten_accuracy_history.append(forgotten_acc)
            # Per-image curves
            per_img_acc = 100.0 * (1.0 - torch.mean(torch.abs(forgotten_batch - forgotten_outputs), dim=1))
            for i in range(len(forgotten_images)):
                forgotten_per_image_history[i].append(float(per_img_acc[i]))
                forgotten_steps_history[i].append(step)
    
    # Plotting and logging
    if step % plot_interval == 0:
        simple_plot(snn, X_val_samples)
        
    if step % print_interval == 0:
        val_acc = snn.validation_accuracy_history[-1] if snn.validation_accuracy_history else 0
        val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
        val_mae = validation_mae_history[-1] if validation_mae_history else 0
        forgot_acc = forgotten_accuracy_history[-1] if forgotten_accuracy_history else 0
        epoch_progress = (current_pos / len(X_shuffled)) * 100
        per_img_str = ""
        if forgotten_accuracy_history and any(len(h) > 0 for h in forgotten_per_image_history):
            parts = []
            for i in range(len(forgotten_images)):
                if forgotten_per_image_history[i]:
                    parts.append(f"F{i+1}:{forgotten_per_image_history[i][-1]:.1f}%")
            if parts:
                per_img_str = " (" + ", ".join(parts) + ")"
        forgot_str = f", Forgot: {forgot_acc:.1f}%{per_img_str}" if forgotten_accuracy_history else ""
        print(f"Step {step}, Epoch: {epoch_progress:.1f}%, Recon: {train_accuracy.item():.1f}%, Val Recon: {val_acc:.1f}%, Val MAE: {val_mae:.4f}, PSNR: {val_psnr:.2f} dB{forgot_str}")

# Final results
elapsed_time = time.time() - start_time
print(f"\nTraining completed in {elapsed_time:.1f}s!")
print(f"Final train reconstruction: {snn.accuracy_history[-1].item():.1f}%")
print(f"Final validation reconstruction: {snn.validation_accuracy_history[-1]:.1f}%")
print(f"Final validation MAE: {validation_mae_history[-1]:.4f}")
print(f"Final validation PSNR: {validation_psnr_history[-1]:.2f} dB")
if forgotten_accuracy_history:
    print(f"Final forgotten images reconstruction: {forgotten_accuracy_history[-1]:.1f}%")
    print(f"Number of forgotten images tracked: {len(forgotten_images)}")

for j, layer in enumerate(snn.layers):
    final_activation = snn.activation_history[j][-1]
    print(f"Layer {j} activation: {final_activation:.3f} (target: {layer.target_activation})")

simple_plot(snn, X_val_samples)
plt.ioff()  # Turn off interactive mode
plt.show(block=False)  # Keep the plot window open

# Display input/output examples
plt.figure(figsize=(20, 4))
for i in range(5):
    # Input image
    plt.subplot(2, 10, i + 1)
    plt.imshow(snn.last_batch[i].cpu().numpy().reshape(32, 32, 3))
    plt.title(f'Input {i+1}')
    plt.axis('off')
    
    # Reconstructed output
    plt.subplot(2, 10, i + 11)
    output_img = np.clip(snn.last_outputs[i].cpu().numpy().reshape(32, 32, 3), 0, 1)
    plt.imshow(output_img)
    plt.title(f'Output {i+1}')
    plt.axis('off')

plt.tight_layout()

# Final small grid: forgotten originals vs current reconstructions
if forgotten_images:
    plt.figure(figsize=(8, 6))
    for i in range(len(forgotten_images)):
        plt.subplot(3, 2, 2*i + 1)
        plt.imshow(forgotten_images[i].numpy().reshape(32, 32, 3))
        plt.title(f'Forgotten {i+1} (orig)')
        plt.axis('off')
        
        forgotten_tensor = forgotten_images[i:i+1][0].unsqueeze(0).to(device)
        recon = np.clip(snn.forward(forgotten_tensor)[0].cpu().numpy().reshape(32, 32, 3), 0, 1)
        plt.subplot(3, 2, 2*i + 2)
        plt.imshow(recon)
        plt.title(f'Forgotten {i+1} (recon)')
        plt.axis('off')
    plt.tight_layout()

plt.show(block=True)  # Keep plots open and block until closed