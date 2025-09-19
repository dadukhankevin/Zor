#!/usr/bin/env python3
"""
Benchmark comparison script for Zor vs MLP autoencoder performance.
Runs both models with configurable parameters and generates comparison results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import argparse
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.datasets import cifar10

# Import Zor components
from zor import Zor, Layer
from activation_functions import *

class BenchmarkConfig:
    """Configuration for benchmark tests"""
    def __init__(self, 
                 pool_sizes=[64, 1000, 5000],
                 epochs_list=[200, 400, 450, 1000],
                 batch_size=512,
                 eval_size=2000,
                 bottleneck_size=728,
                 learning_rate=1e-3,
                 l2_reg=1e-5,
                 validation_interval=25,
                 print_interval=50,
                 seed=42):
        self.pool_sizes = pool_sizes
        self.epochs_list = epochs_list
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.bottleneck_size = bottleneck_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.validation_interval = validation_interval
        self.print_interval = print_interval
        self.seed = seed

def psnr(y_true, y_pred, eps=1e-8):
    """Calculate PSNR for image quality assessment"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse + eps)
    return float(20.0 * np.log10(1.0 / (rmse + eps)))

def recon_accuracy(y_true, y_pred):
    """Calculate reconstruction accuracy (matches Zor's metric)"""
    mae = np.mean(np.abs(y_true - y_pred))
    return 100.0 * (1.0 - float(mae))

class MLPAutoencoder(nn.Module):
    """PyTorch MLP autoencoder for benchmarking"""
    def __init__(self, input_size, bottleneck_size, lr=1e-3, weight_decay=1e-5, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.device = device
        
        # Network architecture matching TensorFlow version
        self.encoder = nn.Linear(input_size, bottleneck_size)
        self.decoder = nn.Linear(bottleneck_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Move to device
        self.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        self.accuracy_history = []
        self.validation_accuracy_history = []
    
    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.sigmoid(self.decoder(encoded))
        return decoded
    
    @torch.no_grad()
    def evaluate(self, validation_data):
        self.eval()
        if isinstance(validation_data, np.ndarray):
            validation_data = torch.tensor(validation_data, dtype=torch.float32, device=self.device)
        
        outputs = self.forward(validation_data)
        accuracy = recon_accuracy(validation_data.cpu().numpy(), outputs.cpu().numpy())
        return accuracy
    
    def train_batch(self, x_batch, y_batch):
        self.train()
        
        # Convert to tensors if needed
        if isinstance(x_batch, np.ndarray):
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        if isinstance(y_batch, np.ndarray):
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=self.device)
        
        # Forward pass
        outputs = self.forward(x_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate MAE
        with torch.no_grad():
            mae = torch.mean(torch.abs(outputs - y_batch))
        
        return float(loss.item()), float(mae.item())

def create_zor_model(input_size, bottleneck_size, device):
    """Create Zor model with specified architecture"""
    return Zor([
        Layer(input_size, device=device),
        Layer(bottleneck_size, device=device),
        Layer(input_size, device=device)
    ])

def run_zor_benchmark(X_pool, X_eval, config, epochs, device):
    """Run Zor benchmark with given parameters"""
    print(f"Running Zor with {len(X_pool)} training images, {epochs} epochs...")
    
    snn = create_zor_model(3072, config.bottleneck_size, device)
    
    validation_accuracy_history = []
    validation_psnr_history = []
    validation_mae_history = []
    
    start_time = time.time()
    
    for step in range(epochs):
        start_idx = (step * config.batch_size) % len(X_pool)
        end_idx = start_idx + config.batch_size
        
        if end_idx <= len(X_pool):
            batch = X_pool[start_idx:end_idx]
        else:
            batch = torch.cat([X_pool[start_idx:], X_pool[:end_idx - len(X_pool)]], dim=0)
        
        errors = snn.train_batch(batch, batch)
        train_accuracy = 100.0 * (1.0 - float(torch.mean(torch.abs(errors))))
        
        if step % config.validation_interval == 0:
            val_accuracy = snn.evaluate(X_eval)
            validation_accuracy_history.append(val_accuracy)
            val_outputs = snn.forward(X_eval, train=False)
            validation_psnr_history.append(psnr(X_eval.cpu().numpy(), val_outputs.cpu().numpy()))
            validation_mae_history.append(float(torch.mean(torch.abs(X_eval - val_outputs))))
        
        if step % config.print_interval == 0:
            val_acc = validation_accuracy_history[-1] if validation_accuracy_history else 0
            val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
            val_mae = validation_mae_history[-1] if validation_mae_history else 0
            print(f"  Step {step}, Train: {train_accuracy:.1f}%, Val: {val_acc:.1f}%, MAE: {val_mae:.4f}, PSNR: {val_psnr:.2f}dB")
    
    elapsed_time = time.time() - start_time
    
    # Final metrics
    final_train_acc = 100.0 * (1.0 - float(torch.mean(torch.abs(snn.train_batch(X_pool[:config.batch_size], X_pool[:config.batch_size])))))
    final_val_acc = validation_accuracy_history[-1] if validation_accuracy_history else 0
    final_val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
    final_val_mae = validation_mae_history[-1] if validation_mae_history else 0
    
    # Measure forward pass speed
    test_batch = X_eval[:config.batch_size]
    num_forward_tests = 100
    
    # Warmup
    for _ in range(10):
        _ = snn.forward(test_batch, train=False)
    
    # Time forward passes
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        
    forward_start = time.time()
    for _ in range(num_forward_tests):
        _ = snn.forward(test_batch, train=False)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    forward_time = time.time() - forward_start
    
    avg_forward_time_ms = (forward_time / num_forward_tests) * 1000  # Convert to milliseconds
    
    return {
        'model': 'Zor',
        'training_time': elapsed_time,
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'final_val_psnr': final_val_psnr,
        'final_val_mae': final_val_mae,
        'avg_forward_time_ms': avg_forward_time_ms,
        'validation_accuracy_history': validation_accuracy_history,
        'validation_psnr_history': validation_psnr_history,
        'validation_mae_history': validation_mae_history
    }

def run_mlp_benchmark(X_pool, X_eval, config, epochs, device):
    """Run MLP benchmark with given parameters"""
    print(f"Running MLP with {len(X_pool)} training images, {epochs} epochs...")
    
    mlp = MLPAutoencoder(3072, config.bottleneck_size, config.learning_rate, config.l2_reg, device)
    
    validation_accuracy_history = []
    validation_psnr_history = []
    validation_mae_history = []
    
    start_time = time.time()
    
    for step in range(epochs):
        start_idx = (step * config.batch_size) % len(X_pool)
        end_idx = start_idx + config.batch_size
        
        if end_idx <= len(X_pool):
            batch = X_pool[start_idx:end_idx]
        else:
            batch = torch.cat([X_pool[start_idx:], X_pool[:end_idx - len(X_pool)]], dim=0)
        
        loss, mae = mlp.train_batch(batch, batch)
        
        # Calculate train accuracy
        with torch.no_grad():
            mlp.eval()
            train_outputs = mlp.forward(batch)
            train_accuracy = recon_accuracy(batch.cpu().numpy(), train_outputs.cpu().numpy())
        
        if step % config.validation_interval == 0:
            val_accuracy = mlp.evaluate(X_eval)
            validation_accuracy_history.append(val_accuracy)
            
            with torch.no_grad():
                mlp.eval()
                val_outputs = mlp.forward(X_eval)
                val_outputs_np = val_outputs.cpu().numpy()
                X_eval_np = X_eval.cpu().numpy()
                validation_psnr_history.append(psnr(X_eval_np, val_outputs_np))
                validation_mae_history.append(np.mean(np.abs(X_eval_np - val_outputs_np)))
        
        if step % config.print_interval == 0:
            val_acc = validation_accuracy_history[-1] if validation_accuracy_history else 0
            val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
            val_mae = validation_mae_history[-1] if validation_mae_history else 0
            print(f"  Step {step}, Train: {train_accuracy:.1f}%, Val: {val_acc:.1f}%, MAE: {val_mae:.4f}, PSNR: {val_psnr:.2f}dB")
    
    elapsed_time = time.time() - start_time
    
    # Final metrics
    final_batch = X_pool[:config.batch_size]
    with torch.no_grad():
        mlp.eval()
        final_train_outputs = mlp.forward(final_batch)
        final_train_acc = recon_accuracy(final_batch.cpu().numpy(), final_train_outputs.cpu().numpy())
    
    final_val_acc = validation_accuracy_history[-1] if validation_accuracy_history else 0
    final_val_psnr = validation_psnr_history[-1] if validation_psnr_history else 0
    final_val_mae = validation_mae_history[-1] if validation_mae_history else 0
    
    # Measure forward pass speed
    test_batch = X_eval[:config.batch_size]
    num_forward_tests = 100
    
    # Warmup
    with torch.no_grad():
        mlp.eval()
        for _ in range(10):
            _ = mlp.forward(test_batch)
    
    # Time forward passes
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        
    forward_start = time.time()
    with torch.no_grad():
        mlp.eval()
        for _ in range(num_forward_tests):
            _ = mlp.forward(test_batch)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    forward_time = time.time() - forward_start
    
    avg_forward_time_ms = (forward_time / num_forward_tests) * 1000  # Convert to milliseconds
    
    return {
        'model': 'MLP',
        'training_time': elapsed_time,
        'final_train_accuracy': final_train_acc,
        'final_val_accuracy': final_val_acc,
        'final_val_psnr': final_val_psnr,
        'final_val_mae': final_val_mae,
        'avg_forward_time_ms': avg_forward_time_ms,
        'validation_accuracy_history': validation_accuracy_history,
        'validation_psnr_history': validation_psnr_history,
        'validation_mae_history': validation_mae_history
    }

def setup_devices():
    """Setup computation devices - both models on GPU"""
    # Use GPU if available (MPS on Mac, CUDA on others)
    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
    
    print(f"PyTorch device: {torch_device}")
    print("Note: Both models running on PyTorch with same device")
    
    return torch_device

def load_and_prepare_data(config, device):
    """Load and prepare CIFAR-10 data for PyTorch"""
    (X_train, _), (X_test, _) = cifar10.load_data()
    
    # Prepare data for PyTorch (both models use PyTorch now)
    X_train_torch = torch.tensor(X_train.reshape(-1, 3072) / 255.0, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(X_test.reshape(-1, 3072) / 255.0, dtype=torch.float32, device=device)
    
    X_eval_torch = X_test_torch[:config.eval_size]
    
    return {
        'train': X_train_torch, 
        'eval': X_eval_torch
    }

def run_comparison_suite(config, models_to_run=['zor', 'mlp']):
    """Run full comparison suite"""
    print("=" * 60)
    print("BENCHMARK COMPARISON: Zor vs MLP Autoencoder")
    print("=" * 60)
    
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = setup_devices()
    data = load_and_prepare_data(config, device)
    
    results = []
    
    for pool_size in config.pool_sizes:
        for epochs in config.epochs_list:
            # Cap pool size to what we can actually use
            max_usable_images = config.batch_size * epochs
            effective_pool_size = min(pool_size, max_usable_images)
            
            print(f"\n{'='*40}")
            print(f"Test: {effective_pool_size} images, {epochs} epochs")
            if effective_pool_size < pool_size:
                print(f"Note: Capped from {pool_size} to {effective_pool_size} images (batch_size * epochs)")
            print(f"{'='*40}")
            
            # Prepare data subsets
            X_pool = data['train'][:effective_pool_size]
            X_eval = data['eval']
            
            test_results = {
                'pool_size': pool_size,
                'effective_pool_size': effective_pool_size,
                'epochs': epochs,
                'batch_size': config.batch_size,
                'models': {}
            }
            
            # Run Zor
            if 'zor' in models_to_run:
                zor_result = run_zor_benchmark(X_pool, X_eval, config, epochs, device)
                test_results['models']['zor'] = zor_result
                print(f"Zor completed in {zor_result['training_time']:.1f}s")
                print(f"  Final Val Acc: {zor_result['final_val_accuracy']:.1f}%")
                print(f"  Final Val PSNR: {zor_result['final_val_psnr']:.2f}dB")
                print(f"  Avg Forward Pass: {zor_result['avg_forward_time_ms']:.1f}ms")
            
            # Run MLP
            if 'mlp' in models_to_run:
                mlp_result = run_mlp_benchmark(X_pool, X_eval, config, epochs, device)
                test_results['models']['mlp'] = mlp_result
                print(f"MLP completed in {mlp_result['training_time']:.1f}s")
                print(f"  Final Val Acc: {mlp_result['final_val_accuracy']:.1f}%")
                print(f"  Final Val PSNR: {mlp_result['final_val_psnr']:.2f}dB")
                print(f"  Avg Forward Pass: {mlp_result['avg_forward_time_ms']:.1f}ms")
            
            # Calculate speedup if both models ran
            if 'zor' in test_results['models'] and 'mlp' in test_results['models']:
                zor_time = test_results['models']['zor']['training_time']
                mlp_time = test_results['models']['mlp']['training_time']
                speedup = ((mlp_time - zor_time) / zor_time) * 100
                test_results['speedup_percent'] = speedup
                print(f"Speedup: {speedup:.1f}% (Zor is {mlp_time/zor_time:.1f}x faster)")
            
            results.append(test_results)
    
    return results

def print_summary_table(results):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    print(f"{'Dataset Size':<12} {'Epochs':<7} {'Zor Acc':<8} {'MLP Acc':<8} {'Zor PSNR':<10} {'MLP PSNR':<10} {'Zor Time':<9} {'MLP Time':<9} {'Speedup':<8} {'Zor FWD':<9} {'MLP FWD':<9}")
    print("-" * 100)
    
    for result in results:
        pool_size = result['pool_size']
        epochs = result['epochs']
        
        zor_data = result['models'].get('zor', {})
        mlp_data = result['models'].get('mlp', {})
        
        zor_acc = f"{zor_data.get('final_val_accuracy', 0):.1f}%" if zor_data else "N/A"
        mlp_acc = f"{mlp_data.get('final_val_accuracy', 0):.1f}%" if mlp_data else "N/A"
        zor_psnr = f"{zor_data.get('final_val_psnr', 0):.2f}dB" if zor_data else "N/A"
        mlp_psnr = f"{mlp_data.get('final_val_psnr', 0):.2f}dB" if mlp_data else "N/A"
        zor_time = f"{zor_data.get('training_time', 0):.1f}s" if zor_data else "N/A"
        mlp_time = f"{mlp_data.get('training_time', 0):.1f}s" if mlp_data else "N/A"
        speedup = f"{result.get('speedup_percent', 0):.0f}%" if 'speedup_percent' in result else "N/A"
        zor_fwd = f"{zor_data.get('avg_forward_time_ms', 0):.1f}ms" if zor_data else "N/A"
        mlp_fwd = f"{mlp_data.get('avg_forward_time_ms', 0):.1f}ms" if mlp_data else "N/A"
        
        print(f"{pool_size:<12} {epochs:<7} {zor_acc:<8} {mlp_acc:<8} {zor_psnr:<10} {mlp_psnr:<10} {zor_time:<9} {mlp_time:<9} {speedup:<8} {zor_fwd:<9} {mlp_fwd:<9}")

def run_time_to_accuracy_benchmark(X_pool, X_eval, config, target_accuracy, max_epochs, device):
    """Run benchmark to see how long it takes each model to reach target accuracy"""
    print(f"\nTime-to-accuracy benchmark: Target {target_accuracy:.1f}% accuracy")
    
    results = {}
    
    # Test Zor
    print(f"Testing Zor to reach {target_accuracy:.1f}% accuracy...")
    snn = create_zor_model(3072, config.bottleneck_size, device)
    
    zor_start_time = time.time()
    zor_reached_target = False
    zor_target_time = None
    zor_target_epoch = None
    
    for step in range(max_epochs):
        start_idx = (step * config.batch_size) % len(X_pool)
        end_idx = start_idx + config.batch_size
        
        if end_idx <= len(X_pool):
            batch = X_pool[start_idx:end_idx]
        else:
            batch = torch.cat([X_pool[start_idx:], X_pool[:end_idx - len(X_pool)]], dim=0)
        
        errors = snn.train_batch(batch, batch)
        
        if step % config.validation_interval == 0:
            val_accuracy = snn.evaluate(X_eval)
            
            if not zor_reached_target and val_accuracy >= target_accuracy:
                zor_target_time = time.time() - zor_start_time
                zor_target_epoch = step
                zor_reached_target = True
                print(f"  Zor reached {target_accuracy:.1f}% at epoch {step} in {zor_target_time:.1f}s")
                break
            
            if step % config.print_interval == 0:
                print(f"  Zor Step {step}: Val Acc {val_accuracy:.1f}%")
    
    if not zor_reached_target:
        print(f"  Zor did not reach {target_accuracy:.1f}% in {max_epochs} epochs")
    
    # Test MLP
    print(f"Testing MLP to reach {target_accuracy:.1f}% accuracy...")
    mlp = MLPAutoencoder(3072, config.bottleneck_size, config.learning_rate, config.l2_reg, device)
    
    mlp_start_time = time.time()
    mlp_reached_target = False
    mlp_target_time = None
    mlp_target_epoch = None
    
    for step in range(max_epochs):
        start_idx = (step * config.batch_size) % len(X_pool)
        end_idx = start_idx + config.batch_size
        
        if end_idx <= len(X_pool):
            batch = X_pool[start_idx:end_idx]
        else:
            batch = torch.cat([X_pool[start_idx:], X_pool[:end_idx - len(X_pool)]], dim=0)
        
        loss, mae = mlp.train_batch(batch, batch)
        
        if step % config.validation_interval == 0:
            val_accuracy = mlp.evaluate(X_eval)
            
            if not mlp_reached_target and val_accuracy >= target_accuracy:
                mlp_target_time = time.time() - mlp_start_time
                mlp_target_epoch = step
                mlp_reached_target = True
                print(f"  MLP reached {target_accuracy:.1f}% at epoch {step} in {mlp_target_time:.1f}s")
                break
            
            if step % config.print_interval == 0:
                print(f"  MLP Step {step}: Val Acc {val_accuracy:.1f}%")
    
    if not mlp_reached_target:
        print(f"  MLP did not reach {target_accuracy:.1f}% in {max_epochs} epochs")
    
    # Calculate speedup
    speedup = None
    if zor_reached_target and mlp_reached_target:
        speedup = ((mlp_target_time - zor_target_time) / zor_target_time) * 100
        print(f"\nTime-to-accuracy results:")
        print(f"  Zor: {zor_target_time:.1f}s ({zor_target_epoch} epochs)")
        print(f"  MLP: {mlp_target_time:.1f}s ({mlp_target_epoch} epochs)")
        print(f"  Speedup: {speedup:.1f}% (Zor is {mlp_target_time/zor_target_time:.1f}x faster)")
    
    return {
        'target_accuracy': target_accuracy,
        'zor_reached_target': zor_reached_target,
        'zor_target_time': zor_target_time,
        'zor_target_epoch': zor_target_epoch,
        'mlp_reached_target': mlp_reached_target,
        'mlp_target_time': mlp_target_time,
        'mlp_target_epoch': mlp_target_epoch,
        'speedup_percent': speedup
    }

def save_results(results, filename=None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
    
    def convert_to_serializable(obj):
        """Convert numpy/torch types to JSON serializable types"""
        if isinstance(obj, (np.ndarray, list)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert all results to serializable format
    serializable_results = convert_to_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Benchmark Zor vs MLP autoencoder performance")
    parser.add_argument('--pool-sizes', nargs='+', type=int, default=[64, 1000, 5000],
                       help='Training dataset sizes to test (default: 64 1000 5000)')
    parser.add_argument('--epochs', nargs='+', type=int, default=[200, 400, 450, 1000],
                       help='Number of epochs to test (default: 200 400 450 1000)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (default: 512)')
    parser.add_argument('--eval-size', type=int, default=2000,
                       help='Evaluation dataset size (default: 2000)')
    parser.add_argument('--models', nargs='+', choices=['zor', 'mlp'], default=['zor', 'mlp'],
                       help='Models to run (default: both)')
    parser.add_argument('--save', type=str, default=None,
                       help='Filename to save results (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--time-to-accuracy', action='store_true',
                       help='Run time-to-accuracy benchmark instead of fixed epochs')
    parser.add_argument('--target-accuracies', nargs='+', type=float, default=[85.0, 87.0, 89.0],
                       help='Target accuracies for time-to-accuracy benchmark (default: 85.0 87.0 89.0)')
    parser.add_argument('--max-epochs', type=int, default=1000,
                       help='Maximum epochs for time-to-accuracy benchmark (default: 1000)')
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        pool_sizes=args.pool_sizes,
        epochs_list=args.epochs,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        seed=args.seed
    )
    
    if args.time_to_accuracy:
        # Run time-to-accuracy benchmark
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        device = setup_devices()
        data = load_and_prepare_data(config, device)
        
        time_to_accuracy_results = []
        
        for pool_size in args.pool_sizes:
            # Cap pool size for time-to-accuracy (use max epochs as limit)
            max_usable_images = config.batch_size * args.max_epochs
            effective_pool_size = min(pool_size, max_usable_images)
            
            print(f"\n{'='*60}")
            print(f"TIME-TO-ACCURACY BENCHMARK: {effective_pool_size} images")
            if effective_pool_size < pool_size:
                print(f"Note: Capped from {pool_size} to {effective_pool_size} images (batch_size * max_epochs)")
            print(f"{'='*60}")
            
            X_pool = data['train'][:effective_pool_size]
            X_eval = data['eval']
            
            for target_acc in args.target_accuracies:
                result = run_time_to_accuracy_benchmark(
                    X_pool, X_eval, config, target_acc, args.max_epochs, device
                )
                result['pool_size'] = pool_size
                time_to_accuracy_results.append(result)
        
        # Print summary
        print(f"\n{'='*80}")
        print("TIME-TO-ACCURACY SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<8} {'Target':<8} {'Zor Time':<10} {'MLP Time':<10} {'Zor Epochs':<12} {'MLP Epochs':<12} {'Speedup':<8}")
        print("-" * 80)
        
        for result in time_to_accuracy_results:
            pool_size = result['pool_size']
            target = f"{result['target_accuracy']:.1f}%"
            zor_time = f"{result['zor_target_time']:.1f}s" if result['zor_reached_target'] else "N/A"
            mlp_time = f"{result['mlp_target_time']:.1f}s" if result['mlp_reached_target'] else "N/A"
            zor_epochs = str(result['zor_target_epoch']) if result['zor_reached_target'] else "N/A"
            mlp_epochs = str(result['mlp_target_epoch']) if result['mlp_reached_target'] else "N/A"
            speedup = f"{result['speedup_percent']:.0f}%" if result['speedup_percent'] else "N/A"
            
            print(f"{pool_size:<8} {target:<8} {zor_time:<10} {mlp_time:<10} {zor_epochs:<12} {mlp_epochs:<12} {speedup:<8}")
        
        if args.save:
            save_results(time_to_accuracy_results, args.save)
    else:
        # Run standard fixed-epoch benchmark
        results = run_comparison_suite(config, args.models)
        print_summary_table(results)
        save_results(results, args.save)

if __name__ == "__main__":
    main()
