# Zor vs MLP Baseline Comparison

Comprehensive comparison of Zor neural network against traditional MLP autoencoder across different dataset sizes on CIFAR-10 reconstruction task.

## Test Configuration
- Architecture: 3072 → 512 → 3072 (autoencoder)
- Task: CIFAR-10 image reconstruction  
- Metrics: Validation accuracy, PSNR, MAE
- Hardware: CPU training

## Results Summary

| Dataset Size | Model | Val Accuracy | PSNR (dB) | MAE | Training Time | Train/Val Gap |
|--------------|-------|--------------|-----------|-----|---------------|---------------|
| 64 images | Zor | **88.1%** | **16.17** | **0.1190** | **52.4s** | **4.5%** |
| 64 images | MLP | 86.5% | 15.34 | 0.1346 | 65.0s | 11.4% |
| 1000 images | Zor | **91.8%** | **19.10** | **0.0817** | **84.3s** | **0.2%** |
| 1000 images | MLP | 90.9% | 18.66 | 0.0906 | 86.0s | 1.7% |
| 5000 images | Zor | **90.4%** | **17.84** | **0.0965** | **20.8s** | **0.0%** |
| 5000 images | MLP | 89.5% | 17.38 | 0.1049 | 27.8s | 1.0% |

**Key Findings:**
- Zor outperforms MLP across all dataset sizes and metrics
- Better generalization: smaller train/validation gaps
- Uses novel learning
## Detailed Results: 64 Images Test

Zor:
Total parameters: 4,472,832
Step 0, Train: 54.4%, Val: 52.8%, MAE: 0.4723, PSNR: 5.34dB LR: 1.000
Step 50, Train: 86.0%, Val: 84.9%, MAE: 0.1514, PSNR: 14.29dB LR: 1.000
Step 100, Train: 88.5%, Val: 86.4%, MAE: 0.1355, PSNR: 15.13dB LR: 1.000
Step 150, Train: 89.5%, Val: 87.0%, MAE: 0.1299, PSNR: 15.46dB LR: 1.000
Step 200, Train: 90.2%, Val: 87.3%, MAE: 0.1267, PSNR: 15.66dB LR: 1.000
Step 250, Train: 90.6%, Val: 87.5%, MAE: 0.1246, PSNR: 15.79dB LR: 1.000
Step 300, Train: 91.0%, Val: 87.7%, MAE: 0.1231, PSNR: 15.89dB LR: 1.000
Step 350, Train: 91.3%, Val: 87.8%, MAE: 0.1220, PSNR: 15.96dB LR: 1.000
Step 400, Train: 91.5%, Val: 87.9%, MAE: 0.1212, PSNR: 16.02dB LR: 1.000
Step 450, Train: 91.7%, Val: 87.9%, MAE: 0.1205, PSNR: 16.06dB LR: 1.000
Step 500, Train: 91.8%, Val: 88.0%, MAE: 0.1200, PSNR: 16.09dB LR: 1.000
Step 550, Train: 92.0%, Val: 88.0%, MAE: 0.1197, PSNR: 16.11dB LR: 1.000
Step 600, Train: 92.1%, Val: 88.1%, MAE: 0.1194, PSNR: 16.13dB LR: 1.000
Step 650, Train: 92.2%, Val: 88.1%, MAE: 0.1192, PSNR: 16.15dB LR: 1.000
Step 700, Train: 92.3%, Val: 88.1%, MAE: 0.1191, PSNR: 16.16dB LR: 1.000
Step 750, Train: 92.4%, Val: 88.1%, MAE: 0.1190, PSNR: 16.16dB LR: 1.000
Step 800, Train: 92.4%, Val: 88.1%, MAE: 0.1191, PSNR: 16.15dB LR: 1.000
Step 850, Train: 92.5%, Val: 88.1%, MAE: 0.1193, PSNR: 16.14dB LR: 1.000
Step 900, Train: 92.5%, Val: 88.1%, MAE: 0.1193, PSNR: 16.15dB LR: 1.000
Step 950, Train: 92.5%, Val: 88.1%, MAE: 0.1191, PSNR: 16.16dB LR: 1.000

Training completed in 52.4s!
Final train reconstruction: 92.6%
Final validation reconstruction: 88.1%
Final validation MAE: 0.1190
Final validation PSNR: 16.17 dB
Layer 0 final activation %: 0.999
Layer 1 final activation %: 0.742
Layer 2 final activation %: 1.000

MLP:
Step 0, Train: 78.8%, Val: 78.8%, MAE: 0.2119, PSNR: 11.92dB
Step 50, Train: 90.9%, Val: 84.7%, MAE: 0.1528, PSNR: 14.32dB
Step 100, Train: 94.8%, Val: 85.3%, MAE: 0.1471, PSNR: 14.55dB
Step 150, Train: 96.4%, Val: 85.7%, MAE: 0.1433, PSNR: 14.79dB
Step 200, Train: 96.9%, Val: 85.9%, MAE: 0.1406, PSNR: 14.98dB
Step 250, Train: 97.4%, Val: 86.0%, MAE: 0.1399, PSNR: 15.02dB
Step 300, Train: 97.6%, Val: 86.1%, MAE: 0.1395, PSNR: 15.04dB
Step 350, Train: 97.8%, Val: 86.1%, MAE: 0.1394, PSNR: 15.04dB
Step 400, Train: 97.9%, Val: 86.0%, MAE: 0.1399, PSNR: 15.00dB
Step 450, Train: 97.8%, Val: 85.9%, MAE: 0.1407, PSNR: 14.93dB
Step 500, Train: 97.8%, Val: 85.9%, MAE: 0.1407, PSNR: 14.94dB
Step 550, Train: 97.7%, Val: 86.2%, MAE: 0.1385, PSNR: 15.10dB
Step 600, Train: 98.0%, Val: 86.1%, MAE: 0.1392, PSNR: 15.05dB
Step 650, Train: 98.0%, Val: 86.1%, MAE: 0.1393, PSNR: 15.03dB
Step 700, Train: 97.7%, Val: 86.2%, MAE: 0.1381, PSNR: 15.12dB
Step 750, Train: 97.8%, Val: 86.0%, MAE: 0.1400, PSNR: 14.97dB
Step 800, Train: 98.0%, Val: 86.2%, MAE: 0.1381, PSNR: 15.11dB
Step 850, Train: 98.0%, Val: 86.1%, MAE: 0.1386, PSNR: 15.07dB
Step 900, Train: 97.9%, Val: 86.1%, MAE: 0.1395, PSNR: 15.01dB
Step 950, Train: 97.7%, Val: 86.2%, MAE: 0.1375, PSNR: 15.17dB
Step 1000, Train: 97.9%, Val: 86.1%, MAE: 0.1394, PSNR: 15.01dB
Step 1050, Train: 97.8%, Val: 86.3%, MAE: 0.1373, PSNR: 15.18dB
Step 1100, Train: 97.8%, Val: 86.1%, MAE: 0.1393, PSNR: 15.01dB
Step 1150, Train: 97.9%, Val: 86.3%, MAE: 0.1369, PSNR: 15.19dB
Step 1200, Train: 97.9%, Val: 86.1%, MAE: 0.1387, PSNR: 15.05dB
Step 1250, Train: 97.7%, Val: 86.3%, MAE: 0.1366, PSNR: 15.22dB
Step 1300, Train: 97.8%, Val: 86.1%, MAE: 0.1388, PSNR: 15.05dB
Step 1350, Train: 98.0%, Val: 86.2%, MAE: 0.1376, PSNR: 15.13dB
Step 1400, Train: 98.1%, Val: 86.3%, MAE: 0.1369, PSNR: 15.18dB
Step 1450, Train: 98.0%, Val: 86.4%, MAE: 0.1364, PSNR: 15.22dB
Step 1500, Train: 98.0%, Val: 86.4%, MAE: 0.1363, PSNR: 15.22dB
Step 1550, Train: 98.0%, Val: 86.2%, MAE: 0.1376, PSNR: 15.12dB
Step 1600, Train: 98.0%, Val: 86.4%, MAE: 0.1359, PSNR: 15.25dB
Step 1650, Train: 97.8%, Val: 86.2%, MAE: 0.1375, PSNR: 15.12dB
Step 1700, Train: 98.0%, Val: 86.4%, MAE: 0.1355, PSNR: 15.27dB
Step 1750, Train: 97.9%, Val: 86.5%, MAE: 0.1352, PSNR: 15.30dB
Step 1800, Train: 98.0%, Val: 86.3%, MAE: 0.1365, PSNR: 15.18dB
Step 1850, Train: 98.0%, Val: 86.4%, MAE: 0.1364, PSNR: 15.20dB
Step 1900, Train: 97.9%, Val: 86.5%, MAE: 0.1348, PSNR: 15.32dB
Step 1950, Train: 97.9%, Val: 86.3%, MAE: 0.1366, PSNR: 15.18dB

Training completed in 65.0s!
Final train reconstruction: 97.9%
Final validation reconstruction: 86.5%
Final validation MAE: 0.1346
Final validation PSNR: 15.34 dB
