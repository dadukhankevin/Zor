import torch
from typing import Optional

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Range: (0, 1)"""
    return torch.sigmoid(x)

def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation function: f(x) = max(0, x)
    Range: [0, ∞)"""
    return torch.relu(x)

def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """Leaky ReLU activation function: f(x) = max(alpha*x, x)
    Range: (-∞, ∞) with negative values scaled by alpha"""
    return torch.nn.functional.leaky_relu(x, negative_slope=alpha)

def tanh(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent activation function: f(x) = tanh(x)
    Range: (-1, 1)"""
    return torch.tanh(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax activation function: f(x) = exp(x) / sum(exp(x))
    Range: (0, 1) with sum of all values = 1"""
    return torch.softmax(x, dim=dim)

def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Swish activation function: f(x) = x * sigmoid(beta * x)
    Range: (-∞, ∞) but typically within (-0.3, ∞)"""
    return x * sigmoid(beta * x)

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit: f(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    Range: (-∞, ∞) but typically within (0, x) for positive x"""
    return torch.nn.functional.gelu(x)
