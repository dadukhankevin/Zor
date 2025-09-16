import numpy as np

def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    Range: (0, 1)"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function: f(x) = max(0, x)
    Range: [0, ∞)"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function: f(x) = max(alpha*x, x)
    Range: (-∞, ∞) with negative values scaled by alpha"""
    return np.maximum(alpha * x, x)

def tanh(x):
    """Hyperbolic tangent activation function: f(x) = tanh(x)
    Range: (-1, 1)"""
    return np.tanh(x)

def softmax(x, axis=-1):
    """Softmax activation function: f(x) = exp(x) / sum(exp(x))
    Range: (0, 1) with sum of all values = 1"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def swish(x, beta=1.0):
    """Swish activation function: f(x) = x * sigmoid(beta * x)
    Range: (-∞, ∞) but typically within (-0.3, ∞)"""
    return x * sigmoid(beta * x)

def gelu(x):
    """Gaussian Error Linear Unit: f(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    Range: (-∞, ∞) but typically within (0, x) for positive x"""
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
