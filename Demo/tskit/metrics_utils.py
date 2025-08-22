import torch
import numpy as np

def _get_backend(backend):
    """Helper function to determine the backend"""
    if backend not in ['torch', 'numpy']:
        raise ValueError("Backend must be either 'torch' or 'numpy'")
    return backend

def mse_loss(pred, label, backend='torch'):
    """
    Mean Squared Error (MSE)
    
    Args:
        pred: predicted values
        label: ground truth values
        backend: 'torch' or 'numpy'
    """
    backend = _get_backend(backend)
    if backend == 'torch':
        return torch.mean((label - pred) ** 2)
    else:
        return np.mean((label - pred) ** 2)

def r2_loss(pred, label, backend='torch'):
    """
    R-squared (Coefficient of Determination)
    """
    backend = _get_backend(backend)
    if backend == 'torch':
        target_mean = torch.mean(label)
        ss_tot = torch.sum((label - target_mean) ** 2)
        ss_res = torch.sum((label - pred) ** 2)
    else:
        target_mean = np.mean(label)
        ss_tot = np.sum((label - target_mean) ** 2)
        ss_res = np.sum((label - pred) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)  # small epsilon to avoid division by zero

def mape_loss(y_pred, y_true, backend='torch'):
    """
    Mean Absolute Percentage Error (MAPE)
    """
    backend = _get_backend(backend)
    epsilon = 1e-8
    if backend == 'torch':
        return torch.mean(torch.abs((y_pred - y_true) / (y_true + epsilon)))
    else:
        return np.mean(np.abs((y_pred - y_true) / (y_true + epsilon)))

def smape_loss(y_pred, y_true, backend='torch'):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    backend = _get_backend(backend)
    epsilon = 1e-8
    if backend == 'torch':
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_pred) + torch.abs(y_true) + epsilon
    else:
        numerator = np.abs(y_pred - y_true)
        denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return 2.0 * (numerator / denominator).mean()

def mae_loss(pred, label, backend='torch'):
    """
    Mean Absolute Error (MAE)
    """
    backend = _get_backend(backend)
    if backend == 'torch':
        return torch.mean(torch.abs(label - pred))
    else:
        return np.mean(np.abs(label - pred))

def rmse_loss(pred, label, backend='torch'):
    """
    Root Mean Squared Error (RMSE)
    """
    backend = _get_backend(backend)
    if backend == 'torch':
        mse = torch.mean((label - pred) ** 2)
        return torch.sqrt(mse)
    else:
        mse = np.mean((label - pred) ** 2)
        return np.sqrt(mse)

def msle_loss(pred, label, backend='torch'):
    """
    Mean Squared Logarithmic Error (MSLE)
    """
    backend = _get_backend(backend)
    if backend == 'torch':
        log_pred = torch.log(pred + 1)
        log_label = torch.log(label + 1)
        return torch.mean((log_label - log_pred) ** 2)
    else:
        log_pred = np.log(pred + 1)
        log_label = np.log(label + 1)
        return np.mean((log_label - log_pred) ** 2)