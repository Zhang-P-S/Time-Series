import torch

def get_device():
    ''' Get device (if GPU is available, use GPU, specifically cuda2) '''
    if torch.cuda.is_available():
        # Check if cuda2 is available
        if torch.cuda.device_count() > 2:
            return 'cuda:2'  # Use cuda2
        else:
            return 'cuda'  # Use default CUDA device if cuda2 is not available
    return 'cpu'  # Fallback to CPU if no CUDA devices are available