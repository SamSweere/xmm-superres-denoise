import torch

class CustomSigmoid:
    """
    A class to apply a custom sigmoid normalization to a given tensor. The sigmoid
    function is defined as:
    
    y = 1 / (1 + exp(-k * (x - x0)))
    
    The function behaves linearly around `x0` with a slope controlled by `k`.
    
    Args:
    - k (float): Controls the steepness of the sigmoid function. Higher values make the 
                 transition sharper.
    - x0 (float): The x-value around which the sigmoid behaves linearly.
    """
    def __init__(self, k: float = 2.0, x0: float = 2.0):
        """
        Initialize the CustomSigmoidNormalizer with specific parameters.
        
        Args:
        - k (float): Steepness parameter of the sigmoid function.
        - x0 (float): Center of the linear region of the sigmoid function.
        """
        self.k = k
        self.x0 = x0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the custom sigmoid function to the input tensor.
        
        Args:
        - x (torch.Tensor): The input tensor to be normalized.
        
        Returns:
        - torch.Tensor: The normalized tensor.
        """
        return 1 / (1 + torch.exp(-self.k * (x - self.x0)))
