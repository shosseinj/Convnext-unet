import torch
import torch.nn as nn
import copy
import numpy as np


import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn


def set_up_logging(logging_dir, model_name):
    """
    Set up logging for the simulation with real-time flushing.
    """
    os.makedirs(logging_dir, exist_ok=True)
    
    # Get root logger and clear any existing handlers
    logger = logging.getLogger()
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with immediate flushing
    log_file = os.path.join(logging_dir, f'{model_name}_log.txt')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    
    # Add handlers to root logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    
    # Silence matplotlib
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    
    # Test log
    logging.info(f"Logging initialized to {log_file}")


def get_optimizer(model, lr):
    """
    Get optimizer for the training on MNIST/Fashion-MNIST dataset.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/5000))
    return optimizer, scheduler

class Conv2DWithBias(nn.Conv2d):
    """Convolutional layer with location-dependent bias."""
    
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)
        self.register_buffer('BN', torch.tensor([0]))
        self.register_buffer('BN_before_ReLU', torch.tensor([0]))
        self.bias_param = nn.Parameter(torch.zeros(9, self.out_channels))
        self.use_custom_bias = False
    
    def set_bias(self, bias, W=None, b_term=None):
        """
        Set location-dependent bias.
        
        Args:
            bias: The bias vector (output_channels,)
            W: Original kernel weights [H, W, in_channels, out_channels]
            b_term: Scaling term for bias adjustment (in_channels,) or scalar
        """
        if b_term is None:
            b_term = [0.0]
        
        self.use_custom_bias = True
        device = self.bias_param.device
        dtype = self.bias_param.dtype
        
        # Ensure bias is on correct device and dtype
        bias = bias.to(device=device, dtype=dtype)
        
        if W is not None:
            W = W.to(device=device, dtype=dtype)
            # W shape: [H, W, in_channels, out_channels]
            W_sum_2D = torch.sum(W, dim=(0, 1))  # Shape: [in_channels, out_channels]
            
            # Convert b_term to tensor if needed
            if isinstance(b_term, torch.Tensor):
                b_term = b_term.to(device=device, dtype=dtype)
            else:
                b_term = torch.tensor(b_term, dtype=dtype, device=device)
            
            # b_term should match input channels of W
            # W shape: [H, W, in_channels, out_channels]
            in_channels = W.shape[2]
            
            if b_term.numel() == 1:
                # Single value - repeat for all input channels
                b_term = b_term.repeat(in_channels)
            elif len(b_term) != in_channels:
                # Adjust b_term to match input channels
                if len(b_term) < in_channels:
                    # Repeat to fill
                    repeat_factor = (in_channels + len(b_term) - 1) // len(b_term)
                    b_term = b_term.repeat(repeat_factor)[:in_channels]
                else:
                    # Truncate
                    b_term = b_term[:in_channels]
        
        for i in range(9):
            if i == 0:
                # Inner part of image - no padding effect
                if W is not None:
                    b_term_tensor = b_term.to(dtype=dtype)
                    delta_sum_W = torch.zeros((b_term_tensor.shape[0], W_sum_2D.shape[1]), dtype=dtype, device=device)
                    delta_bias = torch.sum(torch.matmul(torch.diag(b_term_tensor), delta_sum_W), dim=0)
                    self.bias_param.data[i] = bias - delta_bias
                else:
                    self.bias_param.data[i] = bias
                    
            elif i == 1:
                # Top-left corner
                delta_sum_W = (W_sum_2D - torch.sum(W[1:, 1:, :, :], dim=[0, 1]))
            elif i == 2:
                # Top edge
                delta_sum_W = torch.sum(W[:1, :, :, :], dim=[0, 1])
            elif i == 3:
                # Top-right corner
                delta_sum_W = (W_sum_2D - torch.sum(W[1:, :-1, :, :], dim=[0, 1]))
            elif i == 4:
                # Right edge
                delta_sum_W = torch.sum(W[:, -1:, :, :], dim=[0, 1])
            elif i == 5:
                # Bottom-right corner
                delta_sum_W = (W_sum_2D - torch.sum(W[:-1, :-1, :, :], dim=[0, 1]))
            elif i == 6:
                # Bottom edge
                delta_sum_W = torch.sum(W[-1:, :, :, :], dim=[0, 1])
            elif i == 7:
                # Bottom-left corner
                delta_sum_W = (W_sum_2D - torch.sum(W[:-1, 1:, :, :], dim=[0, 1]))
            elif i == 8:
                # Left edge
                delta_sum_W = torch.sum(W[:, :1, :, :], dim=[0, 1])
            
            if i > 0 and W is not None:
                b_term_tensor = b_term.to(dtype=dtype)
                delta_bias = torch.sum(torch.matmul(torch.diag(b_term_tensor), delta_sum_W), dim=0)
                self.bias_param.data[i] = bias - delta_bias
            
            # Break conditions (same as original TensorFlow code)
            if self.padding == 0 or self.BN != 1 or self.BN_before_ReLU == 1:
                # Copy first bias to all positions
                for j in range(1, 9):
                    self.bias_param.data[j] = self.bias_param.data[0]
                break
    
    def forward(self, inputs):
        result = super().forward(inputs)
        
        if self.use_custom_bias:
            # Move bias to same device/dtype as input
            bias_data = self.bias_param.to(device=inputs.device, dtype=inputs.dtype)
            
            if self.padding == 0 or self.BN != 1 or self.BN_before_ReLU == 1:
                # Same bias for all locations
                result = result + bias_data[0].view(1, -1, 1, 1)
            else:
                # Different bias for 9 different locations (handles 'same' padding)
                result_0 = result[:, :, 1:-1, 1:-1] + bias_data[0].view(1, -1, 1, 1)
                result_1 = result[:, :, :1, :1] + bias_data[1].view(1, -1, 1, 1)
                result_2 = result[:, :, :1, 1:-1] + bias_data[2].view(1, -1, 1, 1)
                result_3 = result[:, :, :1, -1:] + bias_data[3].view(1, -1, 1, 1)
                result_4 = result[:, :, 1:-1, -1:] + bias_data[4].view(1, -1, 1, 1)
                result_5 = result[:, :, -1:, -1:] + bias_data[5].view(1, -1, 1, 1)
                result_6 = result[:, :, -1:, 1:-1] + bias_data[6].view(1, -1, 1, 1)
                result_7 = result[:, :, -1:, :1] + bias_data[7].view(1, -1, 1, 1)
                result_8 = result[:, :, 1:-1, :1] + bias_data[8].view(1, -1, 1, 1)
                
                top_row = torch.cat([result_1, result_2, result_3], dim=3)
                middle = torch.cat([result_8, result_0, result_4], dim=3)
                bottom_row = torch.cat([result_7, result_6, result_5], dim=3)
                result = torch.cat([top_row, middle, bottom_row], dim=2)
        
        return result

def fuse_bn_torch(model, p, q, BN=True, BN_before_ReLU=False):
    """
    Fuse batch normalization layers for faster inference.
    """
    # Get device and dtype from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Deep copy to CPU first, then move back if needed
    model_copy = copy.deepcopy(model).cpu()
    
    fused_model = nn.Sequential()
    layers = list(model_copy.children())
    
    i = 0
    
    # Handle imaginary batch normalization from input scaling
    if not (p == 0 and q == 1):
        i = _fuse_imaginary_bn(fused_model, layers, p, q, i, dtype)
    
    if BN:
        if BN_before_ReLU:
            while i < len(layers):
                if isinstance(layers[i], (nn.BatchNorm2d, nn.BatchNorm1d)):
                    i = _fuse_bn_before_activation(fused_model, layers, i, dtype)
                elif i == len(layers) - 1:
                    fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
                    i += 1
                else:
                    fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
                    i += 1
        else:
            while i < len(layers):
                if isinstance(layers[i], (nn.BatchNorm2d, nn.BatchNorm1d)):
                    i = _fuse_bn_after_activation(fused_model, layers, i, dtype)
                else:
                    fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
                    i += 1
    else:
        _copy_model_from_layers(fused_model, layers, i)
    
    # Move fused model back to original device and dtype
    fused_model = fused_model.to(device=device, dtype=dtype)
    
    return fused_model


def _fuse_imaginary_bn(fused_model, layers, p, q, start_idx=0, dtype=torch.float32):
    """Fuse imaginary BN from input scaling."""
    i = start_idx
    
    while i < len(layers) and not isinstance(layers[i], (nn.Conv2d, nn.Linear)):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
        i += 1
    
    if i >= len(layers):
        return i
    
    first_layer = layers[i]
    
    if isinstance(first_layer, nn.Conv2d):
        input_image_shape = first_layer.kernel_size[0]
        input_channels = first_layer.in_channels
        
        # Create tensors with matching dtype
        device = first_layer.weight.device
        kappa = torch.full((input_channels,), q - p, dtype=dtype, device=device)
        b_term = torch.full((input_channels,), p, dtype=dtype, device=device)
        kappa = kappa.repeat(input_image_shape**2)
        b_term = b_term.repeat(input_image_shape**2)
        
        W = first_layer.weight.data.clone().to(dtype)
        bias = first_layer.bias.data.clone().to(dtype) if first_layer.bias is not None else torch.zeros(first_layer.out_channels, dtype=dtype, device=device)
        
        W_reshaped = W.reshape(-1, first_layer.out_channels)
        kappa_mat = torch.diag(kappa)
        W_fused = torch.matmul(kappa_mat, W_reshaped)
        W_fused = W_fused.reshape(W.shape)
        
        b_fused = bias + torch.sum(torch.matmul(torch.diag(b_term), W_reshaped), dim=0)
        
        layer = _copy_layer(first_layer)
        layer.weight.data = W_fused.to(first_layer.weight.dtype)
        layer.BN.data = torch.tensor([1], device=device)
        layer.BN_before_ReLU.data = torch.tensor([0], device=device)
        layer.set_bias(bias=b_fused, W=W, b_term=b_term[:input_channels])
        
    elif isinstance(first_layer, nn.Linear):
        input_features = first_layer.in_features
        
        device = first_layer.weight.device
        kappa = torch.full((input_features,), q - p, dtype=dtype, device=device)
        b_term = torch.full((input_features,), p, dtype=dtype, device=device)
        
        W = first_layer.weight.data.clone().to(dtype)
        bias = first_layer.bias.data.clone().to(dtype) if first_layer.bias is not None else torch.zeros(first_layer.out_features, dtype=dtype, device=device)
        
        kappa_mat = torch.diag(kappa)
        W_fused = torch.matmul(kappa_mat, W.t()).t()
        b_fused = bias + torch.matmul(torch.diag(b_term), W.t()).sum(dim=1)
        
        layer = _copy_layer(first_layer)
        layer.weight.data = W_fused.to(first_layer.weight.dtype)
        layer.bias.data = b_fused.to(first_layer.weight.dtype)
    
    fused_model.add_module(str(len(fused_model)), layer)
    
    if i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i + 1]))
        return i + 2
    
    return i + 1


def _fuse_bn_before_activation(fused_model, layers, i, dtype=torch.float32):
    """Fuse BN with previous layer (BN before ReLU)."""
    bn = layers[i]
    previous_layer = layers[i - 1]
    
    device = bn.weight.device
    
    gamma = bn.weight.data.clone().to(dtype)
    beta = bn.bias.data.clone().to(dtype)
    mean = bn.running_mean.clone().to(dtype)
    var = bn.running_var.clone().to(dtype)
    eps = bn.eps
    
    kappa = gamma / torch.sqrt(var + eps)
    
    if isinstance(previous_layer, nn.Conv2d):
        W = previous_layer.weight.data.clone().to(dtype)
        bias = previous_layer.bias.data.clone().to(dtype) if previous_layer.bias is not None else torch.zeros(previous_layer.out_channels, dtype=dtype, device=device)
        
        W_reshaped = W.reshape(-1, previous_layer.out_channels)
        kappa_mat = torch.diag(kappa)
        W_fused = torch.matmul(W_reshaped, kappa_mat)
        W_fused = W_fused.reshape(W.shape)
        
        b_fused = beta - mean * kappa + torch.matmul(kappa_mat, bias.unsqueeze(1)).squeeze()
        
        layer = _copy_layer(previous_layer)
        layer.weight.data = W_fused.to(previous_layer.weight.dtype)
        layer.BN.data = torch.tensor([1], device=device)
        layer.BN_before_ReLU.data = torch.tensor([1], device=device)
        layer.set_bias(b_fused)
        
    elif isinstance(previous_layer, nn.Linear):
        W = previous_layer.weight.data.clone().to(dtype)
        bias = previous_layer.bias.data.clone().to(dtype) if previous_layer.bias is not None else torch.zeros(previous_layer.out_features, dtype=dtype, device=device)
        
        kappa_mat = torch.diag(kappa)
        W_fused = torch.matmul(kappa_mat, W.t()).t()
        b_fused = beta - mean * kappa + torch.matmul(kappa_mat, bias)
        
        layer = _copy_layer(previous_layer)
        layer.weight.data = W_fused.to(previous_layer.weight.dtype)
        layer.bias.data = b_fused.to(previous_layer.weight.dtype)
    
    fused_model.add_module(str(len(fused_model)), layer)
    
    if i + 1 < len(layers):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i + 1]))
    
    i += 2
    while i < len(layers) and 'dropout' in str(type(layers[i])).lower():
        i += 1
    
    if i < len(layers) and (isinstance(layers[i], nn.Flatten) or isinstance(layers[i], nn.MaxPool2d)):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
        i += 1
    
    return i



def _fuse_bn_after_activation(fused_model, layers, i, dtype=torch.float32):
    """Fuse BN with following layer (BN after ReLU)."""
    bn = layers[i]
    
    device = bn.weight.device
    
    gamma = bn.weight.data.clone().to(dtype)
    beta = bn.bias.data.clone().to(dtype)
    mean = bn.running_mean.clone().to(dtype)
    var = bn.running_var.clone().to(dtype)
    eps = bn.eps
    
    # kappa and b_term are size [output_channels_of_bn]
    kappa = gamma / torch.sqrt(var + eps)
    b_term = beta - mean * kappa
    
    # Store original before any modifications
    kappa_original = kappa.clone()
    b_term_original = b_term.clone()
    
    i += 1
    while i < len(layers) and 'dropout' in str(type(layers[i])).lower():
        i += 1
    
    if i < len(layers) and isinstance(layers[i], nn.MaxPool2d):
        mp = layers[i]
        mmp = MaxMinPool2D(kernel_size=mp.kernel_size, stride=mp.stride, padding=mp.padding)
        mmp.sign = torch.sign(kappa).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        fused_model.add_module(str(len(fused_model)), mmp)
        i += 1
        
        while i < len(layers) and 'dropout' in str(type(layers[i])).lower():
            i += 1
    
    if i < len(layers) and isinstance(layers[i], nn.Flatten):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
        i += 1
    
    while i < len(layers) and not isinstance(layers[i], (nn.Conv2d, nn.Linear)):
        if not isinstance(layers[i], nn.ReLU):
            fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i]))
        i += 1
    
    if i >= len(layers):
        return i
    
    next_layer = layers[i]
    
    if isinstance(next_layer, nn.Conv2d):
        W = next_layer.weight.data.clone().to(dtype)
        bias = next_layer.bias.data.clone().to(dtype) if next_layer.bias is not None else torch.zeros(next_layer.out_channels, dtype=dtype, device=device)
        
        # W shape: [out_channels, in_channels, H, W] in PyTorch
        # We need to reshape to [H*W*in_channels, out_channels] for matrix multiplication
        out_channels = next_layer.out_channels
        in_channels = next_layer.in_channels
        kernel_h, kernel_w = next_layer.kernel_size
        
        # The kappa from BN has size = BN output channels = previous layer output channels
        # This should equal next_layer.in_channels
        bn_out_channels = len(kappa_original)
        
        # For Conv2d: each input channel gets the same kappa/b_term
        # So we need to repeat kappa/b_term for spatial dimensions
        # kappa_original size: [bn_out_channels] = [in_channels]
        # After repeat: [in_channels * kernel_h * kernel_w]
        kappa_for_conv = kappa_original.repeat(kernel_h * kernel_w)
        b_term_for_conv = b_term_original.repeat(kernel_h * kernel_w)
        
        # Reshape W to [in_channels*kernel_h*kernel_w, out_channels]
        W_reshaped = W.reshape(-1, out_channels)
        
        # Now kappa_for_conv size should match first dimension of W_reshaped
        # W_reshaped[0] = in_channels * kernel_h * kernel_w
        
        if len(kappa_for_conv) != W_reshaped.shape[0]:
            # Size mismatch - try to fix
            if len(kappa_for_conv) < W_reshaped.shape[0]:
                repeat_factor = W_reshaped.shape[0] // len(kappa_for_conv)
                kappa_for_conv = kappa_for_conv.repeat(repeat_factor)
                b_term_for_conv = b_term_for_conv.repeat(repeat_factor)
                
                # If still not matching, pad with last value
                if len(kappa_for_conv) < W_reshaped.shape[0]:
                    pad_size = W_reshaped.shape[0] - len(kappa_for_conv)
                    kappa_for_conv = torch.cat([kappa_for_conv, kappa_for_conv[-1].repeat(pad_size)])
                    b_term_for_conv = torch.cat([b_term_for_conv, b_term_for_conv[-1].repeat(pad_size)])
            else:
                kappa_for_conv = kappa_for_conv[:W_reshaped.shape[0]]
                b_term_for_conv = b_term_for_conv[:W_reshaped.shape[0]]
        
        # Apply kappa to weights (Eq. 10)
        kappa_mat = torch.diag(kappa_for_conv)
        W_fused = torch.matmul(kappa_mat, W_reshaped)
        W_fused = W_fused.reshape(W.shape)
        
        # Apply b_term to bias (Eq. 9)
        # b_term_for_conv @ W_reshaped should give [out_channels]
        b_fused = bias + torch.sum(torch.matmul(torch.diag(b_term_for_conv), W_reshaped), dim=0)
        
        # Create new layer
        layer = _copy_layer(next_layer)
        layer.weight.data = W_fused.to(next_layer.weight.dtype)
        layer.BN.data = torch.tensor([1], device=device)
        layer.BN_before_ReLU.data = torch.tensor([0], device=device)
        
        # For set_bias, we need b_term to match in_channels of the original W
        # W shape: [out_channels, in_channels, H, W] -> we need b_term size [in_channels]
        b_term_for_bias = b_term_original[:in_channels] if len(b_term_original) >= in_channels else b_term_original.repeat((in_channels + len(b_term_original) - 1) // len(b_term_original))[:in_channels]
        
        layer.set_bias(bias=b_fused, W=W.permute(2, 3, 1, 0), b_term=b_term_for_bias)
        
    elif isinstance(next_layer, nn.Linear):
        W = next_layer.weight.data.clone().to(dtype)
        bias = next_layer.bias.data.clone().to(dtype) if next_layer.bias is not None else torch.zeros(next_layer.out_features, dtype=dtype, device=device)
        
        # For Linear layer, kappa and b_term need to match input features
        input_features = next_layer.in_features
        
        # Adjust kappa and b_term to match input features
        if len(kappa_original) != input_features:
            repeat_factor = input_features // len(kappa_original)
            remainder = input_features % len(kappa_original)
            
            if remainder == 0:
                kappa_adjusted = kappa_original.repeat(repeat_factor)
                b_term_adjusted = b_term_original.repeat(repeat_factor)
            else:
                kappa_adjusted = kappa_original.repeat(repeat_factor + 1)[:input_features]
                b_term_adjusted = b_term_original.repeat(repeat_factor + 1)[:input_features]
        else:
            kappa_adjusted = kappa_original
            b_term_adjusted = b_term_original
        
        kappa_mat = torch.diag(kappa_adjusted)
        W_fused = torch.matmul(kappa_mat, W.t()).t()
        b_fused = bias + torch.matmul(torch.diag(b_term_adjusted), W.t()).sum(dim=1)
        
        layer = _copy_layer(next_layer)
        layer.weight.data = W_fused.to(next_layer.weight.dtype)
        layer.bias.data = b_fused.to(next_layer.weight.dtype)
    
    fused_model.add_module(str(len(fused_model)), layer)
    
    if i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU):
        fused_model.add_module(str(len(fused_model)), _copy_layer(layers[i + 1]))
        i += 1
    
    return i + 1







def _copy_layer(orig_layer):
    """Deep copy of a layer with proper replacements."""
    if isinstance(orig_layer, nn.MaxPool2d):
        layer = MaxMinPool2D(kernel_size=orig_layer.kernel_size, 
                            stride=orig_layer.stride, 
                            padding=orig_layer.padding)
    elif isinstance(orig_layer, nn.Conv2d):
        layer = Conv2DWithBias(orig_layer.in_channels, orig_layer.out_channels, 
                               orig_layer.kernel_size, stride=orig_layer.stride, 
                               padding=orig_layer.padding)
        layer.weight.data = orig_layer.weight.data.clone()
    elif isinstance(orig_layer, nn.Linear):
        layer = nn.Linear(orig_layer.in_features, orig_layer.out_features)
        layer.weight.data = orig_layer.weight.data.clone()
        if orig_layer.bias is not None:
            layer.bias.data = orig_layer.bias.data.clone()
    elif isinstance(orig_layer, nn.ReLU):
        layer = nn.ReLU()
    elif isinstance(orig_layer, nn.Flatten):
        layer = nn.Flatten()
    else:
        layer = copy.deepcopy(orig_layer)
    return layer


def _copy_model_from_layers(fused_model, layers, start_idx=1):
    """Copy layers from list into Sequential model."""
    for i in range(start_idx, len(layers)):
        if 'dropout' in str(type(layers[i])).lower():
            continue
        
        fused_layer = _copy_layer(layers[i])
        
        if isinstance(layers[i], nn.Conv2d):
            W = layers[i].weight.data
            b = layers[i].bias.data if layers[i].bias is not None else torch.zeros(layers[i].out_channels, device=W.device)
            fused_layer.set_bias(bias=b)
        elif isinstance(layers[i], nn.Linear):
            fused_layer.weight.data = layers[i].weight.data.clone()
            if layers[i].bias is not None:
                fused_layer.bias.data = layers[i].bias.data.clone()
        
        fused_model.add_module(str(len(fused_model)), fused_layer)


class Conv2DWithBias(nn.Conv2d):
    """Convolutional layer with location-dependent bias."""
    
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)
        self.register_buffer('BN', torch.tensor([0]))
        self.register_buffer('BN_before_ReLU', torch.tensor([0]))
        self.bias_param = nn.Parameter(torch.zeros(9, self.out_channels))
        self.use_custom_bias = False
    
    def set_bias(self, bias, W=None, b_term=None):
        """
        Set location-dependent bias.
        
        Args:
            bias: The bias vector (output_channels,)
            W: Original kernel weights [H, W, in_channels, out_channels] (TF format)
            b_term: Scaling term for bias adjustment (in_channels,)
        """
        if b_term is None:
            b_term = [0.0]
        
        self.use_custom_bias = True
        device = self.bias_param.device
        dtype = self.bias_param.dtype
        
        # Ensure bias is on correct device and dtype
        bias = bias.to(device=device, dtype=dtype)
        
        if W is not None:
            W = W.to(device=device, dtype=dtype)
            # W shape: [H, W, in_channels, out_channels]
            H, W_dim, in_channels, out_channels = W.shape
            W_sum_2D = torch.sum(W, dim=(0, 1))  # Shape: [in_channels, out_channels]
            
            # Convert b_term to tensor
            if isinstance(b_term, torch.Tensor):
                b_term = b_term.to(device=device, dtype=dtype)
            else:
                b_term = torch.tensor(b_term, dtype=dtype, device=device)
            
            # Ensure b_term matches in_channels
            if b_term.numel() == 1:
                b_term = b_term.repeat(in_channels)
            elif len(b_term) != in_channels:
                if len(b_term) < in_channels:
                    # Repeat to fill
                    repeat_factor = (in_channels + len(b_term) - 1) // len(b_term)
                    b_term = b_term.repeat(repeat_factor)[:in_channels]
                else:
                    # Truncate
                    b_term = b_term[:in_channels]
            
            for i in range(9):
                if i == 0:
                    # Inner part - no padding effect
                    delta_sum_W = torch.zeros((in_channels, out_channels), dtype=dtype, device=device)
                elif i == 1:
                    # Top-left corner
                    delta_sum_W = (W_sum_2D - torch.sum(W[1:, 1:, :, :], dim=[0, 1]))
                elif i == 2:
                    # Top edge
                    delta_sum_W = torch.sum(W[:1, :, :, :], dim=[0, 1])
                elif i == 3:
                    # Top-right corner
                    delta_sum_W = (W_sum_2D - torch.sum(W[1:, :-1, :, :], dim=[0, 1]))
                elif i == 4:
                    # Right edge
                    delta_sum_W = torch.sum(W[:, -1:, :, :], dim=[0, 1])
                elif i == 5:
                    # Bottom-right corner
                    delta_sum_W = (W_sum_2D - torch.sum(W[:-1, :-1, :, :], dim=[0, 1]))
                elif i == 6:
                    # Bottom edge
                    delta_sum_W = torch.sum(W[-1:, :, :, :], dim=[0, 1])
                elif i == 7:
                    # Bottom-left corner
                    delta_sum_W = (W_sum_2D - torch.sum(W[:-1, 1:, :, :], dim=[0, 1]))
                elif i == 8:
                    # Left edge
                    delta_sum_W = torch.sum(W[:, :1, :, :], dim=[0, 1])
                
                # Calculate delta_bias = sum(b_term * delta_sum_W, axis=0)
                # b_term: [in_channels], delta_sum_W: [in_channels, out_channels]
                # Result should be [out_channels]
                # Method 1: element-wise multiply and sum
                delta_bias = torch.sum(b_term.unsqueeze(1) * delta_sum_W, dim=0)
                
                # Assign to bias_param
                self.bias_param.data[i] = bias - delta_bias
                
                if self.padding == 0 or self.BN != 1 or self.BN_before_ReLU == 1:
                    # Copy first bias to all positions for valid padding
                    for j in range(1, 9):
                        self.bias_param.data[j] = self.bias_param.data[0]
                    break
        else:
            # No W provided, just set same bias everywhere
            for i in range(9):
                self.bias_param.data[i] = bias




    def forward(self, inputs):
        result = super().forward(inputs)
        
        if self.use_custom_bias:
            if self.padding == 0 or self.BN != 1 or self.BN_before_ReLU == 1:
                result = result + self.bias_param[0].view(1, -1, 1, 1)
            else:
                result_0 = result[:, :, 1:-1, 1:-1] + self.bias_param[0].view(1, -1, 1, 1)
                result_1 = result[:, :, :1, :1] + self.bias_param[1].view(1, -1, 1, 1)
                result_2 = result[:, :, :1, 1:-1] + self.bias_param[2].view(1, -1, 1, 1)
                result_3 = result[:, :, :1, -1:] + self.bias_param[3].view(1, -1, 1, 1)
                result_4 = result[:, :, 1:-1, -1:] + self.bias_param[4].view(1, -1, 1, 1)
                result_5 = result[:, :, -1:, -1:] + self.bias_param[5].view(1, -1, 1, 1)
                result_6 = result[:, :, -1:, 1:-1] + self.bias_param[6].view(1, -1, 1, 1)
                result_7 = result[:, :, -1:, :1] + self.bias_param[7].view(1, -1, 1, 1)
                result_8 = result[:, :, 1:-1, :1] + self.bias_param[8].view(1, -1, 1, 1)
                
                top_row = torch.cat([result_1, result_2, result_3], dim=3)
                middle = torch.cat([result_8, result_0, result_4], dim=3)
                bottom_row = torch.cat([result_7, result_6, result_5], dim=3)
                result = torch.cat([top_row, middle, bottom_row], dim=2)
        
        return result


class MaxMinPool2D(nn.MaxPool2d):
    """Max or Min pooling based on sign."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('sign', torch.ones(1, 1, 1, 1))
    
    def forward(self, inputs):
        if self.sign.shape[1] != inputs.shape[1]:
            self.sign = torch.ones(1, inputs.shape[1], 1, 1, device=inputs.device, dtype=inputs.dtype)
        return super().forward(self.sign * inputs) * self.sign