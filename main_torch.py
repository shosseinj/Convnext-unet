import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import pickle as pkl
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from utils_torch import *  # You'll need to adapt utils for PyTorch

from torch.optim.lr_scheduler import ReduceLROnPlateau




import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os 

class Dataset:
    def __init__(
        self,
        data_name,
        logging_dir,
        flatten,
        ttfs_convert,
        ttfs_noise=0,
        data_path='./dataset/',
        input_size= (256, 256)
    ):
        self.name = data_name
        self.flatten=flatten
        self.data_path = data_path
        self.noise=ttfs_noise
        self.input_size = input_size
        self.logging_dir=logging_dir
        # Load original data.
        self.get_features_vectors()
        # In case of SNN, convert input data with TTFS coding.
        self.ttfss_convert=ttfs_convert
        if ttfs_convert: self.convert_ttfs()
        
    def get_features_vectors(self):
        """
        Load image datasets and transform into features. 
        """
        if 'MNIST' in self.name:
            self.input_shape, self.train_sample=(28, 28, 1), 1/64
            self.q, self.p = 1.0, 0.0
            self.num_of_classes = 10
            if self.name=='MNIST':
                train_data = datasets.MNIST(root='./data', train=True, download=True)
                test_data = datasets.MNIST(root='./data', train=False, download=True)
            else:
                train_data = datasets.FashionMNIST(root='./data', train=True, download=True)
                test_data = datasets.FashionMNIST(root='./data', train=False, download=True)
            self.x_train, self.y_train = train_data.data.numpy(), train_data.targets.numpy()
            self.x_test, self.y_test = test_data.data.numpy(), test_data.targets.numpy()
            self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0
            if self.flatten:
                self.x_train, self.x_test = self.x_train.reshape((len(self.x_train), -1)), self.x_test.reshape((len(self.x_test), -1))
            else:
                self.x_train, self.x_test = self.x_train.reshape(-1, 28, 28, 1), self.x_test.reshape(-1, 28, 28, 1)







        elif self.name == "KvasirSEG":

            from PIL import Image
            from sklearn.model_selection import train_test_split
            import glob

            self.num_of_classes = 2
            self.input_shape = (3, 256, 256)

            self.q = 1.0
            self.p = 0.0

            image_dir = os.path.join(self.data_path, "Kvasir-SEG", "images")
            mask_dir = os.path.join(self.data_path, "Kvasir-SEG", "masks")

            image_files = sorted(glob.glob(os.path.join(image_dir, "*")))

            images = []
            masks = []

            for img_path in image_files:

                filename = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, filename)

                if not os.path.exists(mask_path):
                    continue

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                img = img.resize(self.input_size)
                mask = mask.resize(self.input_size)

                img = np.array(img, dtype=np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))

                mask = np.array(mask, dtype=np.float32)
                mask = (mask > 127).astype(np.float32)

                images.append(img)
                masks.append(mask)

            images = np.array(images, dtype=np.float32)
            masks = np.array(masks, dtype=np.float32)

            (
                self.x_train,
                self.x_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                images,
                masks,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )







        elif 'CIFAR' in self.name:
            # CIFAR10 or CIFAR100 dataset.
            self.input_shape=(3,32, 32)
            self.q, self.p = 3.0, -3.0
            if self.name=='CIFAR10':
                self.num_of_classes = 10

                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])

                train_dataset = datasets.ImageFolder(os.path.join(self.data_path, 'train'), transform=transform_train)
                test_dataset = datasets.ImageFolder(os.path.join(self.data_path, 'test'), transform=transform_test)
                
                # Extract all data from ImageFolder using DataLoader
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
                
                self.x_train, self.y_train = next(iter(train_loader))
                self.x_test, self.y_test = next(iter(test_loader))
                
                # Convert to numpy and change from (N,C,H,W) to (N,H,W,C)
                self.x_train = self.x_train.numpy()
                self.x_test = self.x_test.numpy()
                self.y_train = self.y_train.numpy()
                self.y_test = self.y_test.numpy()
                
                self.mean_test, self.std_test = 0.4914, 0.2023
                
            else:
                # CIFAR100
                self.num_of_classes = 100
                train_data = datasets.CIFAR100(root='./data', train=True, download=True)
                test_data = datasets.CIFAR100(root='./data', train=False, download=True)
                self.mean_test, self.std_test = 121.936, 68.389
                self.x_train, self.y_train = train_data.data, np.array(train_data.targets)
                self.x_test, self.y_test = test_data.data, np.array(test_data.targets)
            
            # Scale to [-3, 3] range.
            self.x_test = (self.x_test - self.mean_test) / (self.std_test + 1e-7)
            self.x_train = (self.x_train - self.mean_test) / (self.std_test + 1e-7)

            
        self.x_train, self.x_test = self.x_train.astype('float32'), self.x_test.astype('float32')

        if self.name == "KvasirSEG":
            self.y_train = self.y_train.astype(np.float32)
            self.y_test = self.y_test.astype(np.float32)
        else:
            self.y_train = self.y_train.astype(np.int64)
            self.y_test = self.y_test.astype(np.int64)

        print ('Train data:', np.shape(self.x_train), np.shape(self.y_train))
        print ('Test data:', np.shape(self.x_test), np.shape(self.y_test))

    def convert_ttfs(self):
        """
        Convert input values into time-to-first-spike spiking times.
        """
        self.x_test, self.x_train = (self.x_test - self.p)/(self.q-self.p), (self.x_train - self.p)/(self.q-self.p)
        self.x_train, self.x_test=1 - np.array(self.x_train), 1 - np.array(self.x_test)
        self.x_test=np.maximum(0, self.x_test + np.random.randn(*self.x_test.shape).astype('float32') * self.noise)






# Model definitions
class SpikingDense(nn.Module):
    def __init__(self, in_features, units, name, X_n=1, outputLayer=False, 
                 robustness_params={}, kernel_regularizer=None, kernel_initializer=None):
        super(SpikingDense, self).__init__()
        self.units = units
        self.B_n = (1 + 0.5) * X_n
        self.outputLayer = outputLayer
        self.robustness_params = robustness_params
        
        # Register buffers for non-trainable parameters
        self.register_buffer('t_min_prev', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('alpha', torch.ones(units, dtype=torch.float32))
        
        # Weight and bias
        self.kernel = nn.Parameter(torch.empty(in_features, units, dtype=torch.float32))
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))
        
        # Initialize weights
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel)
        elif kernel_initializer == 'he_uniform':
            nn.init.kaiming_uniform_(self.kernel)
        else:
            nn.init.xavier_uniform_(self.kernel)
        
        self.name = name
    
    def set_params(self, t_min_prev, t_min):
        """Set timing parameters for the layer."""
        self.t_min_prev.fill_(t_min_prev)
        self.t_min.fill_(t_min)
        t_max = t_min + self.B_n
        self.t_max.fill_(t_max)
        return t_min, t_max
    
    def forward(self, tj):
        """Forward pass with spiking computation."""
        output = call_spiking(
            tj, self.kernel, self.D_i, 
            self.t_min_prev, self.t_min, self.t_max, 
            self.robustness_params
        )
        
        if self.outputLayer:
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            self.alpha.data = self.D_i / (self.t_min - self.t_min_prev)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x
        
        return output


class SpikingConv2D(nn.Module):
    def __init__(self, in_channels, filters, name, X_n=1, padding='same', 
                 kernel_size=(3,3), robustness_params={},
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.5) * X_n
        self.robustness_params = robustness_params
        
        # Register buffers
        self.register_buffer('t_min_prev', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('alpha', torch.ones(filters, dtype=torch.float32))
        self.register_buffer('BN', torch.tensor([0]))
        self.register_buffer('BN_before_ReLU', torch.tensor([0]))
        
        # Convolution weights (PyTorch format: out_channels, in_channels, H, W)
        self.kernel = nn.Parameter(
            torch.empty(filters, in_channels, kernel_size[0], kernel_size[1], dtype=torch.float32)
        )
        self.D_i = nn.Parameter(torch.zeros(9, filters, dtype=torch.float32))
        
        # Initialize weights
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel)
        elif kernel_initializer == 'he_uniform':
            nn.init.kaiming_uniform_(self.kernel)
        else:
            nn.init.xavier_uniform_(self.kernel)
        
        self.name = name
    
    def set_params(self, t_min_prev, t_min):
        """Set timing parameters for the layer."""
        self.t_min_prev.fill_(t_min_prev)
        self.t_min.fill_(t_min)
        t_max = t_min + self.B_n
        self.t_max.fill_(t_max)
        return t_min, t_max
    
    def forward(self, tj):
        """Forward pass with spiking computation."""
        batch_size = tj.shape[0]
        image_same_size = tj.shape[1]
        image_valid_size = image_same_size - self.kernel_size[0] + 1
        
        # Pad input
        padding_size = int(self.padding == 'same') * (self.kernel_size[0] // 2)
        if padding_size > 0:
            tj = F.pad(tj, (padding_size, padding_size, padding_size, padding_size), 
                      value=self.t_min.item())
        
        # Extract patches using unfold
        tj_patches = F.unfold(
            tj.permute(0, 3, 1, 2),  # (B, C, H, W)
            kernel_size=self.kernel_size,
            padding=0,
            stride=1
        )  # (B, C*K*K, L)
        
        # Reshape for computation
        tj_patches = tj_patches.permute(0, 2, 1)  # (B, L, C*K*K)
        W_flat = self.kernel.reshape(-1, self.filters)  # (C*K*K, F)
        
        if self.padding == 'valid' or self.BN != 1 or self.BN_before_ReLU == 1:
            tj_reshaped = tj_patches.reshape(-1, W_flat.shape[0])
            ti = call_spiking(
                tj_reshaped, W_flat, self.D_i[0],
                self.t_min_prev, self.t_min, self.t_max,
                self.robustness_params
            )
            if self.padding == 'valid':
                ti = ti.reshape(batch_size, image_valid_size, image_valid_size, self.filters)
            else:
                ti = ti.reshape(batch_size, image_same_size, image_same_size, self.filters)
        else:
            # Handle 9 different partitions (simplified version)
            # This is a complex case; you may need to adapt this based on your specific needs
            ti = call_spiking(
                tj_patches.reshape(-1, W_flat.shape[0]),
                W_flat, self.D_i[0],
                self.t_min_prev, self.t_min, self.t_max,
                self.robustness_params
            )
            ti = ti.reshape(batch_size, image_same_size, image_same_size, self.filters)
        
        return ti


class ModelTmax(nn.Module):
    """Custom model that tracks minimum spike times."""
    def __init__(self, *args, **kwargs):
        super(ModelTmax, self).__init__(*args, **kwargs)
        self.spiking_layers = []
    
    def forward(self, x):
        min_ti = []
        for layer in self.spiking_layers:
            x = layer(x)
            min_ti.append(torch.min(x))
        return x, min_ti


def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    """
    Calculates spiking times from which ReLU functionality can be recovered.
    PyTorch version.
    """
    # Quantize time if specified
    if robustness_params.get('time_bits', 0) != 0:
        # Simple quantization (you may need to implement proper fake quantization)
        scale = (t_min - t_min_prev) / (2**robustness_params['time_bits'] - 1)
        tj = t_min_prev + torch.round((tj - t_min_prev) / scale) * scale
    
    # Quantize weights if specified
    if robustness_params.get('weight_bits', 0) != 0:
        scale = (robustness_params['w_max'] - robustness_params['w_min']) / (2**robustness_params['weight_bits'] - 1)
        W = robustness_params['w_min'] + torch.round((W - robustness_params['w_min']) / scale) * scale
    
    # Calculate spiking threshold
    threshold = t_max - t_min - D_i
    
    # Calculate output spiking time
    ti = torch.matmul(tj - t_min, W) + threshold + t_min
    
    # Ensure valid spiking time
    ti = torch.where(ti < t_max, ti, t_max)
    
    # Add noise
    if robustness_params.get('noise', 0.0) > 0:
        ti = ti + torch.randn_like(ti) * robustness_params['noise']
    
    return ti

def create_unet_segmentation(input_channels, base_filters=64, BN=True, dropout=0.4, 
                             kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    """
    Create a U-Net architecture for binary segmentation.
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        base_filters: Base number of filters (doubled at each encoder level)
        BN: Whether to use Batch Normalization
        dropout: Dropout rate
        kernel_regularizer: Not used in PyTorch version
        kernel_initializer: Weight initialization method
    """
    
    class DoubleConv(nn.Module):
        """Double convolution block (Conv -> BN? -> ReLU -> Conv -> BN? -> ReLU)"""
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            layers = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not BN),
            ]
            if BN:
                layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not BN))
            if BN:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.double_conv = nn.Sequential(*layers)
            
            # Initialize weights
            for m in self.double_conv.modules():
                if isinstance(m, nn.Conv2d):
                    if kernel_initializer == 'he_uniform':
                        nn.init.kaiming_uniform_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.double_conv(x)
    
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder (downsampling path)
            self.enc1 = DoubleConv(input_channels, base_filters)
            self.enc2 = DoubleConv(base_filters, base_filters * 2)
            self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
            self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
            
            # Bottleneck
            self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
            
            # Decoder (upsampling path)
            self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)
            
            self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
            
            self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
            
            self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base_filters * 2, base_filters)
            
            # Dropout
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            
            # Output layer
            self.out_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
            
            # Max pooling
            self.pool = nn.MaxPool2d(2)
            
            # Initialize decoder convolutions
            for m in [self.up4, self.up3, self.up2, self.up1]:
                if kernel_initializer == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if kernel_initializer == 'he_uniform':
                nn.init.kaiming_uniform_(self.out_conv.weight)
            else:
                nn.init.xavier_uniform_(self.out_conv.weight)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0)
        
        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            
            # Bottleneck with dropout
            bottleneck = self.bottleneck(self.pool(enc4))
            bottleneck = self.dropout(bottleneck)
            
            # Decoder with skip connections
            dec4 = self.up4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)
            dec4 = self.dropout(dec4)
            
            dec3 = self.up3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)
            dec3 = self.dropout(dec3)
            
            dec2 = self.up2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.up1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            
            # Output
            output = self.out_conv(dec1)
            
            return output
    
    return UNet()


def create_vgg_segmentation(layers2D, kernel_size, layers1D, data, BN, dropout=0, 
                          kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    """
    Modified to use U-Net architecture instead of simple encoder-decoder.
    Now ignores layers2D, kernel_size, layers1D parameters and creates a U-Net.
    """
    return create_unet_segmentation(
        input_channels=data.input_shape[0] if isinstance(data.input_shape, (list, tuple)) else 3,
        base_filters=64,
        BN=BN,
        dropout=dropout,
        kernel_initializer=kernel_initializer
    )



def create_convnext_segmentation(layers2D, kernel_size, layers1D, data, BN, dropout=0, 
                          kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    """
    Modified to use U-Net architecture instead of simple encoder-decoder.
    Now ignores layers2D, kernel_size, layers1D parameters and creates a U-Net.
    """


    input_channels=data.input_shape[0] if isinstance(data.input_shape, (list, tuple)) else 3
    base_filters=64
    BN=BN
    dropout=dropout
    kernel_initializer=kernel_initializer
    

    class DoubleConv(nn.Module):
        """Double convolution block (Conv -> BN? -> ReLU -> Conv -> BN? -> ReLU)"""
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            layers = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not BN),
            ]
            if BN:
                layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not BN))
            if BN:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.double_conv = nn.Sequential(*layers)
            
            # Initialize weights
            for m in self.double_conv.modules():
                if isinstance(m, nn.Conv2d):
                    if kernel_initializer == 'he_uniform':
                        nn.init.kaiming_uniform_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.double_conv(x)
    
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder (downsampling path)
            self.enc1 = DoubleConv(input_channels, base_filters)
            self.enc2 = DoubleConv(base_filters, base_filters * 2)
            self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
            self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
            
            # Bottleneck
            self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
            
            # Decoder (upsampling path)
            self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)
            
            self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
            
            self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
            
            self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base_filters * 2, base_filters)
            
            # Dropout
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            
            # Output layer
            self.out_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
            
            # Max pooling
            self.pool = nn.MaxPool2d(2)
            
            # Initialize decoder convolutions
            for m in [self.up4, self.up3, self.up2, self.up1]:
                if kernel_initializer == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if kernel_initializer == 'he_uniform':
                nn.init.kaiming_uniform_(self.out_conv.weight)
            else:
                nn.init.xavier_uniform_(self.out_conv.weight)
            if self.out_conv.bias is not None:
                nn.init.constant_(self.out_conv.bias, 0)
        
        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            
            # Bottleneck with dropout
            bottleneck = self.bottleneck(self.pool(enc4))
            bottleneck = self.dropout(bottleneck)
            
            # Decoder with skip connections
            dec4 = self.up4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)
            dec4 = self.dropout(dec4)
            
            dec3 = self.up3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)
            dec3 = self.dropout(dec3)
            
            dec2 = self.up2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.up1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            
            # Output
            output = self.out_conv(dec1)
            
            return output
    
    return UNet()




def create_vgg_model_ReLU(layers2D, kernel_size, layers1D, data, BN, dropout=0, 
                          kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    """Create VGG-like ReLU network in PyTorch."""
    layers = []
    in_channels = data.input_shape[0]  # Assuming channels last format
    
    i_conv = 0
    for f in layers2D:
        if f != 'pool':
            i_conv += 1
            layers.append(nn.Conv2d(in_channels, f, kernel_size, padding='same'))
            if kernel_initializer == 'he_uniform':
                nn.init.kaiming_uniform_(layers[-1].weight)
            else:
                nn.init.xavier_uniform_(layers[-1].weight)
            
            layers.append(nn.ReLU())
            
            if BN:
                layers.append(nn.BatchNorm2d(f, dtype=torch.float32))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            in_channels = f
        else:
            layers.append(nn.MaxPool2d(2))
    
    layers.append(nn.Flatten())
    
    # Calculate flattened size
    with torch.no_grad():
        dummy_input = torch.randn(1, *data.input_shape)
        dummy_output = nn.Sequential(*layers)(dummy_input)
        flattened_size = dummy_output.shape[1]
    
    in_features = flattened_size
    for d in layers1D:
        layers.append(nn.Linear(in_features, d, dtype=torch.float32))
        if kernel_initializer == 'he_uniform':
            nn.init.kaiming_uniform_(layers[-1].weight)
        else:
            nn.init.xavier_uniform_(layers[-1].weight)
        
        layers.append(nn.ReLU())
        
        if BN:
            layers.append(nn.BatchNorm1d(d, dtype=torch.float32))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        in_features = d
    
    layers.append(nn.Linear(in_features, data.num_of_classes, dtype=torch.float32))
    
    return nn.Sequential(*layers)


def create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, X_n=1000,
                         robustness_params={}, kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    """Create VGG-like SNN in PyTorch."""
    model = ModelTmax()
    
    in_channels = data.input_shape[0]
    image_size = data.input_shape[0]
    j = 0
    
    # First conv layer
    layer = SpikingConv2D(
        in_channels, layers2D[0], 'conv2d_1',
        X_n[0] if isinstance(X_n, list) else X_n,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        padding='same', kernel_size=kernel_size,
        robustness_params=robustness_params
    )
    model.spiking_layers.append(layer)
    in_channels = layers2D[0]
    
    # Remaining conv layers
    for f in layers2D[1:]:
        if f != 'pool':
            j += 1
            layer = SpikingConv2D(
                in_channels, f, f'conv2d_{1+j}',
                X_n[j] if isinstance(X_n, list) else X_n,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
                padding='same', kernel_size=kernel_size,
                robustness_params=robustness_params
            )
            model.spiking_layers.append(layer)
            in_channels = f
        else:
            # Pooling layer (need to implement MaxMinPool2D for PyTorch)
            model.spiking_layers.append(MaxMinPool2D())
            image_size //= 2
    
    # Flatten
    model.spiking_layers.append(nn.Flatten())
    
    # Dense layers
    i_dense = 1
    input_features = (image_size ** 2) * layers2D[-2]
    
    for k, d in enumerate(layers1D):
        layer = SpikingDense(
            input_features, d, f'dense_{i_dense}',
            X_n[j] if isinstance(X_n, list) else X_n,
            robustness_params=robustness_params
        )
        model.spiking_layers.append(layer)
        input_features = d
        j += 1
        i_dense += 1
    
    # Output layer
    output_layer = SpikingDense(
        input_features, 1, f'dense_{i_dense}',
        outputLayer=True, robustness_params=robustness_params
    )
    # output_layer = SpikingDense(
    #     input_features, data.num_of_classes, f'dense_{i_dense}',
    #     outputLayer=True, robustness_params=robustness_params
    # )
    model.spiking_layers.append(output_layer)
    
    return model


def create_fc_model_ReLU(layers=2, N_hid=340, N_in=784, N_out=10):
    """Create 2-layer fully-connected ReLU network in PyTorch."""
    model_layers = []
    in_features = N_in
    
    for i in range(layers - 1):
        out_features = N_hid[i] if isinstance(N_hid, list) else N_hid
        model_layers.append(nn.Linear(in_features, out_features, dtype=torch.float32))
        model_layers.append(nn.ReLU())
        in_features = out_features
    
    model_layers.append(nn.Linear(in_features, N_out, dtype=torch.float32))
    
    return nn.Sequential(*model_layers)


def create_fc_model_SNN(layers, X_n=1000, robustness_params={}, N_hid=340, N_in=784, N_out=10):
    """Create 2-layer fully-connected SNN in PyTorch."""
    model = ModelTmax()
    in_features = N_in
    
    for i in range(layers - 1):
        out_features = N_hid[i] if isinstance(N_hid, list) else N_hid
        layer = SpikingDense(
            in_features, out_features, f'dense_{i+1}',
            X_n[i] if isinstance(X_n, list) else X_n,
            robustness_params=robustness_params
        )
        model.spiking_layers.append(layer)
        in_features = out_features
    
    # Output layer
    output_layer = SpikingDense(
        in_features, N_out, 'dense_output',
        outputLayer=True, robustness_params=robustness_params
    )
    model.spiking_layers.append(output_layer)
    
    return model

bce = nn.BCEWithLogitsLoss()
def train_epoch_segmentation(model, train_loader, optimizer, criterion, device):
    """Training function for segmentation tasks with mixup."""
    model.train()
    running_loss = 0.0
    skipped_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Normalize input
        data = torch.clamp(data, 0.0, 1.0)
        
        # For segmentation, target shape should be (B, 1, H, W) or (B, H, W)
        if target.dim() == 3:
            target = target.unsqueeze(1)
    
        # ===========================================
        
        optimizer.zero_grad()
        output = model(data)
        
        # Ensure output and target have same shape
        if output.shape != target.shape:
            output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        loss = criterion(output, target) +  0.2 * bce(output, target)
        
        # Check for NaN/Inf loss
        if not torch.isfinite(loss):
            logging.warning(f"Batch {batch_idx}: Loss is {loss.item()}. Skipping batch.")
            skipped_batches += 1
            optimizer.zero_grad()
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient in {name}")
        
        optimizer.step()
        
        # Calculate Dice coefficient for monitoring
        with torch.no_grad():
            pred = (torch.sigmoid(output) > 0.3).float()
            dice = dice_coefficient(pred, target)
        
        running_loss += loss.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{dice:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Skip': f'{skipped_batches}'
        })
    
    if skipped_batches > 0:
        logging.warning(f"Skipped {skipped_batches}/{len(train_loader)} batches due to NaN/Inf loss")
    
    return running_loss / max(1, len(train_loader) - skipped_batches)



def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()  # Return as Python float


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Jaccard) score."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()  # Return as Python float


# def test_segmentation(model, test_loader, criterion, device, threshold=0.3):
#     """Testing function for segmentation tasks with adaptive threshold."""
#     model.eval()
#     test_loss = 0
#     dice_scores = []
#     iou_scores = []
    
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
            
#             if target.dim() == 3:
#                 target = target.unsqueeze(1)
            
#             output = model(data)
            
#             if output.shape != target.shape:
#                 output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
            
#             test_loss += criterion(output, target).item()
            
#             # Calculate metrics with LOWER threshold (0.3 instead of 0.5)
#             pred = (torch.sigmoid(output) > threshold).float()
#             dice = dice_coefficient(pred, target)  # Returns Python float
#             iou = iou_score(pred, target)  # Returns Python float
            
#             # Now these are regular Python floats, no .cpu() needed
#             dice_scores.append(dice)
#             iou_scores.append(iou)
    
#     avg_loss = test_loss / len(test_loader)
#     avg_dice = np.mean(dice_scores)  # np.mean works on list of floats
#     avg_iou = np.mean(iou_scores)
    
#     return avg_loss, avg_dice, avg_iou

def evaluate_with_tta(model, test_loader, device, threshold=0.40):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='TTA'):
            data = data.to(device)
            
            # Original
            pred1 = torch.sigmoid(model(data))
            
            # Horizontal flip
            pred2 = torch.flip(torch.sigmoid(model(torch.flip(data, [-1]))), [-1])
            
            # Vertical flip  
            pred3 = torch.flip(torch.sigmoid(model(torch.flip(data, [-2]))), [-2])
            
            # Average
            pred = (pred1 + pred2 + pred3) / 3
            
            all_preds.append((pred > threshold).float().cpu())
            all_targets.append(target.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    dice = dice_coefficient(all_preds, all_targets)
    iou = iou_score(all_preds, all_targets)
    
    return dice, iou


def test_segmentation(model, test_loader, criterion, device, threshold=0.3, use_tta=True):
    """Testing function with optional TTA"""
    model.eval()
    test_loss = 0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            
            if target.dim() == 3:
                target = target.unsqueeze(1)
            
            if use_tta:
                # Use TTA for prediction
                pred = tta_predict(model, data, device, threshold)
                
                # For loss calculation, use original forward pass
                output = model(data)
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
            else:
                # Regular prediction
                output = model(data)
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
                test_loss += criterion(output, target).item()
                pred = (torch.sigmoid(output) > threshold).float()
            
            dice_scores.append(dice_coefficient(pred, target))
            iou_scores.append(iou_score(pred, target))
    
    avg_loss = test_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    return avg_loss, avg_dice, avg_iou

def find_best_threshold(model, test_loader, criterion, device):
    model.eval()

    thresholds = np.arange(0.1, 0.91, 0.05)

    best_threshold = 0.5
    best_iou = 0.0
    best_dice = 0.0

    with torch.no_grad():

        for threshold in thresholds:

            dice_scores = []
            iou_scores = []
            test_loss = 0.0

            for data, target in test_loader:

                data = data.to(device)
                target = target.to(device)

                if target.dim() == 3:
                    target = target.unsqueeze(1)

                output = model(data)

                if output.shape != target.shape:
                    output = F.interpolate(
                        output,
                        size=target.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                test_loss += criterion(output, target).item()

                pred = (torch.sigmoid(output) > threshold).float()

                dice_scores.append(
                    dice_coefficient(pred, target)
                )

                iou_scores.append(
                    iou_score(pred, target)
                )

            avg_dice = np.mean(dice_scores)
            avg_iou = np.mean(iou_scores)

            print(
                f"Threshold={threshold:.2f} "
                f"Dice={avg_dice:.4f} "
                f"IoU={avg_iou:.4f}"
            )

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_dice = avg_dice
                best_threshold = threshold

    print("\n========================")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best Dice      : {best_dice:.4f}")
    print(f"Best IoU       : {best_iou:.4f}")
    print("========================\n")

    return best_threshold, best_dice, best_iou


# Training function
from tqdm import tqdm

# def mixup_data(x, y, alpha=1.0):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



import math



def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # For classification tasks we keep MixUp optional.  Here we use plain loss.
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Gradient clipping with a higher threshold
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Grad': f'{total_norm:.2f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if isinstance(model, ModelTmax):
                output, _ = model(data)
            else:
                output = model(data)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total




class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1e-6):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        
    def forward(self, pred, target):
        # BCE with label smoothing
        target_smooth = target * 0.9 + 0.05  # Label smoothing
        bce = F.binary_cross_entropy_with_logits(pred, target_smooth)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)
        
        return self.weight_bce * bce + self.weight_dice * dice


import albumentations as A
from albumentations.pytorch import ToTensorV2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random


import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class KvasirSEGDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, is_train=True, target_size=352):
        self.images = images
        self.masks = masks
        self.is_train = is_train
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def _gaussian_blur_manual(self, tensor, kernel_size=9, sigma=4):
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        kernel_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2

        return F.conv2d(
            tensor,
            kernel.to(tensor.device),
            padding=padding
        )

    def _elastic_transform(self, image, mask,
                           alpha=150,
                           sigma=10):

        shape = image.shape[1:]

        dx = torch.randn(*shape) * sigma
        dy = torch.randn(*shape) * sigma

        dx = self._gaussian_blur_manual(
            dx.unsqueeze(0).unsqueeze(0),
            kernel_size=9,
            sigma=4
        )[0, 0]

        dy = self._gaussian_blur_manual(
            dy.unsqueeze(0).unsqueeze(0),
            kernel_size=9,
            sigma=4
        )[0, 0]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            indexing="ij"
        )

        grid_x = grid_x.float() + dx * alpha
        grid_y = grid_y.float() + dy * alpha

        grid_x = 2.0 * grid_x / (shape[1] - 1) - 1.0
        grid_y = 2.0 * grid_y / (shape[0] - 1) - 1.0

        grid = torch.stack(
            [grid_x, grid_y],
            dim=-1
        ).unsqueeze(0)

        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[0]

        mask = F.grid_sample(
            mask.unsqueeze(0).float(),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=False
        )[0]

        return image, mask

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        if image.dim() == 2:
            image = image.unsqueeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image.max() > 1:
            image = image / 255.0

        if self.is_train:

            # ----------------------------------
            # Horizontal Flip
            # ----------------------------------
            if random.random() < 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])

            # ----------------------------------
            # Vertical Flip
            # ----------------------------------
            if random.random() < 0.5:
                image = torch.flip(image, [-2])
                mask = torch.flip(mask, [-2])

            # ----------------------------------
            # Rotation
            # ----------------------------------
            if random.random() < 0.5:

                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST
                )

            # ----------------------------------
            # Affine
            # ----------------------------------
            if random.random() < 0.5:

                angle = random.uniform(-10, 10)

                translate = (
                    int(random.uniform(-0.1, 0.1) * image.shape[2]),
                    int(random.uniform(-0.1, 0.1) * image.shape[1])
                )

                scale = random.uniform(0.8, 1.2)

                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=0,
                    interpolation=TF.InterpolationMode.NEAREST
                )

            # ----------------------------------
            # Random Crop + Resize
            # ----------------------------------
            if random.random() < 0.5:

                H, W = image.shape[1:]

                crop_ratio = random.uniform(0.7, 1.0)

                crop_h = int(H * crop_ratio)
                crop_w = int(W * crop_ratio)

                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)

                image = TF.crop(
                    image,
                    top,
                    left,
                    crop_h,
                    crop_w
                )

                mask = TF.crop(
                    mask,
                    top,
                    left,
                    crop_h,
                    crop_w
                )

                image = TF.resize(
                    image,
                    [H, W],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.resize(
                    mask,
                    [H, W],
                    interpolation=TF.InterpolationMode.NEAREST
                )

            # ----------------------------------
            # Elastic
            # ----------------------------------
            if random.random() < 0.3:
                image, mask = self._elastic_transform(
                    image,
                    mask,
                    alpha=150,
                    sigma=10
                )

            # ----------------------------------
            # Brightness
            # ----------------------------------
            if random.random() < 0.5:

                image = TF.adjust_brightness(
                    image,
                    random.uniform(0.8, 1.2)
                )

                image = TF.adjust_contrast(
                    image,
                    random.uniform(0.8, 1.2)
                )

                image = TF.adjust_saturation(
                    image,
                    random.uniform(0.8, 1.2)
                )

            # ----------------------------------
            # Gaussian Blur
            # ----------------------------------
            if random.random() < 0.2:

                image = TF.gaussian_blur(
                    image,
                    kernel_size=5
                )

            # ----------------------------------
            # Gaussian Noise
            # ----------------------------------
            if random.random() < 0.3:

                noise = (
                    torch.randn_like(image) * 0.03
                )

                image = image + noise

            # ----------------------------------
            # Cutout
            # ----------------------------------
            if random.random() < 0.3:

                H, W = image.shape[1:]

                size = random.randint(
                    int(0.05 * H),
                    int(0.15 * H)
                )

                y = random.randint(
                    0,
                    H - size
                )

                x = random.randint(
                    0,
                    W - size
                )

                image[:, y:y+size, x:x+size] = 0

        image = torch.clamp(image, 0, 1)

        mask = (mask > 0.5).float()

        return image, mask
   
class GradualWarmupScheduler:
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)
    def step(self, epoch):
        if epoch < self.total_epoch:
            progress = epoch / self.total_epoch
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * ((1 - progress) / self.multiplier + progress)
        else:
            if not self.finished:
                self.finished = True
            if self.after_scheduler:
                self.after_scheduler.step(epoch - self.total_epoch)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: logits, target: binary [0,1]
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced segmentation."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for False Positives
        self.beta = beta    # Weight for False Negatives (focus on polyps)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = ((1-target) * pred).sum()
        fn = (target * (1-pred)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky

class CombinedTverskyFocalLoss(nn.Module):
    """Combined Tversky and Focal loss for best performance."""
    def __init__(self, tversky_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.tversky_weight * self.tversky(pred, target) + 
                self.focal_weight * self.focal(pred, target))
    
class BoundaryLoss(nn.Module):
    """Boundary-aware loss to improve segmentation edges"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Target boundary detection
        target_boundary = F.max_pool2d(
            1 - target, kernel_size=self.theta0, stride=1, padding=self.theta0//2
        ) - (1 - target)
        
        # Distance-weighted cross entropy for boundaries
        dist_map = F.max_pool2d(
            target_boundary, kernel_size=self.theta, stride=1, padding=self.theta//2
        )
        dist_map = dist_map / (dist_map.max() + 1e-8)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * (1 + dist_map)).mean()
        
        return weighted_bce

class HybridLoss(nn.Module):
    """Combined loss for best segmentation performance"""
    def __init__(self, boundary_weight=0.3, tversky_weight=0.4, focal_weight=0.3):
        super().__init__()
        self.boundary = BoundaryLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
    def forward(self, pred, target):
        # Add label smoothing to prevent overfitting
        target_smooth = target * 0.9 + 0.05
        
        boundary_loss = self.boundary(pred, target_smooth)
        tversky_loss = self.tversky(pred, target_smooth)
        focal_loss = self.focal(pred, target_smooth)
        
        return (self.boundary_weight * boundary_loss + 
                self.tversky_weight * tversky_loss + 
                self.focal_weight * focal_loss)
    


def mixup_segmentation(data, target, alpha=0.2):
        """Mixup for segmentation"""
        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)
        
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
        
        mixed_data = lam * data + (1 - lam) * data[index]
        mixed_target = lam * target + (1 - lam) * target[index]
        
        return mixed_data, mixed_target

class SimpleCombinedLoss(nn.Module):
    """Dice + BCE with label smoothing"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        target_smooth = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target_smooth).sum()
        dice = 1 - (2. * intersection + 1e-6) / (pred_sigmoid.sum() + target_smooth.sum() + 1e-6)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target_smooth)
        
        return self.dice_weight * dice + self.bce_weight * bce
    


# Replace SimpleCombinedLoss with this:
class StandardDiceBCELoss(nn.Module):
    """Dice + BCE WITHOUT label smoothing"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
    def forward(self, pred, target):
        # NO label smoothing! Use hard 0 and 1 targets.
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)
        
        return self.dice_weight * dice + self.bce_weight * bce

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()

bce_loss_fn = nn.BCEWithLogitsLoss()






class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)

        sobel_y = sobel_x.t()

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # 🔥 FORCE SAME DEVICE (IMPORTANT FIX)
        sobel_x = self.sobel_x.to(probs.device)
        sobel_y = self.sobel_y.to(probs.device)

        pred_edge = F.conv2d(probs, sobel_x, padding=1) + \
                    F.conv2d(probs, sobel_y, padding=1)

        target_edge = F.conv2d(targets, sobel_x, padding=1) + \
                      F.conv2d(targets, sobel_y, padding=1)

        return F.l1_loss(pred_edge, target_edge)
    

class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.4, bce_w=0.3, boundary_w=0.3):
        super().__init__()

        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss()

        self.dice_w = dice_w
        self.bce_w = bce_w
        self.boundary_w = boundary_w

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        total = (
            self.dice_w * dice_loss +
            self.bce_w * bce_loss +
            self.boundary_w * boundary_loss
        )

        return total
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha   # FN weight
        self.beta = beta     # FP weight
        self.gamma = gamma   # focusing
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        loss = (1 - tversky) ** self.gamma
        return loss.mean()



import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class L1MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.l1_loss(probs, targets.float())

class CombinedSegLoss(nn.Module):
    def __init__(self, bce_w=0.7, focal_w=0.3, l1_w=0.1, alpha=0.25, gamma=2.0):
        super().__init__()
        self.bce_w = bce_w
        self.focal_w = focal_w
        self.l1_w = l1_w
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLossWithLogits(alpha=alpha, gamma=gamma)
        self.l1 = L1MaskLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets.float())
        focal = self.focal(logits, targets)
        l1 = self.l1(logits, targets)
        return self.bce_w * bce + self.focal_w * focal + self.l1_w * l1
    
def tta_predict(model, image, device, threshold=0.40):
    """
    Apply Test-Time Augmentation to a single batch
    """
    with torch.no_grad():
        # Original prediction
        logits = model(image)
        prob_original = torch.sigmoid(logits)
        
        # Horizontal flip
        flipped_h = torch.flip(image, dims=[-1])
        logits_h = model(flipped_h)
        prob_h = torch.sigmoid(logits_h)
        prob_h = torch.flip(prob_h, dims=[-1])
        
        # Vertical flip
        flipped_v = torch.flip(image, dims=[-2])
        logits_v = model(flipped_v)
        prob_v = torch.sigmoid(logits_v)
        prob_v = torch.flip(prob_v, dims=[-2])
        
        # Average all predictions
        prob_avg = (prob_original + prob_h + prob_v) / 3.0
        
        return (prob_avg > threshold).float()

# Main execution
if __name__ == "__main__":
    # Create model
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("#### Creating the model ####")
    model = None
    start_epoch = 0


        
    start_time = time.time()

    # Set default dtype to float32 (equivalent to TF's float32)
    # torch.set_default_dtype(torch.float32)

    strtobool = (lambda s: s=='True')
    parser = argparse.ArgumentParser(description='TTFS')
    parser.add_argument('--data_name', type=str, default='KvasirSEG', help='(MNIST|CIFAR10|CIFAR100)')
    parser.add_argument('--logging_dir', type=str, default='./logs/ConvNeXt-attention-Gelu-upsample/input_size-352/depth64-dim3/start-84/', help='Directory for logging')
    parser.add_argument('--data_path', type=str, default='../data/', help='Directory for logging')
    # parser.add_argument('--checkpoint_path', type=str, default='', help='Directory for logging')
    parser.add_argument('--checkpoint_path', type=str, default='./logs/ConvNeXt-attention-Gelu-upsample/input_size-352/depth64-dim3/start2-82/checkpoints_KvasirSEG-ConvNeXt/1265-test0.84.pth', help='Directory for logging')
    parser.add_argument('--model_type', type=str, default='Gelu', help='(SNN|ReLU|Gelu)')
    parser.add_argument('--model_name', type=str, default='ConvNeXt', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
    parser.add_argument('--lr', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--escape_lr', type=float, default=1e-4, help='Learning rate for escape')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=0, help='Epochs. 0 -skip training')
    parser.add_argument('--input_size', type=tuple, default=(352, 352), help='Input size for the images')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Epochs. 0 -skip training')
    parser.add_argument('--testing', type=strtobool, default=True, help='Execute testing.')
    parser.add_argument('--training', type=strtobool, default=False, help='Execute training.')
    parser.add_argument('--load', type=str, default=False, help='Load before training.')
    parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
    parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
    parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
    parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
    parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
    parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles for t_max. 0 -disabled')
    args = parser.parse_args()

    args.model_name = args.data_name + '-' + args.model_name
    set_up_logging(args.logging_dir, args.model_name)  # Assuming this exists

    robustness_params = {
        'noise': args.noise,
        'time_bits': args.time_bits,
        'weight_bits': args.weight_bits,
        'w_min': args.w_min,
        'w_max': args.w_max,
        'latency_quantiles': args.latency_quantiles
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data object (assuming Dataset class returns PyTorch tensors)
    data = Dataset(
        args.data_name,
        args.logging_dir,
        flatten='FC' in args.model_name,
        ttfs_convert='SNN' in args.model_type,
        ttfs_noise=args.noise,
        data_path= args.data_path,
        input_size= args.input_size
    )
    # if not args.testing :
    # # Create data loaders
    #     train_loader = DataLoader(
    #         list(zip(data.x_train, data.y_train)),
    #         batch_size=args.batch_size,
    #         shuffle=True
    #     )
    # test_loader = DataLoader(
    #     list(zip(data.x_test, data.y_test)),
    #     batch_size=args.batch_size,
    #     shuffle=False
    # )



    train_dataset = KvasirSEGDataset(data.x_train, data.y_train, is_train=True, target_size=args.input_size[0])  # Pass target size to dataset
    test_dataset = KvasirSEGDataset(data.x_test, data.y_test, is_train=False, target_size=args.input_size[0])  # Pass target size to dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    best_acc = 0.0



    from models.convnext_unet import *

    if 'FC2' in args.model_name:
        if 'SNN' in args.model_type:
            model = create_fc_model_SNN(layers=2, robustness_params=robustness_params)
        elif 'ReLU' in args.model_type:
            model = create_fc_model_ReLU(layers=2)
    elif 'VGG' in args.model_name:
        if 'MNIST' in args.data_name:
            layers2D = [64, 64, 128, 128, 'pool', 256, 256, 256, 'pool', 
                       512, 512, 512, 'pool', 512, 512, 512, 'pool']
            layers1D = [512, 512]
        elif 'Kvasir' in args.data_name:
            layers2D = [
                64, 64, 'pool',
                128, 128, 'pool',
                256, 256, 256, 'pool',
                512, 512, 512, 'pool',
                512, 512, 512
            ]

            layers1D = [512, 512]

        else:
            layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool',
                       512, 512, 512, 'pool', 512, 512, 512, 'pool']
            layers1D = [512]
        
        kernel_size = (3, 3)
        BN = 'BN' in args.model_name
        
        if 'SNN' in args.model_type:
            model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, 
                                        robustness_params=robustness_params)
        elif 'ReLU' in args.model_type:
            if 'Kvasir' in args.data_name:
                model = create_vgg_segmentation(layers2D, kernel_size, layers1D, data, BN=BN, dropout= 0.4)
            else:
                model = create_vgg_model_ReLU(layers2D, kernel_size, layers1D, data, BN=BN, dropout= 0.4)
    


    elif 'ConvNeXt' in args.model_name:
        if 'Kvasir' in args.data_name:
            layers2D = [
                64, 64, 'pool',
                128, 128, 'pool',
                256, 256, 256, 'pool',
                512, 512, 512, 'pool',
                512, 512, 512
            ]

            layers1D = [512, 512]

        else:
            layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool',
                       512, 512, 512, 'pool', 512, 512, 512, 'pool']
            layers1D = [512]
        
        kernel_size = (3, 3)
        BN = 'BN' in args.model_name
        
        if 'SNN' in args.model_type:
            model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, 
                                        robustness_params=robustness_params)
        elif 'Gelu' in args.model_type:
            from models.convnext_attention import *
            model = ConvNeXtTinyUNetAttention(
                        # in_chans=3,
                        # num_classes=1,
                        dims=(96, 192, 384, 768),
                        # dims=(64, 128, 256, 512),
                        # depths=(3, 3, 9, 3),
                        depths=(2, 2, 4, 2),
                        # dims=(32, 64, 128, 256),  # Even smaller
                        # dims=(64, 128, 256, 512),
                        # depths=(1, 2, 2, 1),  # Very shallow

                        dropout=0.01,  # INCREASE from 0.1 to 0.3
                        drop_path_rate=0.01,  # INCREASE from 0.2 to 0.3

    #                         dropout=0.05,        # REDUCE from 0.1
    # drop_path_rate=0.1,  # REDUCE from 0.3 (this is very aggressive!)

                        # use_batch_norm=True
                    )
            print('loaded Relu version of ConvNeXt-Tiny')
    



    model = model.to(device)
    # print(model)
    from torchinfo import summary
    summary(
            model,
            input_size=(1, 3) + args.input_size,  # (batch, channels, H, W)
            device="cuda"
        )
    
    # Optimizer and loss
    # if 'VGG' in args.model_name and not BN and 'ReLU' in args.model_type:
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # else:
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    if 'Kvasir' in args.data_name:
        # optimizer = optim.AdamW(
        #         model.parameters(), 
        #         lr=args.lr, 
        #         weight_decay=0.001,  # Stronger regularization
        #         betas=(0.9, 0.999)
        #     )

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = DiceBCELoss(weight_bce=0.3, weight_dice=0.7)
        # criterion = FocalLoss(alpha=0.75, gamma=2.0)
        # criterion = CombinedTverskyFocalLoss(tversky_weight=0.7, focal_weight=0.3)
        # criterion = HybridLoss(boundary_weight=0.3, tversky_weight=0.4, focal_weight=0.3)
        # criterion = SimpleCombinedLoss(dice_weight=0.5, bce_weight=0.5, label_smoothing=0.1)
        # criterion = CombinedLoss(
        #         dice_w=0.4,
        #         bce_w=0.3,
        #         boundary_w=0.3
        #     )

#         criterion = FocalTverskyLoss(
#     alpha=0.3,   # less FN penalty
#     beta=0.7,    # more FP penalty
#     gamma=0.75   # moderate focusing
# )
        criterion = CombinedSegLoss(bce_w=0.7, focal_w=0.3, l1_w=0.1)
        
        # criterion = StandardDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer, T_0=50, T_mult=1, eta_min=1e-5
    #     )
    from torch.optim.lr_scheduler import CyclicLR
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=1e-5,      # Minimum LR (where you're stuck now)
    #     max_lr=5e-4,       # Maximum LR (to escape local minima)
    #     step_size_up=20,   # Increase LR over 20 epochs
    #     step_size_down=20, # Decrease LR over 20 epochs
    #     mode='triangular2', # Each cycle half the amplitude
    #     cycle_momentum=False
    # )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=15,  # Restart every 15 epochs
#     T_mult=2,  # Double period each restart
#     eta_min=1e-6
# )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=30,      # Restart every 30 epochs
    #     T_mult=2,    # Double period each restart
    #     eta_min=1e-5 # Minimum LR
    # )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=80,
        eta_min=1e-6
    )  



    warmup_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler
        )
        
      # Load checkpoint if exists
    # In your checkpoint loading section, modify to:
    if os.path.exists(args.checkpoint_path):
        logging.info("#### Loading checkpoint ####")
        
        # Handle both directory and file paths
        if os.path.isdir(args.checkpoint_path):
            pth_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.pth')]
            if pth_files:
                checkpoint_file = os.path.join(args.checkpoint_path, pth_files[0])
            else:
                logging.warning(f"No .pth file found in {args.checkpoint_path}")
                checkpoint_file = None
        else:
            checkpoint_file = args.checkpoint_path
        
        if not os.path.exists(checkpoint_file):
            logging.error(f"Checkpoint file not found: {checkpoint_file}")
            sys.exit(1)
        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # DON'T load scheduler state - it was a different scheduler!
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # REMOVE THIS
            
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            
            # Mark warmup as finished since we're resuming
            warmup_scheduler.finished = True
            
            logging.info(f"Resumed from epoch {checkpoint['epoch']}")
            logging.info(f"Previous best accuracy: {best_acc:.4f}")
            logging.info(f"Starting with fresh scheduler at LR: {optimizer.param_groups[0]['lr']:.6f}")
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.escape_lr
        logging.info(f"Restored LR: {optimizer.param_groups[0]['lr']}")
    if not args.checkpoint_path or args.checkpoint_path == '':
        checkpoint_dir = f"{args.logging_dir}checkpoints_{args.model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        args.checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')


    # # Training
    if  args.testing:
        best_threshold, best_dice, best_iou = find_best_threshold(
                model,
                test_loader,
                criterion,
                device
            )
        test_loss, test_dice, test_iou = test_segmentation(model, test_loader, criterion, device)
            
        logging.info(
                        f"First EvaluationTest Loss: {test_loss:.4f}, "
                        f"Test Dice: {test_dice:.4f}, "
                        f"Test IoU: {test_iou:.4f}")
        tta_dice, tta_iou = evaluate_with_tta(model, test_loader, device, threshold=0.40)
        print(f"TTA      → Dice: {tta_dice:.4f}, IoU: {tta_iou:.4f}")
        print(f"IMPROVEMENT: +{(tta_iou - test_iou)*100:.2f}% IoU")        

    if not args.testing and args.epochs > 0:
        logging.info("#### Training ####")
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        
        
        
        last_step = -1
 
        
        
        saved_lr = args.lr  # Default to args.lr if not resuming
        
        os.makedirs(f"{args.logging_dir}checkpoints_{args.model_name}", exist_ok=True)
        
      
        # Create scheduler with ADJUSTED warmup and base LR
        # scheduler =  ReduceLROnPlateau(
        #         optimizer, 
        #         mode='max', 
        #         factor=0.5, 
        #         patience=10, 
        #         verbose=True,
        #         min_lr=args.min_lr
        #     )  

        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=20,  # Restart every 20 epochs
        #     T_mult=2,  # Double the period after each restart
        #     eta_min=1e-6  # Minimum learning rate
        # )
# 
        
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=50,  # Decay every 10 epochs
        #     gamma=0.9      # Multiply LR by 0.9 each step
        # )
        


        # Log the actual starting LR
        current_lr = optimizer.param_groups[0]['lr']
        logging.info( f"initial LR: {current_lr:.6f}")
            
        # Training loop
        best_dice = 0

        
        if args.training:
            for epoch in range(start_epoch, args.epochs):
                
                
                # if epoch < 10:
                #     warmup_scheduler.step(epoch)
                # else:
                #     scheduler.step(epoch - 5)

                # In the main training section, replace the training/testing calls for KvasirSEG:

                if 'Kvasir' in args.data_name:
                    # Use segmentation-specific training/testing
                    train_loss = train_epoch_segmentation(model, train_loader, optimizer, criterion, device)
                    test_loss, test_dice, test_iou = test_segmentation(model, test_loader, criterion, device)
                    tta_dice, tta_iou = evaluate_with_tta(model, test_loader, device, threshold=0.40)

                    logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Test Loss: {test_loss:.4f}, "
                                f"Test Dice: {test_dice:.4f}, "
                                f"Test IoU: {test_iou:.4f}")
                    print(f"TTA      → Dice: {tta_dice:.4f}, IoU: {tta_iou:.4f}")
                    print(f"IMPROVEMENT: +{(tta_iou - test_iou)*100:.2f}% IoU")
                    # Flush immediately for Kvasir
                    for handler in logging.root.handlers:
                        handler.flush()
                    
                    test_acc  = test_iou
                else:
                    # Use classification-specific training/testing (existing code)
                    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                    test_loss, test_acc = test(model, test_loader, criterion, device)




                # scheduler.step()  # Step scheduler based on validation loss
                # scheduler.step(epoch + 1)
                scheduler.step(test_iou)
                logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Test Loss: {test_loss:.4f}")
                
                # Flush logging to disk immediately
                for handler in logging.root.handlers:
                    handler.flush()
                # Save checkpoint with full scheduler state
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # Now saving full state!
                    'best_acc': best_acc,
            
            
                }
                

                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(checkpoint_dict, 
                            f"{args.logging_dir}checkpoints_{args.model_name}/{epoch}-test{test_acc:.2f}.pth")
                    logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")
                    # Flush after saving
                    for handler in logging.root.handlers:
                        handler.flush()

    # Final testing
    # if args.testing :
    #     logging.info("#### Final test set accuracy testing ####")
    #     test_loss, test_acc = test(model, test_loader, criterion, device)
    #     fused_model = fuse_bn_torch(model.to(device), p=0.0, q=1.0, BN=True, BN_before_ReLU=False)
    #     test_loss, test_acc_fused = test(fused_model, test_loader, criterion, device)

    #     logging.info(f"Final testing accuracy is {test_acc:.2f}%.   fused testing accuracy is {test_acc_fused:.2f}%")
    
    # Save model
    if args.save and 'ReLU' in args.model_type:
        logging.info("#### Saving ReLU model ####")
        torch.save(model.state_dict(), f"{args.logging_dir}/{args.model_name}_weights.pth")
    
    print(f'### Total elapsed time [s]: {time.time() - start_time:.2f}')