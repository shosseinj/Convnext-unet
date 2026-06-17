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
    ):
        self.name = data_name
        self.flatten=flatten
        self.data_path = data_path
        self.noise=ttfs_noise
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

                img = img.resize((256, 256))
                mask = mask.resize((256, 256))

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



def train_epoch_segmentation(model, train_loader, optimizer, criterion, device):
    """Training function for segmentation tasks."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # For segmentation, target shape should be (B, 1, H, W) or (B, H, W)
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add channel dimension
        
        optimizer.zero_grad()
        output = model(data)
        
        # Ensure output and target have same shape
        if output.shape != target.shape:
            output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        loss = criterion(output, target)
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate Dice coefficient for monitoring
        with torch.no_grad():
            pred = (torch.sigmoid(output) > 0.5).float()
            dice = dice_coefficient(pred, target)  # Now returns Python float
        
        running_loss += loss.item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{dice:.4f}',  # dice is already a float
            'Grad': f'{total_norm:.2f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return running_loss / len(train_loader)




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


def test_segmentation(model, test_loader, criterion, device):
    """Testing function for segmentation tasks."""
    model.eval()
    test_loss = 0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if target.dim() == 3:
                target = target.unsqueeze(1)
            
            output = model(data)
            
            if output.shape != target.shape:
                output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
            
            test_loss += criterion(output, target).item()
            
            # Calculate metrics
            pred = (torch.sigmoid(output) > 0.5).float()
            dice = dice_coefficient(pred, target)  # Returns Python float
            iou = iou_score(pred, target)  # Returns Python float
            
            # Now these are regular Python floats, no .cpu() needed
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    avg_loss = test_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)  # np.mean works on list of floats
    avg_iou = np.mean(iou_scores)
    
    return avg_loss, avg_dice, avg_iou




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
        
        # MixUp is generally not used for segmentation tasks.  We keep it
        # commented out for reference but use plain BCE loss.
        # data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2)
        
        optimizer.zero_grad()
        
        if isinstance(model, ModelTmax):
            output, min_ti = model(data)
        # Use standard loss for segmentation
        loss = criterion(output, target)
        else:
            output = model(data)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        
        loss.backward()
        
        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Gradient clipping was set to 1.0 which was limiting updates.
        # We increase the threshold to 5.0 to allow larger steps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + 
                   (1 - lam) * predicted.eq(targets_b).sum().item())
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Grad': f'{total_norm:.2f}',
            'LR': f'{current_lr:.6f}'
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
    parser.add_argument('--logging_dir', type=str, default='./logs/ConvNeXt/start1/', help='Directory for logging')
    parser.add_argument('--data_path', type=str, default='./data/', help='Directory for logging')
    # parser.add_argument('--checkpoint_path', type=str, default='', help='Directory for logging')
    parser.add_argument('--checkpoint_path', type=str, default='./logs/start10/checkpoints_KvasirSEG-ConvNeXt/69_test0.49.pth', help='Directory for logging')
    parser.add_argument('--model_type', type=str, default='ReLU', help='(SNN|ReLU)')
    parser.add_argument('--model_name', type=str, default='ConvNeXt', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs. 0 -skip training')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Epochs. 0 -skip training')
    parser.add_argument('--testing', type=strtobool, default=False, help='Execute testing.')
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
        data_path= args.data_path
    )
    if not args.testing :
    # Create data loaders
        train_loader = DataLoader(
            list(zip(data.x_train, data.y_train)),
            batch_size=args.batch_size,
            shuffle=True
        )
    test_loader = DataLoader(
        list(zip(data.x_test, data.y_test)),
        batch_size=args.batch_size,
        shuffle=False
    )



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
        elif 'ReLU' in args.model_type:
            model = ConvNeXtTinyUNet(
                        in_chans=3,
                        num_classes=1,
                        dims=(96, 192, 384, 768),
                        depths=(3, 3, 9, 3)
                    )
            print('loaded Relu version of ConvNeXt-Tiny')
    



    model = model.to(device)
    # print(model)
    from torchinfo import summary
    summary(
            model,
            input_size=(1, 3, 256, 256),  # (batch, channels, H, W)
            device="cuda"
        )
    
    # Optimizer and loss
    if 'VGG' in args.model_name and not BN and 'ReLU' in args.model_type:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    if 'Kvasir' in args.data_name:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0
      # Load checkpoint if exists
    if os.path.exists(args.checkpoint_path):
        logging.info("#### Loading checkpoint ####")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        # Get the saved learning rate from optimizer
        saved_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Saved LR from checkpoint: {saved_lr:.6f} - started with {args.lr:.6f}")
        
        # Use the saved LR as the new base LR for remaining training
        # Reset optimizer to use this as starting point
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        estimated_step = start_epoch * len(train_loader)
        last_step = estimated_step
        logging.info(f"Resumed from epoch {checkpoint['epoch']}, "
                    f"estimated step: {estimated_step}, "
                    f"will warmup from {saved_lr:.6f}")
        
        # Calculate approximate step based on epoch progress
     
    if not args.checkpoint_path or args.checkpoint_path == '':
        checkpoint_dir = f"{args.logging_dir}checkpoints_{args.model_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        args.checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')


    # Training
    if not args.testing and args.epochs > 0:
        logging.info("#### Training ####")
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        
        best_acc = 0.0
        
        last_step = -1
 
        
        
        saved_lr = args.lr  # Default to args.lr if not resuming
        
        os.makedirs(f"{args.logging_dir}checkpoints_{args.model_name}", exist_ok=True)
        
      
        # Create scheduler with ADJUSTED warmup and base LR
        scheduler =  ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.95, 
                patience=3, 
                verbose=True
            )      
        # Log the actual starting LR
        current_lr = optimizer.param_groups[0]['lr']
        logging.info( f"initial LR: {current_lr:.6f}")
            
        # Training loop
        best_dice = 0
        for epoch in range(start_epoch, args.epochs):
            
            


            # In the main training section, replace the training/testing calls for KvasirSEG:

            if 'Kvasir' in args.data_name:
                # Use segmentation-specific training/testing
                train_loss = train_epoch_segmentation(model, train_loader, optimizer, criterion, device)
                test_loss, test_dice, test_iou = test_segmentation(model, test_loader, criterion, device)
                
                logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Test Loss: {test_loss:.4f}, "
                            f"Test Dice: {test_dice:.4f}, "
                            f"Test IoU: {test_iou:.4f}")
                test_acc  = test_iou
            else:
                # Use classification-specific training/testing (existing code)
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                test_loss, test_acc = test(model, test_loader, criterion, device)




            scheduler.step(test_loss)  # Step scheduler based on validation loss
            logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}")
            
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
                          f"{args.logging_dir}checkpoints_{args.model_name}/{epoch}_test{test_acc:.2f}.pth")
                logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")

    # Final testing
    if args.testing :
        logging.info("#### Final test set accuracy testing ####")
        test_loss, test_acc = test(model, test_loader, criterion, device)
        fused_model = fuse_bn_torch(model.to(device), p=0.0, q=1.0, BN=True, BN_before_ReLU=False)
        test_loss, test_acc_fused = test(fused_model, test_loader, criterion, device)

        logging.info(f"Final testing accuracy is {test_acc:.2f}%.   fused testing accuracy is {test_acc_fused:.2f}%")
    
    # Save model
    if args.save and 'ReLU' in args.model_type:
        logging.info("#### Saving ReLU model ####")
        torch.save(model.state_dict(), f"{args.logging_dir}/{args.model_name}_weights.pth")
    
    print(f'### Total elapsed time [s]: {time.time() - start_time:.2f}')