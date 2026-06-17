# import logging
# import os
# import sys
# import numpy as np
# import tensorflow as tf


# def set_up_logging(logging_dir, model_name):
#     """
#     Set up logging for the simulation.
#     """
#     os.makedirs(logging_dir, exist_ok=True)
#     logging.basicConfig(
#         level=logging.DEBUG,
#         handlers=[
#            logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
#            logging.StreamHandler(sys.stdout),
#         ],
#     )
#     mpl_logger = logging.getLogger("matplotlib")
#     mpl_logger.setLevel(logging.WARNING)


# def get_optimizer(lr):
#     """
#     Get optimizer for the training on MNIST/Fashion-MNIST dataset.
#     """
#     learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=lr,
#         decay_steps=5000,  
#         decay_rate=0.9, 
#     )
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule) 
#     return optimizer


# class Conv2DWithBias(tf.keras.layers.Conv2D):
#     """
#     Convolutional layer with potentially different bias at different locations.
#     """
#     def build(self, input_shape):
#         super().build(input_shape)
#         # These two variables determine the processing of the layer.
#         self.BN=tf.Variable(tf.constant([0]), name='BN', trainable=False)
#         self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU', trainable=False)
        
#     def set_bias(self, bias, W=None, b_term=[0.0]):
#         """
#         Creates bias variable and changes bias on certain locations when needed.
#         """
#         # The bias Variable is added in this function and can have 9 potential values for each filter.
#         # W can represent the kernel before fusion and b_term corresponds to the term which is multiplied with kernel (see Eqs. 9, 11, etc.).  
#         self.bias = self.add_weight(shape=(9, self.filters), initializer='zeros', dtype=tf.float64, name='bias')
#         self.use_custom_bias = True
#         self.use_bias = False  # disable standard bias logic
#         if W is not None: 
#             # Calculate the overall kernel summation.
#             W_sum_2D  = tf.math.reduce_sum(W, axis=(0, 1))
#             b_term = b_term[:tf.shape(W)[2]]
#         for i in range(9):
#             # delta_sum_W calculates kernel summation of the weights which correspond to the zero-padded inputs for the particular image part. 
#             if i==0:
#                 # This branch corresponds to the inner part of the image which has unchanged bias.
#                 b_term=tf.cast(b_term, dtype=tf.float64)
#                 # delta_sum_W is equal to 0 which yields unchanged bias.
#                 delta_sum_W = tf.zeros((tf.shape(b_term)[0], 1), dtype=tf.float64)
#             elif i==1:
#                 # This branch corresponds to the top left corner of the image, etc. 
#                 delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, 1:, :, :], axis=[0, 1]))
#             elif i==2:
#                 delta_sum_W = tf.reduce_sum(W[:1, :, :, :], axis=[0, 1])
#             elif i==3:
#                 delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, :-1, :, :], axis=[0, 1]))
#             elif i==4:
#                  delta_sum_W = tf.reduce_sum(W[:, -1:, :, :], axis=[0, 1])
#             elif i==5:
#                 delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, :-1, :, :], axis=[0, 1]))
#             elif i==6:
#                 delta_sum_W = tf.reduce_sum(W[-1:, :, :, :], axis=[0, 1])
#             elif i==7:
#                 delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, 1:, :, :], axis=[0, 1]))
#             elif i==8:
#                 delta_sum_W = tf.reduce_sum(W[:, :1, :, :], axis=[0, 1])
#             # See Eqs. 11, 14.  
#             delta_bias = tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), delta_sum_W), axis=0)
#             # The bias is decreased for the value which corresponds to the terms which come from zero-padding. 
#             self.bias[i].assign(bias - delta_bias)
#             # When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias and we break from the for loop. 
#             if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: break
                
#     def call(self, inputs):
#         # Convolution operation is initially called. 
#         #result = self._convolution_op(inputs, self.kernel)  #incompatible with never TF
#         result = super().call(inputs)
#         if self.use_custom_bias:
#             if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1:
#                 #When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias.
#                 result=result + self.bias[0]
#             else:
#                 #Otherwise we add different bias to 9 different locations.
#                 result_0 = result[:, 1:-1, 1:-1, :] + self.bias[0]
#                 result_1 = result[:, :1, :1, :] + self.bias[1]
#                 result_2 = result[:, :1, 1:-1, :] + self.bias[2]
#                 result_3 = result[:, :1, -1:, :] + self.bias[3]
#                 result_4 = result[:, 1:-1, -1:, :] + self.bias[4]
#                 result_5 = result[:, -1:, -1:, :] + self.bias[5]
#                 result_6 = result[:, -1:, 1:-1, :] + self.bias[6]
#                 result_7 = result[:, -1:, :1, :] + self.bias[7]
#                 result_8 = result[:, 1:-1, :1, :] + self.bias[8]
#                 # Parts of image are concatenated to generate a complete output.
#                 top_row = tf.concat([result_1, result_2, result_3], axis=2)
#                 middle = tf.concat([result_8, result_0, result_4], axis=2)
#                 bottom_row = tf.concat([result_7, result_6, result_5], axis=2)
#                 result = tf.concat([top_row, middle, bottom_row], axis=1)
#         return result  


# class MaxMinPool2D(tf.keras.layers.MaxPool2D):
#     """
#     Max Pooling or Min Pooling operation, depends on the sign of the batch normalization layer before.
#     """
#     def build(self, input_shape):
#         super().build(input_shape)
#         # By default the sign is set to 1, which yields max pooling functionality.
#         # The sign variable can be changed for some channels when batch normalization is fused with the next convolutonal layer and it changes the sign of the weights. 
#         self.sign=tf.Variable(tf.constant(np.ones((1, 1, 1, input_shape[-1]))), dtype=tf.float64, name='sign', trainable=False)
#     def call(self, inputs):
#         # Max pooling functionality is called on (self.sign*inputs) input. 
#         return super().call(self.sign*inputs)*self.sign


# def copy_layer(orig_layer):
#     """
#     Deep copy of a layer with MaxPooling2D layer being replaced with MaxMinPool2D and Conv2D with Conv2DWithBias layer.
#     """
#     config = orig_layer.get_config()
#     if 'pool' in orig_layer.name:
#         layer = MaxMinPool2D()
#     elif 'conv' in orig_layer.name: 
#         config['use_bias']=False
#         layer = Conv2DWithBias.from_config(config)
#     else:
#         layer = type(orig_layer).from_config(config)
#     layer.build(orig_layer.input_shape)
#     return layer


# def copy_model(fused_model, model, i):
#     """
#     Deep copy of a model and exchange Conv with Conv2DWithBias layers and MaxPool with MaxMinPool layers.
#     The layers which are not used during inference are dropped.
#     """
#     while i < len(model.layers):
#         while 'dropout' in model.layers[i].name or 'activity_regularization' in model.layers[i].name: i+=1
#         # Deep copy of layer.
#         fused_layer = copy_layer(model.layers[i])
#         if 'conv' in model.layers[i].name:
#             W, b = model.layers[i].get_weights()
#             # Set parameters of Conv2DWithBias layer.
#             fused_layer.set_weights([W, np.array([0]), np.array([0])])
#             fused_layer.set_bias(bias=b)
#         elif 'dense' in model.layers[i].name:
#             # Set parameters of fully-connected layer.
#             fused_layer.set_weights(model.layers[i].get_weights())
#         fused_model.add(fused_layer)
#         i+=1


# def fuse_bn(model, p, q, optimizer, BN = True, BN_before_ReLU = False):
#     """
#     Creates new models which:
#         Fuses all (imaginary) batch normalization layers; 
#         Changes bias on locations where it is needed; 
#         Transforms MaxPooling layers in MaxMinPooling layers and Conv2D layers in Conv2DWithBias.  
#     """
#     fused_model = tf.keras.Sequential()
#     # Add input layer.
#     fused_model.add(copy_layer(model.layers[0]))
#     i=1
#     # If condition is satisfied, there is an imaginary batch normalization layer which is merged.
#     if not (p==0 and q==1): i = fuse_imaginary_bn(fused_model, model, p, q)
#     if BN:
#         # There are batch normalization layers.
#         if BN_before_ReLU:
#             # Batch normalization layers are always found before ReLU activation function.
#             while i<len(model.layers):
#                 if 'batch_norm' in model.layers[i].name:
#                     # Fuse this batch normalization layer with previous convolutional or fully-connected layer. 
#                     i = fuse_bn_before_activation(fused_model, model, i)
#                 if i==(len(model.layers)-1):
#                     # Add last Dense layer with 'softmax'.
#                     layer = copy_layer(model.layers[-2])
#                     layer.set_weights(model.layers[-2].get_weights())
#                     fused_model.add(layer)
#                     fused_model.add(copy_layer(model.layers[-1]))
#                 i+=1
#         else:
#             # Batch normalization layers are always found after ReLU activation function.
#             while i<len(model.layers): 
#                 # Add first Dense or Convolutional layer if there was no imaginary batch normalization.
#                 if (p==0 and q==1) and i==1:
#                     layer = copy_layer(model.layers[1])
#                     if 'conv' in model.layers[1].name:
#                         kernel, bias = model.layers[1].get_weights()
#                         # Set BN flag to 0 and BN_before_ReLU to 0.
#                         layer.set_weights([kernel, np.array([0]), np.array([0])])
#                         # Bias is same for all locations.
#                         layer.set_bias(bias)
#                     else:
#                         layer.set_weights(model.layers[1].get_weights())
#                     fused_model.add(layer)
#                     fused_model.add(copy_layer(model.layers[2]))
#                 if 'batch_norm' in model.layers[i].name:
#                     # Fuse this batch normalization layer with next convolutional or fully-connected layer.
#                     i = fuse_bn_after_activation(fused_model, model, i)
#                 i+=1
#     else:
#         # If there is no batch normalization layers, copy model such that Conv2D and MaxPooling layers are replaced with ConvWithBias and MaxMinPooling respectively. 
#         copy_model(fused_model, model, i)
#     fused_model.compile(metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=optimizer)  
#     return fused_model


# def fuse_imaginary_bn(fused_model, model, p, q): 
#     """
#     Fuse an imaginary batch normalization layer due to an input on arbitrary [p, q] range different from [0, 1].
#     """
#     first_layer = model.layers[1]
#     input_image_shape, _, input_channels, _ = tf.shape(first_layer.kernel)
#     kappa = tf.cast(tf.fill((input_channels), value=q-p), dtype=tf.float64)
#     b_term=tf.cast(tf.fill((input_channels), value=p), dtype=tf.float64)
#     if 'conv' in first_layer.name:
#         kappa=tf.tile(kappa, [input_image_shape**2])
#         b_term=tf.tile(b_term, [input_image_shape**2])
#     W = tf.reshape(first_layer.kernel, (-1, first_layer.filters))
#     kappa = tf.linalg.diag(kappa)
#     W_fused = tf.matmul(kappa, W)
#     # See Eq. 13. 
#     W_fused = tf.reshape(W_fused, tf.shape(first_layer.kernel))   
#     # See Eq. 12. 
#     b_fused = first_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
#     # Copy first convolutional or fully-connected layer.
#     layer = copy_layer(first_layer)
#     if 'conv' in first_layer.name:
#         # Set BN flag to 1 and BN_before_ReLU to 0.
#         layer.set_weights([W_fused, np.array([1]), np.array([0])])
#         # Create bias which will have 9 different values. 
#         # Those values are generated by subtracting from b_fused the terms which come through padded input.
#         # The obtained results is in Eq. 14. 
#         layer.set_bias(bias=b_fused, W=first_layer.kernel, b_term=b_term)
#     else:
#         layer.set_weights([W_fused, b_fused])
#     fused_model.add(layer)
#     fused_model.add(copy_layer(model.layers[2]))  
#     return 3


# def fuse_bn_before_activation(fused_model, model, i):
#     """
#     Fuses batch normalization layer with previous layer.
#     """
#     bn = model.layers[i]
#     kappa = tf.linalg.diag(bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance))
#     previous_layer = model.layers[i-1]
#     output_shape = tf.shape(previous_layer.kernel)[-1]
#     W = tf.reshape(previous_layer.kernel, (-1, output_shape))
#     # See Eq. 8.
#     W_fused = tf.transpose(tf.matmul(kappa, tf.transpose(W)))
#     W_fused = tf.reshape(W_fused, tf.shape(previous_layer.kernel))  
#     # See Eq. 7.
#     b_fused = bn.beta - bn.moving_mean*tf.linalg.diag_part(kappa)
#     b_fused += tf.squeeze(tf.matmul(kappa, previous_layer.bias[:, tf.newaxis]))
#     layer = copy_layer(previous_layer)
#     if 'conv' in previous_layer.name:
#         # Set BN flag to 1 and BN_before_ReLU to 1.
#         layer.set_weights([W_fused, np.array([1]), np.array([1])])
#         # Bias is same everywhere.
#         layer.set_bias(b_fused)
#     else:
#         layer.set_weights([W_fused, b_fused])
#     fused_model.add(layer)
#     fused_model.add(copy_layer(model.layers[i+1]))
#     # Skip Dropout and ActivityRegularization layers. 
#     while 'dropout' in model.layers[i+2].name or 'activity_regularization' in model.layers[i+2].name: i+=1
#     # Add Flatten layer when it appears. 
#     if 'flatten' in model.layers[i+2].name or 'pool' in model.layers[i+2].name: 
#         fused_model.add(copy_layer(model.layers[i+2]))
#         i+=1
#     return i+1

        
# def fuse_bn_after_activation(fused_model, model, i): 
#     """
#     Fuse batch normalization with following layer.
#     """
#     bn = model.layers[i]
#     kappa = bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance)
#     # See Eq. 9. 
#     b_term = bn.beta - bn.moving_mean*kappa
#     # Skip Dropout and ActivityRegularization layers. 
#     while 'dropout' in model.layers[i+1].name: i+=1
#     # if there is a MaxPooling layer in before the next parameterized layer, the sign will be changed for the channels where it is needed. 
#     if 'max_pool' in model.layers[i+1].name:
#         mp = model.layers[i+1]
#         mmp = MaxMinPool2D() 
#         mmp.build(mp.input_shape)
#         # Change sign to the sign of kappa. 
#         mmp.sign.assign(tf.math.sign(kappa)[tf.newaxis, tf.newaxis, tf.newaxis, :])
#         fused_model.add(mmp)
#         i+=1
#         if 'dropout' in model.layers[i+1].name: i+=1
#     if 'flatten' in model.layers[i+1].name:
#         # Add Flatten layer when it appears.
#         fused_model.add(copy_layer(model.layers[i+1]))
#         ft = model.layers[i+1]
#         kappa = tf.tile(kappa, [ft.output_shape[-1]//ft.input_shape[-1]])
#         b_term = tf.tile(tf.squeeze(b_term), [ft.output_shape[-1]//ft.input_shape[-1]])
#         i+=1
#     next_layer = model.layers[i+1]
#     input_image_shape = tf.shape(next_layer.kernel)[0]
#     output_shape = tf.shape(next_layer.kernel)[-1]
#     if 'conv' in next_layer.name:
#         kappa=tf.tile(kappa, [input_image_shape**2])
#         b_term=tf.tile(b_term, [input_image_shape**2])
#     W = tf.reshape(next_layer.kernel, (-1, output_shape))
#     kappa = tf.linalg.diag(kappa)
#     # See Eq. 10. 
#     W_fused = tf.matmul(kappa, W)
#     W_fused = tf.reshape(W_fused, tf.shape(next_layer.kernel))    
#     # See Eq. 9. 
#     b_fused = next_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
#     layer = copy_layer(next_layer)
#     if 'conv' in next_layer.name:
#         # Set BN flag to 1 and BN_before_ReLU to 0.
#         layer.set_weights([W_fused, np.array([1]), np.array([0])])
#         # Create bias which will have 9 different values. 
#         # Those values are generated by subtracting from b_fused the terms which come through padded input.
#         # The obtained results is in Eq. 11.
#         layer.set_bias(bias=b_fused, W=next_layer.kernel, b_term=b_term)
#     else:
#         layer.set_weights([W_fused, b_fused])
#     fused_model.add(layer)
#     if (i+1)!=len(model.layers)-1:
#         fused_model.add(copy_layer(model.layers[i+2])) 
#     return i+2



import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn


def set_up_logging(logging_dir, model_name):
    """
    Set up logging for the simulation.
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
           logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
           logging.StreamHandler(sys.stdout),
        ],
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)


def get_optimizer(model, lr):
    """
    Get optimizer for the training on MNIST/Fashion-MNIST dataset.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/5000))
    return optimizer, scheduler


class Conv2DWithBias(nn.Conv2d):
    """
    Convolutional layer with potentially different bias at different locations.
    """
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False
        super().__init__(*args, **kwargs)
        # These two variables determine the processing of the layer.
        self.register_buffer('BN', torch.tensor([0]))
        self.register_buffer('BN_before_ReLU', torch.tensor([0]))
        self.use_custom_bias = False
        
    def set_bias(self, bias, W=None, b_term=[0.0]):
        """
        Creates bias variable and changes bias on certain locations when needed.
        """
        self.bias_param = nn.Parameter(torch.zeros(9, self.out_channels, dtype=torch.float64))
        self.use_custom_bias = True
        
        if W is not None: 
            W_sum_2D = torch.sum(W, dim=(0, 1))
            b_term = b_term[:W.shape[2]]
            
        for i in range(9):
            if i==0:
                b_term = b_term.to(torch.float64)
                delta_sum_W = torch.zeros((b_term.shape[0], 1), dtype=torch.float64)
            elif i==1:
                delta_sum_W = (W_sum_2D - torch.sum(W[1:, 1:, :, :], dim=[0, 1]))
            elif i==2:
                delta_sum_W = torch.sum(W[:1, :, :, :], dim=[0, 1])
            elif i==3:
                delta_sum_W = (W_sum_2D - torch.sum(W[1:, :-1, :, :], dim=[0, 1]))
            elif i==4:
                 delta_sum_W = torch.sum(W[:, -1:, :, :], dim=[0, 1])
            elif i==5:
                delta_sum_W = (W_sum_2D - torch.sum(W[:-1, :-1, :, :], dim=[0, 1]))
            elif i==6:
                delta_sum_W = torch.sum(W[-1:, :, :, :], dim=[0, 1])
            elif i==7:
                delta_sum_W = (W_sum_2D - torch.sum(W[:-1, 1:, :, :], dim=[0, 1]))
            elif i==8:
                delta_sum_W = torch.sum(W[:, :1, :, :], dim=[0, 1])
                
            delta_bias = torch.sum(torch.matmul(torch.diag(b_term), delta_sum_W), dim=0)
            self.bias_param.data[i] = bias - delta_bias
            
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: 
                break
                
    def forward(self, inputs):
        result = super().forward(inputs)
        if self.use_custom_bias:
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1:
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
    """
    Max Pooling or Min Pooling operation, depends on the sign of the batch normalization layer before.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('sign', torch.ones(1, 1, 1, 1))
        
    def forward(self, inputs):
        if self.sign.shape[-1] != inputs.shape[1]:
            self.sign = torch.ones(1, inputs.shape[1], 1, 1, device=inputs.device, dtype=inputs.dtype)
        return super().forward(self.sign * inputs) * self.sign


def copy_layer(orig_layer):
    """
    Deep copy of a layer with MaxPooling2D layer being replaced with MaxMinPool2D and Conv2D with Conv2DWithBias layer.
    """
    if isinstance(orig_layer, nn.MaxPool2d):
        layer = MaxMinPool2D(kernel_size=orig_layer.kernel_size, stride=orig_layer.stride, padding=orig_layer.padding)
    elif isinstance(orig_layer, nn.Conv2d):
        layer = Conv2DWithBias(orig_layer.in_channels, orig_layer.out_channels, orig_layer.kernel_size,
                               stride=orig_layer.stride, padding=orig_layer.padding)
        layer.weight.data = orig_layer.weight.data.clone()
    elif isinstance(orig_layer, nn.Linear):
        layer = nn.Linear(orig_layer.in_features, orig_layer.out_features)
        layer.weight.data = orig_layer.weight.data.clone()
        layer.bias.data = orig_layer.bias.data.clone()
    elif isinstance(orig_layer, nn.ReLU):
        layer = nn.ReLU()
    elif isinstance(orig_layer, nn.Flatten):
        layer = nn.Flatten()
    else:
        layer = orig_layer
    return layer


def copy_model(fused_model, model, i):
    """
    Deep copy of a model and exchange Conv with Conv2DWithBias layers and MaxPool with MaxMinPool layers.
    """
    layers = list(model.children())
    while i < len(layers):
        while 'dropout' in str(type(layers[i])) or 'activity_regularization' in str(type(layers[i])):
            i += 1
            if i >= len(layers):
                return
        fused_layer = copy_layer(layers[i])
        if isinstance(layers[i], nn.Conv2d):
            W = layers[i].weight.data
            b = layers[i].bias.data if layers[i].bias is not None else torch.zeros(layers[i].out_channels)
            fused_layer.set_bias(bias=b)
        fused_model.add_module(str(i), fused_layer)
        i += 1


def fuse_bn(model, p, q, optimizer, BN=True, BN_before_ReLU=False):
    """
    Creates new models which fuses all (imaginary) batch normalization layers.
    """
    fused_model = nn.Sequential()
    layers = list(model.children())
    fused_model.add_module('0', copy_layer(layers[0]))
    i = 1
    
    if not (p == 0 and q == 1):
        i = fuse_imaginary_bn(fused_model, layers, p, q)
    
    if BN:
        if BN_before_ReLU:
            while i < len(layers):
                if 'batch_norm' in str(type(layers[i])):
                    i = fuse_bn_before_activation(fused_model, layers, i)
                if i == (len(layers) - 1):
                    layer = copy_layer(layers[-2])
                    fused_model.add_module(str(len(fused_model)), layer)
                    fused_model.add_module(str(len(fused_model)), copy_layer(layers[-1]))
                i += 1
        else:
            while i < len(layers):
                if (p == 0 and q == 1) and i == 1:
                    layer = copy_layer(layers[1])
                    if isinstance(layers[1], nn.Conv2d):
                        layer.set_bias(layers[1].bias.data if layers[1].bias is not None else torch.zeros(layers[1].out_channels))
                    fused_model.add_module(str(len(fused_model)), layer)
                    fused_model.add_module(str(len(fused_model)), copy_layer(layers[2]))
                if 'batch_norm' in str(type(layers[i])):
                    i = fuse_bn_after_activation(fused_model, layers, i)
                i += 1
    else:
        copy_model(fused_model, model, i)
    
    return fused_model


def fuse_imaginary_bn(fused_model, layers, p, q):
    """
    Fuse an imaginary batch normalization layer due to an input on arbitrary [p, q] range.
    """
    first_layer = layers[1]
    if isinstance(first_layer, nn.Conv2d):
        input_image_shape = first_layer.kernel_size[0]
        input_channels = first_layer.in_channels
        kappa = torch.full((input_channels,), q - p, dtype=torch.float64)
        b_term = torch.full((input_channels,), p, dtype=torch.float64)
        kappa = kappa.repeat(input_image_shape**2)
        b_term = b_term.repeat(input_image_shape**2)
    else:
        input_features = first_layer.in_features
        kappa = torch.full((input_features,), q - p, dtype=torch.float64)
        b_term = torch.full((input_features,), p, dtype=torch.float64)
    
    W = first_layer.weight.data.reshape(-1, first_layer.out_channels if hasattr(first_layer, 'out_channels') else first_layer.out_features)
    kappa = torch.diag(kappa)
    W_fused = torch.matmul(kappa, W)
    
    if isinstance(first_layer, nn.Conv2d):
        W_fused = W_fused.reshape(first_layer.weight.data.shape)
        b_fused = first_layer.bias.data + torch.sum(torch.matmul(torch.diag(b_term), W), dim=0)
    else:
        b_fused = first_layer.bias.data + torch.sum(torch.matmul(torch.diag(b_term), W), dim=0)
    
    layer = copy_layer(first_layer)
    if isinstance(first_layer, nn.Conv2d):
        layer.weight.data = W_fused
        layer.BN.data = torch.tensor([1])
        layer.BN_before_ReLU.data = torch.tensor([0])
        layer.set_bias(bias=b_fused, W=first_layer.weight.data, b_term=b_term)
    else:
        layer.weight.data = W_fused
        layer.bias.data = b_fused
    
    fused_model.add_module(str(len(fused_model)), layer)
    fused_model.add_module(str(len(fused_model)), copy_layer(layers[2]))
    return 3


def fuse_bn_before_activation(fused_model, layers, i):
    """
    Fuses batch normalization layer with previous layer.
    """
    bn = layers[i]
    kappa = bn.weight.data / torch.sqrt(bn.eps + bn.running_var)
    previous_layer = layers[i-1]
    
    if isinstance(previous_layer, nn.Conv2d):
        output_shape = previous_layer.out_channels
    else:
        output_shape = previous_layer.out_features
    
    W = previous_layer.weight.data.reshape(-1, output_shape)
    kappa = torch.diag(kappa)
    W_fused = torch.transpose(torch.matmul(kappa, torch.transpose(W, 0, 1)), 0, 1)
    
    if isinstance(previous_layer, nn.Conv2d):
        W_fused = W_fused.reshape(previous_layer.weight.data.shape)
    
    b_fused = bn.bias.data - bn.running_mean * torch.diag(kappa)
    b_fused += torch.squeeze(torch.matmul(kappa, previous_layer.bias.data.unsqueeze(1)))
    
    layer = copy_layer(previous_layer)
    if isinstance(previous_layer, nn.Conv2d):
        layer.weight.data = W_fused
        layer.BN.data = torch.tensor([1])
        layer.BN_before_ReLU.data = torch.tensor([1])
        layer.set_bias(b_fused)
    else:
        layer.weight.data = W_fused
        layer.bias.data = b_fused
    
    fused_model.add_module(str(len(fused_model)), layer)
    fused_model.add_module(str(len(fused_model)), copy_layer(layers[i+1]))
    
    while 'dropout' in str(type(layers[i+2])) or 'activity_regularization' in str(type(layers[i+2])):
        i += 1
    if 'flatten' in str(type(layers[i+2])) or 'pool' in str(type(layers[i+2])):
        fused_model.add_module(str(len(fused_model)), copy_layer(layers[i+2]))
        i += 1
    return i + 1


def fuse_bn_after_activation(fused_model, layers, i):
    """
    Fuse batch normalization with following layer.
    """
    bn = layers[i]
    kappa = bn.weight.data / torch.sqrt(bn.eps + bn.running_var)
    b_term = bn.bias.data - bn.running_mean * kappa
    
    while 'dropout' in str(type(layers[i+1])):
        i += 1
    
    if 'max_pool' in str(type(layers[i+1])):
        mp = layers[i+1]
        mmp = MaxMinPool2D(kernel_size=mp.kernel_size, stride=mp.stride, padding=mp.padding)
        mmp.sign = torch.sign(kappa).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        fused_model.add_module(str(len(fused_model)), mmp)
        i += 1
        if 'dropout' in str(type(layers[i+1])):
            i += 1
    
    if 'flatten' in str(type(layers[i+1])):
        fused_model.add_module(str(len(fused_model)), copy_layer(layers[i+1]))
        ft = layers[i+1]
        kappa = kappa.repeat(ft.out_features // ft.in_features) if hasattr(ft, 'in_features') else kappa
        b_term = b_term.repeat(ft.out_features // ft.in_features) if hasattr(ft, 'in_features') else b_term
        i += 1
    
    next_layer = layers[i+1]
    if isinstance(next_layer, nn.Conv2d):
        input_image_shape = next_layer.kernel_size[0]
        kappa = kappa.repeat(input_image_shape**2)
        b_term = b_term.repeat(input_image_shape**2)
    
    output_shape = next_layer.out_channels if isinstance(next_layer, nn.Conv2d) else next_layer.out_features
    W = next_layer.weight.data.reshape(-1, output_shape)
    kappa = torch.diag(kappa)
    W_fused = torch.matmul(kappa, W)
    
    if isinstance(next_layer, nn.Conv2d):
        W_fused = W_fused.reshape(next_layer.weight.data.shape)
    
    b_fused = next_layer.bias.data + torch.sum(torch.matmul(torch.diag(b_term), W), dim=0)
    
    layer = copy_layer(next_layer)
    if isinstance(next_layer, nn.Conv2d):
        layer.weight.data = W_fused
        layer.BN.data = torch.tensor([1])
        layer.BN_before_ReLU.data = torch.tensor([0])
        layer.set_bias(bias=b_fused, W=next_layer.weight.data, b_term=b_term)
    else:
        layer.weight.data = W_fused
        layer.bias.data = b_fused
    
    fused_model.add_module(str(len(fused_model)), layer)
    if (i+1) != len(layers)-1:
        fused_model.add_module(str(len(fused_model)), copy_layer(layers[i+2]))
    return i + 2