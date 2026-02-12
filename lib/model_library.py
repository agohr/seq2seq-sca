from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation, Reshape, Add, AveragePooling1D, MaxPooling1D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, GlobalMaxPooling1D, Layer, ZeroPadding1D, UpSampling1D, concatenate, Cropping1D, LeakyReLU, Lambda
from keras.regularizers import l2

import tensorflow as tf
import keras

def net_ches_2018(outputs, q1, q2, depth=2, width=None, reg_term=10**-5, output_activation=None, activation='relu', dropout=0.0):
    """
    Creates a neural network model based on the CHES Challenge 2018 architecture.
    
    This model implements a 1D convolutional network with residual connections,
    designed for side-channel analysis tasks. The input is reshaped into a 2D tensor
    before being processed by convolutional layers.
    
    Parameters
    ----------
    outputs : int
        Number of output units in the final layer.
    q1 : int
        First dimension for reshaping the input data.
    q2 : int
        Second dimension for reshaping the input data. The input is expected
        to have n=q1*q2 elements.
    depth : int, optional
        Number of convolutional layers with residual connections, defaults to 2.
    width : int, optional
        Width (number of filters) of the convolutional layers. If None, 
        it is set to the value of `outputs`, defaults to None.
    reg_term : float, optional
        L2 regularization coefficient for convolutional layers, defaults to 10^-5.
    output_activation : str or None, optional
        Activation function for the output layer. If None, no activation 
        function is applied, defaults to None.
    activation : str, optional
        Activation function for the convolutional layers, defaults to 'relu'.
    dropout : float, optional
        Dropout rate applied after convolutions and before the output layer,
        defaults to 0.0.
        
    Returns
    -------
    keras.Model
        The compiled neural network model ready for training.
        
    Notes
    -----
    This architecture was first introduced in the CHES CTF side-channel challenge 2018 and is designed for multi-target extraction on long traces. While it is flexible with respect to number of extracted targets or length of the traces, it requires good alignment of the traces. Compared to the competing UNet approach to multi-target extraction, this architecture is expected to perform better on long, well-aligned traces where the leakage for any given secret is spread out over many leakage sites, and do much worse when alignment is poor but the order of leakage sites within the trace is known.
    """
    if (width is None):
        width = outputs
    p = q1 * q2
    inp = Input(shape=(p,))
    res1 = Reshape((q1, -1))(inp)
    bn = BatchNormalization()(res1)
    conv = Conv1D(width, 1, activation=activation, padding='same',
                  kernel_regularizer=l2(reg_term))(bn)
    shortcut = conv
    for i in range(depth):
        bn = BatchNormalization()(conv)
        conv = Conv1D(width, 3, activation=activation, padding='same',
                      kernel_regularizer=l2(reg_term))(bn)
        conv = Dropout(dropout)(conv)  # Add dropout layer here
        conv = Add()([conv, shortcut])
        shortcut = conv
    if (output_activation is None):
        out = AveragePooling1D(pool_size=q1)(conv)
        out = Flatten()(out)
        out = Dropout(dropout)(out)  # Add dropout layer here
        out = Dense(outputs)(out)
    else:
        out = AveragePooling1D(pool_size=q1)(conv)
        out = Flatten()(out)
        out = Dropout(dropout)(out)
        out = Dense(outputs, activation=output_activation)(out)
    model = Model(inputs=inp, outputs=out)
    return(model)

def VGG_1D(input_length, num_classes, conv_filters, dense_sizes, kernel_size = 3, l2_reg=1e-5, activation='relu', output_activation="default", dense_batch_norm=False):
    """
    Constructs a 1D VGG-like model.
    
    Parameters:
    - input_length (int): The length of the 1D input data.
    - num_classes (int): The number of output classes.
    - conv_filters (list): A list containing the number of filters for each Conv1D block.
                           Each element corresponds to a block, with the first two numbers being for the first block, 
                           the next two for the second block, and so on.
    - dense_sizes (list): A list containing the sizes of the dense layers.
    - kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
    - l2_reg (float, optional): The L2 regularization factor. Defaults to 1e-5.
    - activation (str, optional): The activation function to use for the convolutional and dense layers. Defaults to 'relu'.
    - output_activation (str, optional): The activation function to use for the output layer. 
                                         If 'default', 'softmax' is used for multi-class classification, 
                                         and 'sigmoid' is used for binary classification. Defaults to 'default'.
    
    Returns:
    - A Keras model instance.
    """
    # Input shape is a single dimension for 1D data.
    input_layer = Input(shape=(input_length,))
    # Reshape input to have a "channel" dimension.
    x = Reshape((input_length, 1))(input_layer)
    
    # Add convolutional blocks based on conv_filters.
    for filters in conv_filters:
        # Apply Conv1D with 'relu' activation and same padding.
        x = Conv1D(filters, kernel_size, activation=activation, padding='same', kernel_regularizer=l2(l2_reg))(x)
        # Apply BatchNormalization.
        x = BatchNormalization()(x)
        x = MaxPooling1D(2, strides=2)(x)
    
    # Flatten the output to feed into the dense layers.
    x = Flatten()(x)
    
    # Add dense layers based on dense_sizes.
    for size in dense_sizes:
        x = Dense(size, activation=activation)(x)
        if dense_batch_norm:
            x = BatchNormalization()(x)
    
    # Output layer with 'softmax' activation for classification.
    if num_classes > 2:
        if output_activation == "default":
            output_activation = 'softmax'
        output_layer = Dense(num_classes, activation=output_activation)(x)
    else:
        if output_activation == "default":
            output_layer = Dense(1, activation='sigmoid')(x)
        else:
            output_layer = Dense(num_classes, activation=output_activation)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def ResBlock(x, filters, kernel_size=3, activation='relu', dropout=0.0, l2_reg=1e-5):
    """
    Creates a Residual Block for 1D data.
    
    This function implements a standard residual block with two convolutional layers
    and a skip connection. The block includes batch normalization, activation, and
    optional dropout for regularization.
    
    Parameters
    ----------
    x : Tensor
        Input tensor to the residual block.
    filters : int
        Number of filters for the convolutional layers.
    kernel_size : int, optional
        Size of the convolutional kernel, defaults to 3.
    activation : str, optional
        Activation function to use after convolutions, defaults to 'relu'.
    dropout : float, optional
        Dropout rate between 0 and 1 applied after the first activation, defaults to 0.0.
    l2_reg : float, optional
        L2 regularization coefficient for convolutional layers, defaults to 1e-5.
        
    Returns
    -------
    Tensor
        Output tensor after applying the residual block.
        
    Notes
    -----
    The input tensor and output tensor have the same shape, which is necessary
    for the residual connection to work properly.
    """
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Skip connection
    x = Activation(activation)(x)
    return x

def ResNet_1D(input_length, num_classes, depth, res_filters, dense_sizes, kernel_size=3, final_filters=1, l2_reg=1e-5, output_activation="default", activation='relu', strides=None):
    """
    Constructs a 1D ResNet-like model.
    
    Parameters:
    - input_length: The length of the 1D input data.
    - num_classes: The number of output classes.
    - depth: The number of ResBlocks to add.
    - res_filters: Number of convolutional filters in each ResBlock.
    - dense_sizes: A list containing the sizes of the dense layers.
    - final_filters: Number of filters in the final convolutional layer. Default is 1.
    - l2_reg: L2 regularization term. Default is 1e-5.
    - dropout: Dropout rate. Default is 0.0.
    - output_activation: Activation function for the output layer. Default is "default".
    - activation: Activation function for the convolutional and dense layers. Default is 'relu'.
    
    Returns:
    - A Keras model instance.
    """
    # if res_filters is a single number, use the same number for all ResBlocks
    if not isinstance(res_filters, list):
        res_filters = [res_filters] * depth
    # Input shape is a single dimension for 1D data.
    input_layer = Input(shape=(input_length,))
    # Reshape input to have a "channel" dimension.
    x = Reshape((input_length, 1))(input_layer)

    # initial convolutional layer
    x = Conv1D(res_filters[0], 1, padding='same', activation=activation, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # Add convolutional blocks based on res_filters.
    for i in range(depth):
        # Add a Residual Block
        if i > 0 and res_filters[i] != res_filters[i-1]:
            x = Conv1D(res_filters[i], 1, padding='same', activation=activation, kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
        x = ResBlock(x, res_filters[i], kernel_size=kernel_size)
        if strides is not None:
            x = MaxPooling1D(2, strides=strides)(x)

    # final convolutional layer
    
    x = BatchNormalization()(x)
    x = Conv1D(final_filters, 1, padding='same', activation=activation, kernel_regularizer=l2(l2_reg))(x)
    
    # Flatten the output to feed into the dense layers.
    x = Flatten()(x)
    
    # Add dense layers based on dense_sizes.
    for size in dense_sizes:
        x = BatchNormalization()(x)
        x = Dense(size, activation=activation, kernel_regularizer=l2(l2_reg))(x)
    
    # Output layer with 'softmax' activation for classification.
    if num_classes > 2:
        if output_activation == "default":
            output_activation = 'softmax'
        output_layer = Dense(num_classes, activation=output_activation)(x)
    else:
        if output_activation == "default":
            output_layer = Dense(1, activation='sigmoid')(x)
        else:
            output_layer = Dense(num_classes, activation=output_activation)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def unet_1d(input_size, n_filters=64, depth=4, output_size=None, output_channels=1, dropout_rate=0.3, activation='relu', output_activation='default'):
    """
    Creates a 1D U-Net model for multi-target extraction or segmentation tasks.
    
    This function builds a U-Net architecture with configurable depth and filter sizes,
    designed for processing 1D signals. The model follows an encoder-decoder structure
    with skip connections, making it suitable for tasks where preserving spatial
    information is important, such as multi-target extraction from side-channel traces.
    
    Parameters
    ----------
    input_size : int
        The size (length) of the input vector.
    n_filters : int, optional
        The number of filters in the first layer, which doubles with each depth level,
        defaults to 64.
    depth : int, optional
        The depth of the U-Net (number of downsampling/upsampling operations),
        defaults to 4.
    output_size : int or None, optional
        The size of the output vector. If None, it is set to the input_size,
        defaults to None.
    output_channels : int, optional
        The number of output channels (features per position), defaults to 1.
    dropout_rate : float, optional
        The dropout rate between 0 and 1 applied in convolutional blocks,
        defaults to 0.3.
    activation : str, optional
        The activation function for the convolutional layers, defaults to 'relu'.
    output_activation : str, optional
        The activation function for the output layer. If 'default', uses 'sigmoid' 
        for single-channel output and 'softmax' for multi-channel output.
        Defaults to 'default'.
        
    Returns
    -------
    keras.Model
        The compiled U-Net model ready for training.
        
    Notes
    -----
    This model is particularly effective for tasks where there is a large number of targets to extract and where each target is associated to a relatively well-localized region of the input trace (but that region can be large and can overlap with other targets).

    The depth parameter should be set such that the receptive field of nodes in the
    middle of the U-Net spans a significant portion of the input. This architecture
    is expected to perform best with relatively small input vectors (at least in training - once a target has been learned, the same network can be used for any input size), although this expectation may change with future research.
    
    Compared to the net_ches_2018 model, this architecture is more robust to alignment
    issues in the input data but may require more parameters for equivalent performance
    on well-aligned data.
    """
    if output_size is None:
        output_size = input_size

    def conv_block(inputs, n_filters, kernel_size=3, use_dropout=True):
        x = Conv1D(n_filters, kernel_size, padding='same', activation=activation)(inputs)
        x = BatchNormalization()(x)
        if use_dropout:
            x = Dropout(dropout_rate)(x)
        x = Conv1D(n_filters, kernel_size, padding='same', activation=activation)(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(shape=(input_size, 1))
    x = inputs

    # Encoder (downsampling)
    skip_connections = []
    for i in range(depth):
        x = conv_block(x, n_filters * (2**i))
        skip_connections.append(x)
        x = MaxPooling1D(pool_size=2)(x)

    # Bridge
    x = conv_block(x, n_filters * (2**depth))

    # Decoder (upsampling)
    for i in reversed(range(depth)):
        x = UpSampling1D(size=2)(x)
        # Stop upsampling if size meets or exceeds output_size
        if x.shape[1] >= output_size:
            break
        # Match size with skip connection
        skip = skip_connections[i]
        if x.shape[1] != skip.shape[1]:
            diff = skip.shape[1] - x.shape[1]
            if diff > 0:  # Need to pad
                x = tf.keras.layers.ZeroPadding1D(padding=(diff // 2, diff - diff // 2))(x)
            elif diff < 0:  # Need to crop
                x = x[:, -diff // 2:x.shape[1] + (diff - diff // 2)]
        x = concatenate([x, skip])
        x = conv_block(x, n_filters * (2**i), use_dropout=False)

    # Output layer
    x = Conv1D(n_filters, kernel_size=1, padding='same', activation=activation)(x)
    x = BatchNormalization()(x)

    # Final convolution layer with output_channels
    if output_activation == 'default':
        if output_channels == 1:
            output_activation = 'sigmoid'
        else:
            output_activation = 'softmax'
    
    x = Conv1D(output_channels, kernel_size=1, activation=output_activation, padding='same')(x)

    # Custom function to perform 1D resizing using interpolation
    def resize_1d(inputs, output_size):
        inputs_expanded = tf.expand_dims(inputs, axis=2)  # Make it (batch, time, channels)
        resized = tf.image.resize(inputs_expanded, size=(output_size, 1), method='bilinear')
        return tf.squeeze(resized, axis=2)  # Return to shape (batch, output_size, channels)

    # Resize using the custom resizing function
    if output_size:
        if output_size != x.shape[1]:
            x = Lambda(lambda tensor: resize_1d(tensor, output_size))(x)

    # The output shape is now (batch_size, output_size, output_channels)
    outputs = x

    model = Model(inputs=inputs, outputs=outputs)
    return model
