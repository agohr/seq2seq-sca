import numpy as np

from sklearn.metrics import confusion_matrix

from tqdm import tqdm

import tensorflow as tf

import tensorflow.keras as keras

import json

import tensorflow as tf
from keras.layers import Lambda, Input
from keras import backend as K
import model_library as ml
import learning_rates as lr

from os import remove

def synthetic_data_alignment(num_samples=1000, num_features=10, seq_length=20, noise=0.0, p_other=0.1):
    """
    Generate synthetic data for an alignment experiment.
    
    Creates data with binary features (0 or 1) interspersed with irrelevant data (2).
    The goal is to train models that can extract the first `num_features` relevant values
    (0s and 1s) from each sequence, ignoring the irrelevant values.
    
    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate, by default 1000
    num_features : int, optional
        Number of relevant binary features to include in each sample, by default 10
    seq_length : int, optional
        Length of each input sequence, by default 20
    noise : float, optional
        Probability of flipping a bit in the output features (0->1 or 1->0), by default 0.0
    p_other : float, optional
        Probability of replacing a position with irrelevant data (value 2), by default 0.1
    
    Returns
    -------
    tuple of np.ndarray
        X : Input sequences with shape (num_samples, seq_length)
            Contains values 0, 1, and 2 (where 2 represents irrelevant data)
        Y : Target features with shape (num_samples, num_features)
            Contains the first num_features binary values (0 or 1) from each sequence
    
    Notes
    -----
    The function first generates random binary data, then replaces some values with 2s
    according to p_other. It then extracts the first num_features non-2 values from
    each sequence to create the target Y. If noise is specified, some bits in Y are flipped.
    """
    X = np.random.randint(0, 2, size=(num_samples, seq_length))
    
    # Intersperse/pad with other data
    r = np.random.rand(num_samples, seq_length)
    X[r < p_other] = 2
    
    Y = np.zeros((num_samples, num_features), dtype=int)
    pos = np.zeros(num_samples, dtype=int)
    
    for i in range(seq_length):
        feature = (X[:, i] != 2)
        mask = (pos < num_features) & feature
        Y[mask, pos[mask]] = X[mask, i]
        pos += feature
    
    # Add noise if specified
    if noise > 0:
        noise_mask = np.random.rand(*Y.shape) < noise
        Y[noise_mask] = 1 - Y[noise_mask]
    
    return X, Y

# A version of the above that simply places num_features 0/1 values into random positions and fills the rest with 2
def synthetic_data_alignment_clean(num_samples=1000, num_features=10, seq_length=20, noise=0.0):
    """
    Generate clean synthetic data for alignment experiments with controlled positioning.
    
    Creates sequences where exactly `num_features` positions contain binary values (0 or 1),
    and all other positions contain irrelevant data (2). The binary values are placed at
    random but ordered positions within each sequence.
    
    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate, by default 1000
    num_features : int, optional
        Number of binary features to include in each sample, by default 10
    seq_length : int, optional
        Length of each input sequence, by default 20
    noise : float, optional
        Probability of flipping a bit in the output features (0->1 or 1->0), by default 0.0
        Note: This parameter is currently not used in the implementation.
    
    Returns
    -------
    tuple of np.ndarray
        X : Input sequences with shape (num_samples, seq_length)
            Contains values 0, 1, and 2 (where 2 represents irrelevant data)
        Y : Target features with shape (num_samples, num_features)
            Contains the binary values (0 or 1) in the order they appear in X
    
    Notes
    -----
    Unlike `synthetic_data_alignment`, this function guarantees exactly `num_features`
    binary values in each sequence, placed at random but sorted positions. This creates
    a cleaner dataset where the alignment challenge is more controlled.
    """
    X = np.full((num_samples, seq_length), 2)
    Y = np.zeros((num_samples, num_features), dtype=int)
    
    for i in range(num_samples):
        # Generate sorted positions to maintain left-to-right ordering
        positions = np.sort(np.random.choice(seq_length, num_features, replace=False))
        Y[i, :] = np.random.randint(0, 2, num_features)
        X[i, positions] = Y[i]
    return X, Y

def train_unet_ctc(X, Y, seq_length=512, num_features=32, depth=4, n_filters=64, 
                   epochs=101, batch_size=32, validation_split=0.1, 
                   learning_rate=0.001, min_learning_rate=1e-5, 
                   learning_rate_cycle=20, output_channels=3, unet=None, check=None):
    """
    Trains a U-Net model with CTC loss for tasks where we want to directly predict a sequence from side-channel datawithout knowing alignment.
    
    Parameters:
    X : np.ndarray
        Input data with shape (num_samples, seq_length) or (num_samples, seq_length, 1).
    Y : np.ndarray
        Target labels with shape (num_samples, num_features).
    seq_length : int, optional
        Length of the input sequences. Default is 512.
    num_features : int, optional
        Number of features to extract from each sequence. Default is 32. This is essentially an assumption on the training data. We assume that the training data always has a fixed number of labels.
    depth : int, optional
        Depth of the U-Net architecture. Default is 4.
    n_filters : int, optional
        Number of filters in the first layer of the U-Net. Default is 64.
    epochs : int, optional
        Number of training epochs. Default is 101.
    batch_size : int, optional
        Batch size for training. Default is 32.
    validation_split : float, optional
        Fraction of the data to use for validation. Default is 0.1.
    learning_rate : float, optional
        Maximum learning rate for cyclic learning rate schedule. Default is 0.001.
    min_learning_rate : float, optional
        Minimum learning rate for cyclic learning rate schedule. Default is 1e-5.
    learning_rate_cycle : int, optional
        Number of epochs in one learning rate cycle. Default is 20.
    output_channels : int, optional
        Number of output channels for the U-Net. Default is 3.
    unet : keras.Model, optional
        Pre-built U-Net model. If None, a new model will be created. Default is None.
    check : keras.callbacks.ModelCheckpoint, optional
        ModelCheckpoint callback for saving model weights. Default is None.
        
    Returns:
    tuple
        A tuple containing:
        - unet (keras.Model): The trained U-Net model
        - history (keras.callbacks.History): Training history object
        
    This function implements a U-Net architecture with Connectionist Temporal Classification (CTC)
    loss for sequence alignment tasks. It uses a cyclic learning rate schedule to improve
    convergence and can optionally save model checkpoints during training.
    """
    # Create base U-Net model
    if unet is None:
        unet = ml.unet_1d(seq_length, n_filters=n_filters, depth=depth,
                      output_size=seq_length, output_channels=output_channels,
                      output_activation='softmax')
    
    # Create input tensors for CTC
    main_input = Input(shape=(seq_length, 1), name='main_input')
    labels = Input(name='labels', shape=(num_features,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,))
    label_length = Input(name='label_length', shape=(1,))
    
    # Get U-Net output for our custom input
    unet_out = unet(main_input)
    
    # Define CTC loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    # Add CTC loss layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [unet_out, labels, input_length, label_length]
    )
    
    # Create training model
    model = tf.keras.Model(
        inputs=[main_input, labels, input_length, label_length],
        outputs=loss_out
    )
    
    # Prepare cyclic learning rate
    lr_cyclic = lr.lr_cyclic_sawtooth(
        learning_rate_cycle,
        lr_max=learning_rate,
        lr_min=min_learning_rate,
        shift=learning_rate_cycle//2
    )
    
    # Compile model
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    
    # Prepare input data for CTC
    input_length_data = np.ones((X.shape[0], 1)) * seq_length
    label_length_data = np.ones((Y.shape[0], 1)) * Y.shape[1]
    
    # Reshape X if needed
    if len(X.shape) == 2:
        X = X[..., np.newaxis]
    
    # Create inputs dictionary
    train_inputs = {
        'main_input': X,
        'labels': Y,
        'input_length': input_length_data,
        'label_length': label_length_data
    }
    
    # Create dummy outputs for CTC
    train_outputs = {'ctc': np.zeros(X.shape[0])}
    
    if check is not None:
        callbacks = [check, lr_cyclic]
    else:
        callbacks = [lr_cyclic]

    # Train the model
    history = model.fit(
        train_inputs,
        train_outputs,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )

    if check is not None:
        unet_filepath = check.filepath.replace('.keras', '_unet.keras')
        model.load_weights(check.filepath)
        unet.save(unet_filepath)
        print(f"Saved unet to {unet_filepath}")
        # delete the checkpoint file
        remove(check.filepath)

    return unet, history

    
