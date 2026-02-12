from keras.callbacks import LearningRateScheduler

def lr_cyclic_sawtooth(epochs, lr_max=0.001, lr_min=0.00005, shift=0):
    """
    Creates a learning rate scheduler with a cyclic sawtooth pattern.
    
    This function generates a learning rate scheduler that follows a sawtooth pattern,
    where the learning rate increases and decreases symmetrically within each cycle.
    
    Parameters:
    -----------
    epochs : int
        The number of epochs in one complete cycle.
    lr_max : float, optional
        The maximum learning rate in the cycle (default: 0.001).
    lr_min : float, optional
        The minimum learning rate in the cycle (default: 0.00005).
    shift : int, optional
        Number of epochs to shift the cycle (default: 0).
        
    Returns:
    --------
    LearningRateScheduler
        A Keras callback that can be passed to model.fit().
    """
    def lr_schedule(epoch):
        epoch = (epoch + shift) % epochs
        abs_epoch = min(epoch, epochs - epoch)
        return lr_max - (lr_max - lr_min) * (2 * abs_epoch / epochs)
    lr = LearningRateScheduler(lr_schedule)
    return lr

def lr_cyclic(epochs, lr_max=0.001, lr_min=0.00005):
    """
    Creates a learning rate scheduler with a simple linear cyclic pattern.
    
    This function generates a learning rate scheduler that linearly decreases
    from lr_max to lr_min over each cycle of epochs.
    
    Parameters:
    -----------
    epochs : int
        The number of epochs in one complete cycle. Must be greater than 1.
    lr_max : float, optional
        The maximum learning rate in the cycle (default: 0.001).
    lr_min : float, optional
        The minimum learning rate in the cycle (default: 0.00005).
        
    Returns:
    --------
    LearningRateScheduler
        A Keras callback that can be passed to model.fit().
        
    Raises:
    -------
    AssertionError
        If epochs is not greater than 1.
    """
    assert epochs > 1
    def lr_schedule(epoch):
        epoch = epoch % epochs
        return lr_max - (lr_max - lr_min) * (epoch / (epochs - 1))
    lr = LearningRateScheduler(lr_schedule)
    return lr