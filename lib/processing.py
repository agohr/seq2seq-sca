import numpy as np
from scipy.signal import convolve, resample

from tqdm import tqdm

def gaussian_convolution(trace, kernel_width):
  """
  Performs a Gaussian convolution on a 1D trace.

  Args:
    trace: A 1D NumPy array representing the trace data.
    kernel_width: The width of the Gaussian kernel.

  Returns:
    A 1D NumPy array representing the convolved trace.
  """

  # Calculate the standard deviation based on the kernel width
  sigma = kernel_width / 3 

  # Create the Gaussian kernel
  kernel = np.exp(-np.arange(-(kernel_width // 2), kernel_width // 2 + 1) ** 2 / (2 * sigma ** 2))
  kernel /= np.sum(kernel)  # Normalize the kernel

  # Perform the convolution (with 'same' padding to maintain original trace length)
  convolved_trace = convolve(trace, kernel, mode='same')

  return convolved_trace

def resample_recursive(segment, segment_size, max_segment_size=10**7):
    '''
    Resamples the segment to the given segment size using a recursive approach.
    
    This function divides large segments into smaller pieces for resampling when
    the input is too large to process at once, then recombines the results.
    
    Args:
        segment: A 1D NumPy array representing the segment to resample.
        segment_size: The target size for the resampled segment.
        max_segment_size: Maximum size of segment that can be resampled at once.
            Defaults to 10^7.
    
    Returns:
        A 1D NumPy array of length segment_size containing the resampled data.
    '''
    if segment.shape[0] <= max_segment_size:
        return resample(segment, segment_size)
    else:
        n = segment.shape[0]
        n1 = n//2
        n2 = n - n1
        left, right = segment[:n1], segment[-n2:]
        target_size_left = segment_size//2
        target_size_right = segment_size - target_size_left
        left_resampled = resample_recursive(left, target_size_left, max_segment_size)
        right_resampled = resample_recursive(right, target_size_right, max_segment_size)
        return np.concatenate((left_resampled, right_resampled))

def resize_segment(segment, segment_size, method='fourier'):
    '''
    Resizes a segment to the given segment size.
    
    Args:
        segment: A 1D NumPy array to be resized.
        segment_size: The target size for the resized segment.
        method: The resampling method to use. Must be either 'linear' or 'fourier'.
            Defaults to 'fourier'. Normally, the default is strongly expected to be the best choice.
    
    Returns:
        A 1D NumPy array of length segment_size containing the resized data.
    
    Raises:
        ValueError: If segment is not a 1D NumPy array or if method is not 'linear' or 'fourier'.
    '''
    MAX_SEGMENT_SIZE = 10**7
    # raise a ValueError if the segment is not a 1D numpy array
    if not isinstance(segment, np.ndarray):
        raise ValueError('segment must be a numpy array')
    if not len(segment.shape) == 1:
        raise ValueError('segment must be a 1D numpy array')

    # if the segment is already of the right size, do nothing
    if segment.shape[0] == segment_size:
        return segment
    else:
        # resize by interpolation
        if method == 'linear':
            return np.interp(np.linspace(0, segment.shape[0], segment_size), np.arange(segment.shape[0]), segment)
        elif method == 'fourier':
            if segment.shape[0] > MAX_SEGMENT_SIZE:
                return resample_recursive(segment, segment_size, max_segment_size=MAX_SEGMENT_SIZE)
            else:
                # print(f"Resampling segment of size {segment.shape[0]} to {segment_size}")
                return resample(segment, segment_size)
        else:
            raise ValueError('method must be "linear" or "fourier"')

def resize_data(X, n, method='fourier', verbose=True):
    '''
    Resizes multiple segments to the same target size.
    
    Args:
        X: A 2D NumPy array where each row is a segment to be resized.
        n: The target size for each resized segment.
        method: The resampling method to use. Must be either 'linear' or 'fourier'.
            Defaults to 'fourier'. Normally, the default is strongly expected to be the best choice.
        verbose: Whether to display a progress bar. Defaults to True.
    
    Returns:
        A 2D NumPy array where each row is a resized segment of length n.
    '''
    tmp = np.zeros((len(X),n))
    for i in tqdm(range(len(X)), disable=not verbose):
        tmp[i] = resize_segment(X[i], n, method=method)
    return(tmp)