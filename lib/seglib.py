import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import optuna

from processing import resize_segment

import json
from datetime import datetime

def non_max_suppression(data, seg_points, window_size=10, iou_threshold=0.5, mode='absolute'):
    """
    Performs non-maximum suppression on a list of segment points.
    
    This function filters segment points by removing those that are too close to
    segment points with higher signal strength, based on an IoU (Intersection over Union) threshold for the neighborhood to be considered.

    Points are prioritized by their signal strength according to the specified mode. The rationale behind this is that picking survivors by strongest signal is likely to lead to good alignment of segment boundaries across traces. Hence, the function assumes that the underlying segmentation mechanism proposes good segment boundaries, and uses non-maximum suppression to enforce good alignment.
    
    Parameters:
        data (numpy.ndarray): 1D array of the curve data.
        seg_points (list): List of segment boundary indices.
        window_size (int, optional): Size of the window around each point for IoU calculation. Defaults to 10.
        iou_threshold (float, optional): Threshold for considering points as overlapping. Defaults to 0.5.
        mode (str, optional): Method to determine signal strength. Options:
            - 'absolute': Use absolute value of data at segment point.
            - 'positive': Use raw value of data at segment point.
            - 'negative': Use negative value of data at segment point.
            Defaults to 'absolute'.
    
    Returns:
        list: Filtered list of segment points after non-maximum suppression, sorted in ascending order.
    
    Raises:
        ValueError: If mode is not one of 'absolute', 'positive', or 'negative'.
    """
    # we have to set a function that determines the signal strength of a segment point
    if mode == 'absolute':
        signal_strength = lambda x: abs(data[x])
    elif mode == 'positive':
        signal_strength = lambda x: data[x]
    elif mode == 'negative':
        signal_strength = lambda x: -data[x]
    else:
        raise ValueError("Invalid mode. Must be one of 'absolute', 'positive', 'negative'.")

    # first, sort the segment points by absolute signal strength
    seg_points.sort(key=signal_strength, reverse=True)
    # then, iterate through the sorted segmentation boundaries, compute bounding boxes around them, and if two bounding boxes overlap with more than accepted iou, remove the one with lower signal strength
    def iou(a, b):
        a_start, a_end = max(0, a - window_size), min(len(data), a + window_size)
        b_start, b_end = max(0, b - window_size), min(len(data), b + window_size)
        intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        return intersection / union
    new_seg_points = []
    for seg_point in seg_points:
        if len(new_seg_points) == 0:
            new_seg_points.append(seg_point)
            continue
        # check if the current segment point overlaps with any of the existing segment points
        overlap = False
        for new_seg_point in new_seg_points:
            if iou(seg_point, new_seg_point) > iou_threshold:
                overlap = True
                break
        if not overlap:
            new_seg_points.append(seg_point)
    new_seg_points.sort()
    return new_seg_points

def max_align(data, seg_points, window_size=10, mode='absolute'):
    """
    Aligns segment points to local maxima of signal strength within a window.
    
    This function adjusts each segment point to the position of maximum signal
    strength within a specified window around the original point. Signal strength
    is determined according to the specified mode.
    
    Parameters:
        data (numpy.ndarray): 1D array of the curve data.
        seg_points (list): List of segment boundary indices.
        window_size (int, optional): Size of the window around each point for finding local maximum. Defaults to 10.
        mode (str, optional): Method to determine signal strength. Options:
            - 'absolute': Use absolute value of data.
            - 'positive': Use raw value of data.
            - 'negative': Use negative value of data.
            Defaults to 'absolute'.
    
    Returns:
        list: List of aligned segment points, sorted in ascending order.
    
    Raises:
        ValueError: If mode is not one of 'absolute', 'positive', or 'negative'.
    """
    # we have to set a function that determines the signal strength of a segment point
    if mode == 'absolute':
        signal_strength = lambda x: abs(data[x])
    elif mode == 'positive':
        signal_strength = lambda x: data[x]
    elif mode == 'negative':
        signal_strength = lambda x: -data[x]
    else:
        raise ValueError("Invalid mode. Must be one of 'absolute', 'positive', 'negative'.")
    
    new_seg_points = []
    for seg_point in seg_points:
        start = max(0, seg_point - window_size)
        end = min(len(data), seg_point + window_size)
        new_seg_point = start + np.argmax(data[start:end])
        new_seg_points.append(new_seg_point)
    new_seg_points.sort()
    return new_seg_points

def segment_curve_fourier_global(data, downsample_factor=10, error_threshold=2.0):
    """
    Segments a curve based on prediction errors using Fourier downsampling and upsampling.
    
    This function identifies segment boundaries by:
    1. Downsampling the signal using Fourier methods
    2. Upsampling it back to the original size
    3. Calculating the absolute error between original and reconstructed signals
    4. Finding peaks in the error that exceed a threshold
    
    Parameters:
        data (numpy.ndarray): 1D array of the curve data.
        downsample_factor (int, optional): Factor by which to downsample the signal. Defaults to 10.
        error_threshold (float, optional): Threshold for peak detection in error,
            expressed as a multiple of the error's standard deviation. Defaults to 2.0.
    
    Returns:
        list: List of indices where segments start, with 0 as the first element.
    """
    # Downsample the entire signal
    downsampled_size = len(data) // downsample_factor
    downsampled = resize_segment(data, downsampled_size, method='fourier')
    
    # Upsample back to original size
    upsampled = resize_segment(downsampled, len(data), method='fourier')
    
    # Calculate error
    errors = np.abs(data - upsampled)
    
    # Find peaks in the error
    peaks, _ = find_peaks(errors, height=error_threshold * np.std(errors))
    
    # These peaks directly correspond to segment boundaries
    segment_boundaries = peaks
    
    return [0] + list(segment_boundaries)

def segment_trace_fourier(trace, num_segments=2, k=500, downsample_factor=10, error_threshold=2.0, buffer=50):
    """
    Segments a trace using Fourier-based method and returns resampled segments.
    
    This function:
    1. Segments the trace using Fourier-based error detection
    2. Selects the largest segments (by length)
    3. Adds buffer zones around each segment
    4. Resamples each segment to a fixed length using Fourier methods
    
    Parameters:
        trace (numpy.ndarray): 1D array of the original trace.
        num_segments (int, optional): Number of top segments to return. If 0, returns all segments. Defaults to 2.
        k (int, optional): Number of points to resample each segment to. Defaults to 500.
        downsample_factor (int, optional): Factor for downsampling in segmentation. Defaults to 10.
        error_threshold (float, optional): Threshold for peak detection in segmentation. Defaults to 2.0.
        buffer (int, optional): Number of samples to include before and after each segment. Defaults to 50.
    
    Returns:
        list: List of numpy arrays, each representing a resampled segment.
    """
    # Segment the curve
    boundaries = segment_curve_fourier_global(trace, downsample_factor, error_threshold)
    
    # Create segments
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        segments.append((start, end, end - start, i))
    
    # Sort segments by size (largest first) and get the top num_segments
    segments.sort(key=lambda x: x[2], reverse=True)
    
    if num_segments > 0:
        top_segments = segments[:num_segments]
    else:
        top_segments = segments
    
    # Sort the top segments by their original order in the trace
    top_segments.sort(key=lambda x: x[3])
    
    # Resample each segment to k points using Fourier method
    resampled_segments = []
    for start, end, _, _ in top_segments:
        a,b = start - buffer, end + buffer
        a = max(0, a)
        b = min(len(trace), b)
        segment = trace[a:b]
        resampled = resize_segment(segment, k, method='fourier')
        resampled_segments.append(resampled)
    
    return resampled_segments


def get_segment(trace, start, end, target_length, padding='zero'):
    """
    Extract a segment from a trace and resize it to a target length.
    
    This function handles cases where the requested segment extends beyond the trace boundaries
    by applying the specified padding method. The extracted segment is then resized to the
    target length using the resize_segment function.
    
    Parameters:
        trace (numpy.ndarray): 1D array of the original trace data.
        start (int): Starting index of the segment to extract.
        end (int): Ending index of the segment to extract.
        target_length (int): Desired length of the output segment.
        padding (str, optional): Method to use for padding when segment extends beyond trace boundaries.
            Options:
            - 'zero': Pad with zeros.
            - 'mirror': Pad with mirrored values from the trace.
            - 'wrap': Pad by wrapping around to the other end of the trace.
            - 'none': No padding (may result in shorter segments).
            Defaults to 'zero'.
    
    Returns:
        numpy.ndarray: Extracted and resized segment.
    
    Raises:
        AssertionError: If padding method is not one of the supported options.
    """
    assert padding in ['zero', 'mirror', 'wrap', 'none'], "Invalid padding method"
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0
    if end > len(trace):
        pad_right = end - len(trace)
        end = len(trace)
    else:
        pad_right = 0
    a = max(0, start)
    b = min(len(trace), end)
    segment = trace[a:b]
    if padding == 'zero':
        segment = np.pad(segment, (pad_left, pad_right), 'constant', constant_values=0)
    elif padding == 'mirror':
        segment = np.pad(segment, (pad_left, pad_right), 'reflect')
    elif padding == 'wrap':
        segment = np.pad(segment, (pad_left, pad_right), 'wrap')
    segment = resize_segment(segment, target_length)
    return segment

def create_segments(trace, seg_points, target_length, buffer=0, padding='zero'):
    """
    Create multiple segments from a trace based on segment boundary points.
    
    This function extracts segments between consecutive segment points, optionally
    extending each segment by a buffer zone on both sides. Each segment is then
    resized to a target length. The buffer zone helps ensure that important features
    near segment boundaries are not missed due to slight misalignments.
    
    Parameters:
        trace (numpy.ndarray): 1D array of the original trace data.
        seg_points (list): List of indices marking segment boundaries.
        target_length (int): Desired length of each output segment.
        buffer (int, optional): Number of samples to include before and after each segment.
            Defaults to 0.
        padding (str, optional): Method to use for padding when segments extend beyond trace boundaries.
            Options: 'zero', 'mirror', 'wrap', 'none'. Defaults to 'zero'.
    
    Returns:
        numpy.ndarray: 2D array where each row is a resized segment.
    
    Raises:
        AssertionError: If padding method is not one of the supported options.
    """
    assert padding in ['zero', 'mirror', 'wrap', 'none'], "Invalid padding method"
    segments = []
    for i in range(len(seg_points) - 1):
        start, end = seg_points[i], seg_points[i+1]
        a,b = start - buffer, end + buffer
        seg = get_segment(trace, a, b, target_length, padding)
        segments.append(seg)
    segments = np.array(segments)
    return segments

def plot_segpoints(trace, seg_points, a=None, b=None):
    """
    Visualize a trace with segment boundary points marked as vertical lines.
    
    This function creates a plot showing a portion of the trace with vertical lines
    indicating the positions of segment boundaries. This is useful for visually
    inspecting the quality of segmentation.
    
    Parameters:
        trace (numpy.ndarray): 1D array of the trace data to plot.
        seg_points (list): List of indices marking segment boundaries.
        a (int, optional): Starting index of the trace portion to plot. If None,
            starts from the beginning of the trace. Defaults to None.
        b (int, optional): Ending index of the trace portion to plot. If None,
            ends at the end of the trace. Defaults to None.
    
    Returns:
        None: This function displays a plot but does not return any value.
    """
    if a is None:
        a = 0
    if b is None:
        b = len(trace)
    plt.plot(np.arange(a,b), trace[a:b])
    seg_points_filtered = [x for x in seg_points if a <= x < b]
    for seg_point in seg_points_filtered:
        plt.axvline(seg_point, color='r', linestyle='--')
    plt.show()

def insert_segpoint(seg_points, point):
    """
    Insert a new segment boundary point into a sorted list of segment points.
    
    This function maintains the sorted order of segment points when inserting a new point.
    The point is inserted at the appropriate position to preserve ascending order.
    
    Parameters:
        seg_points (list): Sorted list of segment boundary indices.
        point (int): New segment boundary index to insert.
    
    Returns:
        None: The seg_points list is modified in-place.
    """
    for i in range(len(seg_points)):
        if seg_points[i] > point:
            seg_points.insert(i, point)
            return
    seg_points.append(point)

def remove_segpoint(seg_points, point):
    """
    Remove a segment boundary point from a list of segment points.
    
    This function removes the segment point at the specified index in the list.
    
    Parameters:
        seg_points (list): List of segment boundary indices.
        point (int): Index in the seg_points list of the boundary to remove.
            This is not the boundary value itself, but its position in the list.
    
    Returns:
        None: The seg_points list is modified in-place.
    """
    del seg_points[point]


def show_longest_segments(trace, segpoints, num_segments=5, context_segments=2):
    """
    Visualize the longest segments in a trace with surrounding context.
    
    This function identifies the longest segments in a trace based on the distance
    between segment points, then creates a multi-panel plot showing each segment
    with its surrounding context. The main segment is highlighted with a yellow
    background, and all segment boundaries are marked with vertical red lines.
    
    Parameters:
        trace (numpy.ndarray): 1D array of the trace data to visualize.
        segpoints (list or numpy.ndarray): List of indices marking segment boundaries.
        num_segments (int, optional): Number of longest segments to display.
            Defaults to 5.
        context_segments (int, optional): Number of segments to show on each side
            of the target segment for context. Defaults to 2.
    
    Returns:
        None: This function displays a plot but does not return any value.
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000)
        >>> segpoints = [0, 100, 150, 400, 450, 700, 1000]
        >>> show_longest_segments(trace, segpoints, num_segments=3, context_segments=1)
    """
    d = np.diff(segpoints)
    longest_segments = np.argsort(d)[-num_segments:]
    
    fig, axs = plt.subplots(num_segments, 1, figsize=(12, 3*num_segments), sharex=False)
    fig.suptitle("Longest Segments in Trace", fontsize=16, y=0.98)  # Moved higher up
    
    if num_segments == 1:
        axs = [axs]
    
    for i, seg in enumerate(longest_segments):
        start_idx = max(0, seg - context_segments)
        end_idx = min(len(segpoints) - 1, seg + context_segments + 1)
        
        start = segpoints[start_idx]
        end = segpoints[end_idx]
        
        segment_data = trace[start:end]
        x_range = range(start, end)
        
        axs[i].plot(x_range, segment_data)
        
        for j in range(start_idx, end_idx):
            seg_start = segpoints[j]
            axs[i].axvline(x=seg_start, color='r', linestyle='--', alpha=0.5)
            axs[i].text(seg_start, axs[i].get_ylim()[1], f'{j}', 
                        rotation=45, va='bottom', ha='right', fontsize=8)
        
        main_segment_start = segpoints[seg]
        main_segment_end = segpoints[seg+1]
        axs[i].axvspan(main_segment_start, main_segment_end, alpha=0.2, color='yellow')
        
        # Changed from set_title to text annotation to avoid overlap with suptitle
        axs[i].text(0.5, 0.98, f"Segment {seg} (Length: {d[seg]})", 
                   transform=axs[i].transAxes, ha='center', va='top', fontsize=10)
        
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlim(start, end)
        
        # Add padding to y-axis
        y_min, y_max = np.min(segment_data), np.max(segment_data)
        y_range = y_max - y_min
        axs[i].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
    axs[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Increased top margin to make room for suptitle
    plt.show()

def show_shortest_segments(trace, segpoints, num_segments=5, context_segments=2):
    """
    Visualize the shortest segments in a trace with surrounding context.
    
    This function identifies the shortest segments in a trace based on the distance
    between segment points, then creates a multi-panel plot showing each segment
    with its surrounding context. The main segment is highlighted with a yellow
    background, and all segment boundaries are marked with vertical red lines.
    
    Parameters:
        trace (numpy.ndarray): 1D array of the trace data to visualize.
        segpoints (list or numpy.ndarray): List of indices marking segment boundaries.
        num_segments (int, optional): Number of shortest segments to display.
            Defaults to 5.
        context_segments (int, optional): Number of segments to show on each side
            of the target segment for context. Defaults to 2.
    
    Returns:
        None: This function displays a plot but does not return any value.
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000)
        >>> segpoints = [0, 100, 150, 400, 450, 700, 1000]
        >>> show_shortest_segments(trace, segpoints, num_segments=3, context_segments=1)
    """
    d = np.diff(segpoints)
    shortest_segments = np.argsort(d)[:num_segments]
    
    fig, axs = plt.subplots(num_segments, 1, figsize=(12, 3*num_segments), sharex=False)
    fig.suptitle("Shortest Segments in Trace", fontsize=16, y=0.95)
    
    if num_segments == 1:
        axs = [axs]
    
    for i, seg in enumerate(shortest_segments):
        start_idx = max(0, seg - context_segments)
        end_idx = min(len(segpoints) - 1, seg + context_segments + 1)
        
        start = segpoints[start_idx]
        end = segpoints[end_idx]
        
        segment_data = trace[start:end]
        x_range = range(start, end)
        
        axs[i].plot(x_range, segment_data)
        
        for j in range(start_idx, end_idx):
            seg_start = segpoints[j]
            axs[i].axvline(x=seg_start, color='r', linestyle='--', alpha=0.5)
            axs[i].text(seg_start, axs[i].get_ylim()[1], f'{j}', 
                        rotation=45, va='bottom', ha='right', fontsize=8)
        
        main_segment_start = segpoints[seg]
        main_segment_end = segpoints[seg+1]
        axs[i].axvspan(main_segment_start, main_segment_end, alpha=0.2, color='yellow')
        
        axs[i].set_title(f"Segment {seg} (Length: {d[seg]})", pad=20)
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlim(start, end)
        
        # Add padding to y-axis
        y_min, y_max = np.min(segment_data), np.max(segment_data)
        y_range = y_max - y_min
        axs[i].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        axs[i].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
    axs[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust top margin
    plt.show()
