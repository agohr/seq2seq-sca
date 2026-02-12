import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from processing import resize_segment

def get_default_params():
    """
    Returns a dictionary of default parameters for trace augmentation.
    
    Returns:
        dict: A dictionary containing default parameters for all augmentation methods,
              including noise levels, interrupt characteristics, time warping strength,
              and which augmentations are enabled by default.
    """
    return {
        "noise": {"mean": 0, "std_dev": 0.01},
        "interrupts": {"num_interrupts": 5, "max_duration": 10},
        "time_warp": {"time_warp_strength": 0.1},
        "power_management": {"num_changes": 3},
        "context_switch": {"num_switches": 2, "max_duration": 100},
        "frequency_change": {"num_changes": 2, "max_duration": 200, "max_slowdown": 2, "max_power_reduction": 0.5},
        "enabled_augmentations": ["noise", "interrupts", "jitter", "cache", "power_management", "context_switch", "frequency_change"]
    }

def add_gaussian_noise(trace, mean=0, std_dev=0.01):
    """
    Adds Gaussian noise to the trace.

    Rationale: Gaussian noise is added to simulate the random fluctuations in power
    consumption caused by various background processes and electronic noise in a
    complex system.
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
        std_dev (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.01.
        
    Returns:
        numpy.ndarray: The trace with added Gaussian noise
    """
    noise = np.random.normal(mean, std_dev, len(trace))
    return trace + noise

def simulate_interrupts(trace, num_interrupts=5, max_duration=10):
    """
    Simulates interrupts in the trace.

    Rationale: Interrupts are simulated as sudden spikes in power consumption,
    representing the CPU briefly switching to handle an interrupt service routine.
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        num_interrupts (int, optional): Number of interrupts to simulate. Defaults to 5.
        max_duration (int, optional): Maximum duration of each interrupt in samples. Defaults to 10.
        
    Returns:
        numpy.ndarray: The trace with simulated interrupts
    """
    for _ in range(num_interrupts):
        start = np.random.randint(0, len(trace))
        duration = np.random.randint(1, max_duration)
        interrupt_amplitude = np.random.uniform(0.5, 2) * np.mean(trace)
        trace[start:start+duration] += interrupt_amplitude
    return trace

def add_time_warp(trace, jitter_strength=0.1):
    """
    Adds clock jitter to the trace.

    Rationale: Clock jitter is simulated by slightly shifting the time axis of the trace.
    This represents the small variations in timing that occur in real systems due to
    imperfect clock sources.
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        jitter_strength (float, optional): Standard deviation of the time shifts. Defaults to 0.1.
        
    Returns:
        numpy.ndarray: The trace with added time warping/jitter
    """
    time_axis = np.arange(len(trace))
    jittered_time = time_axis + np.random.normal(0, jitter_strength, len(trace))
    return np.interp(time_axis, jittered_time, trace)

def add_power_management_artifacts(trace, num_changes=3):
    """
    Adds power management artifacts to the trace.

    Rationale: Power management artifacts are simulated as step changes in the baseline
    power consumption. This represents the system entering different power states
    (e.g., low power modes) or frequency scaling events.
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        num_changes (int, optional): Number of power state changes to simulate. Defaults to 3.
        
    Returns:
        numpy.ndarray: The trace with added power management artifacts
    """
    for _ in range(num_changes):
        start = np.random.randint(0, len(trace))
        end = np.random.randint(start, len(trace))
        shift = np.random.uniform(-0.1, 0.1) * np.mean(trace)
        trace[start:end] += shift
    return trace

def simulate_context_switch(trace, num_switches=2, max_duration=100):
    """
    Simulates context switches in the trace.

    Rationale: Context switches are simulated by inserting random segments from
    elsewhere in the trace. This represents the CPU switching from the cryptographic
    task to other tasks, which may have different power consumption patterns.
    The inserted segments are slightly modified to represent the variability
    in power consumption between different executions of the same task.
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        num_switches (int, optional): Number of context switches to simulate. Defaults to 2.
        max_duration (int, optional): Maximum duration of each context switch in samples. Defaults to 100.
        
    Returns:
        numpy.ndarray: The trace with simulated context switches
    """
    original_length = len(trace)
    augmented_trace = trace.copy()
    
    for _ in range(num_switches):
        # Choose a random location for the context switch
        switch_start = np.random.randint(0, len(augmented_trace) - max_duration)
        switch_duration = np.random.randint(20, max_duration)
        
        # Select a random segment from elsewhere in the trace
        segment_start = np.random.randint(0, len(augmented_trace) - switch_duration)
        while abs(segment_start - switch_start) < switch_duration:
            segment_start = np.random.randint(0, len(augmented_trace) - switch_duration)
        
        switch_segment = augmented_trace[segment_start:segment_start + switch_duration].copy()
        
        # Apply random shift and scale to the switch segment
        shift = np.random.uniform(-0.1, 0.1) * np.mean(switch_segment)
        scale = np.random.uniform(0.9, 1.1)
        switch_segment = (switch_segment + shift) * scale
        
        # Insert the modified segment into the trace
        augmented_trace = np.concatenate([
            augmented_trace[:switch_start],
            switch_segment,
            augmented_trace[switch_start + switch_duration:]
        ])
    
    # Resize the augmented trace to match the original length
    augmented_trace = resize_segment(augmented_trace, original_length)
    
    return augmented_trace

def simulate_frequency_change(trace, num_changes=2, max_duration=200, max_slowdown=2, max_power_reduction=0.5):
    """
    Simulates frequency changes in the trace by time warping and amplitude reduction.
    
    The function:
    1. Selects random segments to modify
    2. Stretches these segments according to the slowdown factor
    3. Reduces their amplitude to simulate power reduction
    4. Concatenates everything and resamples back to original length
    
    Args:
        trace (numpy.ndarray): The input trace to be augmented
        num_changes (int, optional): Number of frequency changes to simulate. Defaults to 2.
        max_duration (int, optional): Maximum duration of each frequency change in samples. Defaults to 200.
        max_slowdown (float, optional): Maximum factor by which to slow down segments. Defaults to 2.
        max_power_reduction (float, optional): Maximum proportional reduction in power. Defaults to 0.5.
        
    Returns:
        numpy.ndarray: The trace with simulated frequency changes
    """
    trace_out = trace.copy()
    segments = []
    last_end = 0
    
    # Sort the change points to process segments in order
    change_points = []
    for _ in range(num_changes):
        start = np.random.randint(0, len(trace) - max_duration)
        duration = np.random.randint(max_duration // 2, max_duration)
        change_points.append((start, duration))
    change_points.sort(key=lambda x: x[0])
    
    # Process each segment
    for start, duration in change_points:
        # Add unchanged segment before the change point
        if start > last_end:
            segments.append(trace_out[last_end:start])
            
        # Process the frequency change segment
        end = start + duration
        slowdown = np.random.uniform(1, max_slowdown)
        power_reduction = np.random.uniform(1 - max_power_reduction, 1)
        
        # Stretch the segment according to slowdown factor
        stretched_length = int(duration * slowdown)
        stretched_time = np.linspace(0, duration-1, stretched_length)
        original_time = np.arange(duration)
        interpolator = interp1d(original_time, trace[start:end], kind='linear')
        stretched_segment = interpolator(stretched_time)
        
        # Apply power reduction
        stretched_segment *= power_reduction
        segments.append(stretched_segment)
        
        last_end = end
    
    # Add the final unchanged segment
    if last_end < len(trace):
        segments.append(trace_out[last_end:])
    
    # Concatenate all segments and resize to original length
    concatenated = np.concatenate(segments)
    trace_out = resize_segment(concatenated, len(trace))
    
    return trace_out

def augment_trace(original_trace, params=None):
    """
    Augments the original trace based on the provided parameters.
    If no parameters are provided, default parameters are used.

    Args:
        original_trace (numpy.ndarray): The original side-channel trace
        params (dict, optional): A dictionary of parameters for various augmentations.
                                If None, default parameters are used. Defaults to None.
                                
    Returns:
        numpy.ndarray: The augmented trace with all enabled augmentations applied
        
    Note:
        The augmentations are applied in the order specified in the 'enabled_augmentations'
        list in the params dictionary. The final trace is resized to match the original
        trace length.
    """
    trace_length = len(original_trace)
    if params is None:
        params = get_default_params()
    else:
        # Merge with default params to ensure all keys exist
        default_params = get_default_params()
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
            elif isinstance(value, dict):
                params[key] = {**value, **params.get(key, {})}

    trace = original_trace.copy()
    
    augmentations = {
        "noise": lambda t: add_gaussian_noise(t, **params["noise"]),
        "interrupts": lambda t: simulate_interrupts(t, **params["interrupts"]),
        "time_warp": lambda t: add_time_warp(t, **params["jitter"]),
        "power_management": lambda t: add_power_management_artifacts(t, **params["power_management"]),
        "context_switch": lambda t: simulate_context_switch(t, **params["context_switch"]),
        "frequency_change": lambda t: simulate_frequency_change(t, **params["frequency_change"])
    }

    for aug in params["enabled_augmentations"]:
        if aug in augmentations:
            trace = augmentations[aug](trace)

    trace = resize_segment(trace, trace_length)

    return trace

# Example usage
if __name__ == "__main__":
    # Generate a sample trace (replace this with your actual trace data)
    original_trace = np.random.rand(1000)

    # Augment the trace using default parameters
    augmented_trace_default = augment_trace(original_trace)

    # Augment the trace using custom parameters
    custom_params = {
        "noise": {"std_dev": 0.02},
        "interrupts": {"num_interrupts": 10},
        "frequency_change": {"num_changes": 3, "max_slowdown": 1.5},
        "enabled_augmentations": ["noise", "interrupts", "frequency_change"]
    }
    augmented_trace_custom = augment_trace(original_trace, custom_params)

    print("Original trace shape:", original_trace.shape)
    print("Augmented trace (default) shape:", augmented_trace_default.shape)
    print("Augmented trace (custom) shape:", augmented_trace_custom.shape)
