"""
Test script for EMG recording analysis.
Demonstrates:
1. Loading recordings via dataset class
2. Analyzing ground truth labels
3. Custom segmentation logic
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Add classification module to path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import dataset








def consolidate_recording(recording_dict, target_fs=2000, label_fs=60, n_biosignal_channels=32):
    """
    Consolidate a single recording:
    1. Flatten frames into continuous array
    2. Select only biosignal channels (0-31, exclude auxiliary)
    3. Upsample ground truth labels to target sampling frequency
    
    Args:
        recording_dict: dict from dataset with biosignal, ground_truth, timings
        target_fs: target sampling frequency (2000 Hz)
        label_fs: original label sampling frequency (60 Hz)
        n_biosignal_channels: number of biosignal channels to keep (32)
    
    Returns:
        dict with consolidated data:
            - 'biosignal': (n_biosignal_channels, total_samples)
            - 'ground_truth': (9, total_samples)
            - 'timings': timing array for the consolidated samples
            - 'metadata': original metadata
    """
    
    # Extract data
    biosignal = recording_dict['biosignal']  # (38, 18, frames)
    ground_truth = recording_dict['ground_truth']  # (9, label_frames)
    biosignal_timings = recording_dict['biosignal_timings']  # (frames,)
    ground_truth_timings = recording_dict['ground_truth_timings']  # (label_frames,)
    
    print(f"Original shapes:")
    print(f"  biosignal: {biosignal.shape}")
    print(f"  ground_truth: {ground_truth.shape}")
    

    # The raw shape is (channels, samples_per_frame, frames).
    # reorder: (channels, frames, samples_per_frame)
    # and then flatten (perviously it wrongly flattened)
    channels, samples_per_frame, num_frames = biosignal.shape
    biosignal_reordered = biosignal.transpose(0, 2, 1)
    biosignal_flat = biosignal_reordered.reshape(channels, samples_per_frame * num_frames)
    
    # Select only biosignal channels (0-31)
    biosignal_consolidated = biosignal_flat[:n_biosignal_channels, :]
    
    print(f"\nAfter flattening and channel selection:")
    print(f"  biosignal: {biosignal_consolidated.shape}")
    
    # Create timing array for each sample within each frame.
    total_samples = biosignal_consolidated.shape[1]
    within_frame_offsets = np.arange(samples_per_frame) / target_fs
    consolidated_timings = np.repeat(biosignal_timings, samples_per_frame) + np.tile(within_frame_offsets, num_frames)
    
 
    # Interpolate from (9, label_frames) 60 Hz -> (9, total_samples) 2000 Hz
    upsampled_gt = np.zeros((ground_truth.shape[0], total_samples))
    
    for ch in range(ground_truth.shape[0]):
        # Linear interpolation from label timing to consolidated timing
        upsampled_gt[ch, :] = np.interp(
            consolidated_timings,
            ground_truth_timings,
            ground_truth[ch, :],
            left=0.0,
            right=0.0
        )
    
    print(f"After upsampling ground truth:")
    print(f"  ground_truth: {upsampled_gt.shape}")
    print(f"  Duration: {total_samples / target_fs:.2f}s")

    # Plot example: all channels stacked and sum plot with ground truth overlay
    if False:
        plot_samples = min(total_samples, int(target_fs * 50))
        plot_times = consolidated_timings[:plot_samples]
        plot_signals = biosignal_consolidated[:, :plot_samples]

        max_amplitude = np.max(np.abs(plot_signals))
        offset = max_amplitude * 4.0
        stacked = plot_signals + np.arange(plot_signals.shape[0])[:, None] * offset

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for ch in range(stacked.shape[0]):
            axs[0].plot(plot_times, stacked[ch], lw=0.8)
        axs[0].set_title('Stacked biosignal channels (first 2 seconds)')
        axs[0].set_ylabel('channel + offset')
        axs[0].set_yticks(np.arange(plot_signals.shape[0]) * offset)
        axs[0].set_yticklabels([f'C{ch}' for ch in range(plot_signals.shape[0])])
        axs[0].grid(True, alpha=0.2)

        sum_signal = np.sum(np.abs(plot_signals), axis=0)
        gt_example = upsampled_gt[0, :plot_samples]
        gt_norm = (gt_example - gt_example.min()) / (np.ptp(gt_example) + 1e-9)
        gt_scaled = gt_norm * (sum_signal.max() - sum_signal.min()) + sum_signal.min()

        axs[1].plot(plot_times, sum_signal, label='sum(abs(channels))', color='tab:blue')
        axs[1].plot(plot_times, gt_scaled, label='ground truth (scaled)', color='tab:red', alpha=0.75)
        axs[1].set_title('Summed channel magnitude and ground truth overlay')
        axs[1].set_xlabel('time (s)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    # Return consolidated data
    return {
        'biosignal': biosignal_consolidated,  # (32, total_samples)
        'ground_truth': upsampled_gt,  # (9, total_samples)
        'timings': consolidated_timings,
        'sampling_frequency': target_fs,
        'metadata': {
            'filename': recording_dict['filename'],
            'task': recording_dict['task'],
            'device_info': recording_dict['device_info'],
            'active_gesture_idx': recording_dict['active_gesture_idx'],
        }
    }


def bandpass_filter(signal, fs=2000, lowcut=20, highcut=300, order=4):
    """Apply a Butterworth bandpass filter to a 1D signal."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as e:
        raise ImportError("scipy is required for bandpass filtering. Install it with pip install scipy") from e

    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def plot_recording_channels(recording_dict,
                             channels_per_page=8,
                             target_fs=2000,
                             label_fs=60,
                             n_biosignal_channels=32,
                             max_seconds=15):
    """Plot all biosignal channels in groups of channels_per_page plus a ground truth subplot.

    Args:
        recording_dict: recording dict from dataset
        channels_per_page: number of biosignal channels shown per figure
        target_fs: target signal frequency used during consolidation
        label_fs: original label sampling frequency, unused here but kept for API compatibility
        n_biosignal_channels: number of biosignal channels to use
        max_seconds: maximum duration shown in seconds
    """
    consolidated = consolidate_recording(
        recording_dict,
        target_fs=target_fs,
        label_fs=label_fs,
        n_biosignal_channels=n_biosignal_channels
    )

    biosignal = consolidated['biosignal']
    ground_truth = consolidated['ground_truth']
    timings = consolidated['timings']
    task = consolidated['metadata'].get('task', 'unknown')
    filename = consolidated['metadata'].get('filename', 'recording')
    active_gt_idx = consolidated['metadata'].get('active_gesture_idx', None)

    total_samples = biosignal.shape[1]
    max_samples = min(total_samples, int(max_seconds * target_fs))
    plot_times = timings[:max_samples]

    # Apply bandpass filtering only for visualization
    filtered_biosignal = np.zeros_like(biosignal[:, :max_samples])
    for ch in range(min(biosignal.shape[0], n_biosignal_channels)):
        filtered_biosignal[ch, :] = bandpass_filter(
            biosignal[ch, :max_samples],
            fs=target_fs,
            lowcut=20,
            highcut=300,
            order=4
        )

    # Fallback: infer active ground truth if metadata did not contain a valid index
    if active_gt_idx is None or active_gt_idx < 0 or active_gt_idx >= ground_truth.shape[0]:
        activity = np.sum(np.abs(ground_truth), axis=1)
        if np.any(activity > 0):
            active_gt_idx = int(np.argmax(activity))
        else:
            active_gt_idx = None

    n_pages = int(np.ceil(n_biosignal_channels / channels_per_page))
    for page in range(n_pages):
        start_ch = page * channels_per_page
        end_ch = min(start_ch + channels_per_page, n_biosignal_channels)
        page_channels = list(range(start_ch, end_ch))

        n_plots = len(page_channels) + 1  # plus ground truth
        fig, axs = plt.subplots(n_plots, 1, figsize=(16, 2.4 * n_plots), sharex=True)

        if n_plots == 2:
            axs = [axs[0], axs[1]]

        # Per-channel plots
        y_min = np.min(biosignal[page_channels, :max_samples])
        y_max = np.max(biosignal[page_channels, :max_samples])
        y_margin = max(1e-3, (y_max - y_min) * 0.05)

        for idx, ch in enumerate(page_channels):
            ax = axs[idx]
            ax.plot(plot_times, filtered_biosignal[ch, :max_samples], color='tab:blue', lw=0.7)
            ax.set_ylabel(f'C{ch}')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Ground truth subplot at the bottom
        gt_ax = axs[-1]
        if active_gt_idx is not None:
            gt_signal = ground_truth[active_gt_idx, :max_samples]
            gt_ax.plot(plot_times, gt_signal, label=f'GT{active_gt_idx}', color='tab:red', lw=1.2)
            gt_ax.set_title(f'Active ground truth channel GT{active_gt_idx} for {filename} / {task}')
            gt_ax.legend(fontsize='small', loc='upper right')
        else:
            gt_ax.text(0.5, 0.5, 'No active ground truth channel detected',
                       ha='center', va='center', transform=gt_ax.transAxes,
                       color='tab:red', fontsize=12)
            gt_ax.set_title(f'No active ground truth channel for {filename} / {task}')

        gt_ax.set_ylabel('ground truth')
        gt_ax.set_xlabel('time (s)')
        gt_ax.grid(True, alpha=0.2)

        fig.suptitle(f'Channels {start_ch}-{end_ch - 1} plus active ground truth (first {max_seconds}s)', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()





path_to_raw = r"C:\Users\leonv\AIBE_LAB_new\aibe_ins_lab_recordings\raw"
ds = dataset(path_to_raw)
ds.summary()

data_dict = ds.get_data()

# Access a specific recording
if 'mohammad' in data_dict and 'Grasp' in data_dict['mohammad']:
    recordings_gesture_0 = data_dict['franzi']['Cellphone']
    print(f"\nFound {len(recordings_gesture_0)} recordings for mohammad gesture_0")
    recordings_gesture_0_0 = recordings_gesture_0[0]
consolidated = consolidate_recording(recordings_gesture_0_0)
plot_recording_channels(recordings_gesture_0_0, channels_per_page=3, max_seconds=15)

print(f"\nConsolidated output:")
print(f"  biosignal shape: {consolidated['biosignal'].shape}")
print(f"  ground_truth shape: {consolidated['ground_truth'].shape}")
print(f"  timings shape: {consolidated['timings'].shape}")
print(f"  Total duration: {consolidated['timings'][-1] - consolidated['timings'][0]:.2f}s")