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
from preprocessing import preprocess
from repeatability.analyse import analyse_repeatability
from repeatability.visualize import vizualize_repeatability
from classification.preprocess_for_classification import get_data_for_model
from classification.trainandevalmodel import train_model_for_part







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


def plot_all_segments(preprocessor, segments_by_gesture, channel_idx=0, max_segments_per_recording=5):
    """
    Plot segments grouped by recording with shared time axis.
    For each recording: full signal + GT at top, then individual segments below.
    All subplots share the same x-axis (time), so segments appear at their true positions.

    Args:
        preprocessor: the preprocess instance to access full recordings
        segments_by_gesture (dict): output from preprocessor.cut_signal()
        channel_idx (int): which channel to plot (default 0)
        max_segments_per_recording (int): how many segments to show per recording (default 5)
    """
    for gesture, segments_list in segments_by_gesture.items():
        if not segments_list:
            continue

        # Group segments by (subject, recording_index)
        by_recording = {}
        for seg in segments_list:
            key = (seg['subject'], seg['recording_index'])
            if key not in by_recording:
                by_recording[key] = []
            by_recording[key].append(seg)

        # Plot each recording
        for (subject, rec_idx), rec_segs in by_recording.items():
            # Get full recording from preprocessor
            gestures_for_subject = preprocessor.data_dict[subject]
            recordings_for_gesture = gestures_for_subject[gesture]
            full_recording = recordings_for_gesture[rec_idx]

            full_biosignal = full_recording['biosignal']
            full_gt = full_recording['ground_truth']
            full_timings = full_recording['biosignal_timings']

            rec_segs_sorted = sorted(rec_segs, key=lambda x: x['start'])

            n_plots = 2 + min(len(rec_segs_sorted), max_segments_per_recording)
            fig, axs = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), sharex=True)

            # Get global time limits from full recording
            t_min = full_timings[0] if len(full_timings) > 0 else 0
            t_max = full_timings[-1] if len(full_timings) > 0 else 1

            # Plot 1: Full signal
            ax = axs[0]
            signal_full = full_biosignal[channel_idx, :]
            ax.plot(full_timings, signal_full, color='tab:blue', lw=0.8)
            ax.set_title(f"{subject} - Recording {rec_idx} - Full Signal (Channel {channel_idx})")
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.2)
            ax.set_xlim([t_min, t_max])

            # Plot 2: Ground truth
            ax = axs[1]
            active_idx = rec_segs[0]['active_gt_idx']
            if active_idx == None:
                active_idx = 1 # just for plotting
            gt_signal = full_gt[active_idx, :]
            ax.plot(full_timings, gt_signal, color='tab:red', lw=1.2)
            ax.set_title(f"Ground Truth (Channel {active_idx})")
            ax.set_ylabel('GT Value')
            ax.grid(True, alpha=0.2)
            ax.set_ylim([-0.1, 1.1])
            ax.set_xlim([t_min, t_max])

            # Plots 3+: Individual segments with same time axis
            for seg_plot_idx, seg in enumerate(rec_segs_sorted[:max_segments_per_recording]):
                ax = axs[2 + seg_plot_idx]
                timings = seg['timings']
                signal = seg['biosignal'][channel_idx, :]
                ax.plot(timings, signal, color='tab:blue', lw=0.8)
                duration = timings[-1] - timings[0] if len(timings) > 1 else 0
                ax.set_title(f"Segment {seg['segment_index']} ({duration:.2f}s)")
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.2)
                ax.set_xlim([t_min, t_max])

            axs[-1].set_xlabel('Time (s)')
            fig.suptitle(f"Gesture: {gesture}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()


def print_segment_lengths(segments_by_gesture, fs=2000):
    """Print all segment lengths for the cut_signal output."""
    rows = []
    for gesture, segments_list in segments_by_gesture.items():
        for seg in segments_list:
            n_samples = seg['biosignal'].shape[1]
            duration_s = n_samples / fs
            rows.append({
                'gesture': gesture,
                'subject': seg['subject'],
                'recording_index': seg['recording_index'],
                'segment_index': seg['segment_index'],
                'start_sample': seg['start'],
                'end_sample': seg['end'],
                'n_samples': n_samples,
                'duration_s': duration_s,
            })

    df = pd.DataFrame(rows)
    print(df['gesture'].unique())
    print(df['gesture'].value_counts())
    print(sorted(df['gesture'].unique()))
    if df.empty:
        print('No segments found.')
        return df

    df = df.sort_values(['gesture', 'subject', 'recording_index', 'segment_index'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df['gesture'].unique())
    print(df['gesture'].value_counts())
    print(sorted(df['gesture'].unique()))
    print(df.to_string(index=False))
    df.to_csv("data_frame")
    return df





path_to_raw = r"C:\Users\leonv\AIBE_LAB_new\aibe_ins_lab_recordings\raw"
path_to_output = r"C:\Users\leonv\AIBE_LAB_new\output"
do_8_channels = False

# Get dataset and structure raw data:
ds = dataset(path_to_raw)
data_dict = ds.get_data()

# Preprocess:
preprocessor = preprocess(data_dict)
if do_8_channels:
    bad_channels = [32,33,34,35,36,37]
else:
    bad_channels = [9,32,33,34,35,36,37]
filter_list = ['bandpass', 'notch']
preprocessor.remove_bad_channel(bad_channels)
preprocessor.filter_signal(filter_list)
segments = preprocessor.cut_signal(truncate_to_shortest=False)


print_segment_lengths(segments)

if False:
    # Repeatability:
    repeatability_analyzer = analyse_repeatability(segments, data_dict)
    repeatability_analyzer.get_features()
    df_repeat = repeatability_analyzer.get_df_from_segments()
    metric_dict, metrics_df = repeatability_analyzer.get_repeat_metrics()

    # Visualize Repeatability:
    repeatability_visualizer = vizualize_repeatability(segments, data_dict, df_repeat, metric_dict, metrics_df, path_to_output)
    repeatability_visualizer.plot_heatmap_features_gestures_per_participant(True)
    repeatability_visualizer.plot_feature_repeat_ranking(True)
    repeatability_visualizer.plot_gesture_repeat_ranking(True)
    repeatability_visualizer.plot_mixed_model_results(True)

# Classification:
if True:
    split_data_per_part, label_map = get_data_for_model(segments, use_freq=True, do_8_channel=do_8_channels) # get training and testing split for each participant
    train_model_for_part(split_data_per_part,'franzi', label_map, do_8_channels, path_to_output)
    train_model_for_part(split_data_per_part,'mohammad', label_map, do_8_channels, path_to_output)
    train_model_for_part(split_data_per_part,'leon', label_map, do_8_channels, path_to_output)




# Visualize all segments
#plot_all_segments(preprocessor, segments, channel_idx=0, max_segments_per_recording=4)

proband = 'leon'
gesture = 'Cellphone'
recording_idx = 1
channel = 0

example_ground_trouth = preprocessor.return_example_recording(proband, gesture, recording_idx, channel, True)











# Access a specific recording
#if 'leon' in data_dict and 'Grasp' in data_dict['leon']:
#    recordings_gesture_0 = data_dict['leon']['Cellphone']
#    print(f"\nFound {len(recordings_gesture_0)} recordings for leon gesture_0")
#    recordings_gesture_0_0 = recordings_gesture_0[0]
#consolidated = consolidate_recording(recordings_gesture_0_0)
#plot_recording_channels(recordings_gesture_0_0, channels_per_page=3, max_seconds=15)
