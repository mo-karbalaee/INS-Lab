import numpy as np
import matplotlib.pyplot as plt
#Planing
# take recording data strcture from dataset class
# prepcrocessing:
# 

class preprocess():
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.reorganize_structure()


    def remove_bad_channel(self, bad_channels: list):
        """
        Remove channels from biosignal in data_dict for all participants according to bad_channels

        Args:
            bad_channels (list): list of noisy or non-EMG channel indices.
        """
        for subject in self.data_dict.items():
            for gesture in subject[1].items():
                for recording in gesture[1]:
                    recording['biosignal'] = np.delete(recording['biosignal'], bad_channels, axis=0)
    
    def reorganize_structure(self):
        """
        Reorganize structure:
        """
        for subject in self.data_dict.items():
            for gesture in subject[1].items():
                for recording in gesture[1]:
                    biosignal = recording['biosignal'] # (38, 18, frames)
                    ground_truth = recording['ground_truth'] # (9, label_frames)
                    biosignal_timings = recording['biosignal_timings']  # (frames,)
                    ground_truth_timings = recording['ground_truth_timings']  # (label_frames,)

                    channels, samples_per_frame, num_frames = biosignal.shape
                    biosignal_reordered = biosignal.transpose(0, 2, 1)
                    recording['biosignal'] = biosignal_reordered.reshape(channels, samples_per_frame * num_frames)
                    total_samples = recording['biosignal'].shape[1]
                    within_frame_offsets = np.arange(samples_per_frame) / 2000
                    
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

                    recording['biosignal_timings'] = consolidated_timings
                    recording['ground_truth'] = upsampled_gt

    def normalize(self):
        pass

    def cut_signal(self,
                   threshold: float = 0.99,
                   min_duration_sec: float = 2,
                   merge_gap_sec: float = 0.05,
                   pad_pre_sec: float = 0.0,
                   pad_post_sec: float = 0.0,
                   rest_segments: int = 5,
                   rest_segment_duration_sec: float = 4,
                   truncate_to_shortest: bool = True,
                   fs: int = 2000):
        """Cut each recording into gesture segments based on the active ground truth channel."""
        min_duration = max(1, int(min_duration_sec * fs))
        merge_gap = int(merge_gap_sec * fs)
        pad_pre = int(pad_pre_sec * fs)
        pad_post = int(pad_post_sec * fs)
        rest_segment_length = int(rest_segment_duration_sec * fs)

        segments_by_gesture = {}

        for subject, gestures in self.data_dict.items():
            for gesture, recordings in gestures.items():
                for rec_idx, recording in enumerate(recordings):
                    gt = recording['ground_truth']
                    if gt.ndim != 2:
                        raise ValueError(f"Expected ground_truth shape (channels, samples), got {gt.shape}")

                    active_gt_idx = self._infer_active_gt_idx(gt)

                    segments = []
                    if active_gt_idx is None:
                        total_samples = recording['biosignal'].shape[1]
                        left_time = total_samples - rest_segments * rest_segment_length
                        start_point = round(left_time / rest_segments)
                        for seg_id in range(rest_segments):
                            start = start_point + (seg_id * rest_segment_length) + (start_point * seg_id)
                            end = start + rest_segment_length
                            if end > total_samples:
                                break

                            segment = {
                                'subject': subject,
                                'gesture': gesture,
                                'recording_index': rec_idx,
                                'segment_index': seg_id,
                                'active_gt_idx': None,
                                'start': start,
                                'end': end,
                                'biosignal': recording['biosignal'][:, start:end],
                                'ground_truth': gt[:, start:end],
                                'timings': recording['biosignal_timings'][start:end]
                            }
                            segments.append(segment)
                            segments_by_gesture.setdefault(gesture, []).append(segment)

                        recording['segments'] = segments
                        continue

                    gt_signal = gt[active_gt_idx, :]
                    events = self._find_gesture_events(gt_signal, threshold, min_duration, merge_gap)

                    for seg_id, (start, end) in enumerate(events):
                        start = max(0, start - pad_pre)
                        end = min(gt_signal.shape[0], end + pad_post)

                        segment = {
                            'subject': subject,
                            'gesture': gesture,
                            'recording_index': rec_idx,
                            'segment_index': seg_id,
                            'active_gt_idx': active_gt_idx,
                            'start': start,
                            'end': end,
                            'biosignal': recording['biosignal'][:, start:end],
                            'ground_truth': gt[:, start:end],
                            'timings': recording['biosignal_timings'][start:end]
                        }
                        segments.append(segment)
                        segments_by_gesture.setdefault(gesture, []).append(segment)

                    recording['segments'] = segments
        
        if truncate_to_shortest:
            self.truncate_segments_to_shortest(segments_by_gesture)

        return segments_by_gesture
    

    def truncate_segments_to_shortest(self, segments_by_gesture):
        """Truncate all segments to the length of the shortest segment."""
        shortest = None
        for segments_list in segments_by_gesture.values():
            for seg in segments_list:
                length = seg['biosignal'].shape[1]
                if shortest is None or length < shortest:
                    shortest = length

        if shortest is None or shortest <= 0:
            return

        for segments_list in segments_by_gesture.values():
            for seg in segments_list:
                seg['biosignal'] = seg['biosignal'][:, :shortest]
                seg['ground_truth'] = seg['ground_truth'][:, :shortest]
                seg['timings'] = seg['timings'][:shortest]
                seg['end'] = seg['start'] + shortest

    def _infer_active_gt_idx(self, ground_truth):
        """Infer the active ground truth channel from the remapped GT array."""
        activity = np.sum(np.abs(ground_truth), axis=1)
        if np.any(activity > 0):
            return int(np.argmax(activity))
        return None

    def _find_gesture_events(self, gt_signal, threshold, min_duration, merge_gap):
        """Find gesture start/end pairs from a single GT channel."""
        active = gt_signal >= threshold
        active_int = active.astype(np.int32)
        edges = np.diff(active_int, prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]

        events = []
        for start, end in zip(starts, ends):
            if end - start >= min_duration:
                events.append((start, end))

        if not events:
            return []

        merged = [events[0]]
        for start, end in events[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def filter_signal(self, filter_list: list):
        """
        Filters all biosignals individually from all praticipants using filter_list.

        Args:
            filter_list (list): List of filters to apply and order, i.e. ['bandpass', 'notch']: bandpass filter -> notch filter
        """
        for subject in self.data_dict.items():
            for gesture in subject[1].items():
                for recording in gesture[1]:
                    for filter_name in filter_list:
                        if filter_name == 'bandpass':
                            recording['biosignal'] = self._bandpass_filter(recording['biosignal'])
                        elif filter_name == 'notch':
                            recording['biosignal'] = self._notch_filter(recording['biosignal'])




    def _bandpass_filter(self, signal):
        # expect 2khz freq
        for ch in range(signal.shape[0]):
            signal[ch, :] = self.__bandpass_filter(
                signal=signal[ch,:],
                fs=2000,
                lowcut=20,
                highcut=400,
                order=4
            )

        return signal
    
    def _notch_filter(self, signal):
        # expect 2khz freq
        for ch in range(signal.shape[0]):
            signal[ch, :] = self.__notch_filter(
                signal=signal[ch,:],
                fs=2000,
                freq_notch=50,
                Q = 30
            )

        return signal

    def __bandpass_filter(self, signal, fs, lowcut, highcut, order):
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

    def __notch_filter(self, signal, fs, freq_notch, Q):
        """Apply a notch filter to a 1D signal."""
        try:
            from scipy.signal import iirnotch, filtfilt
        except ImportError as e:
            raise ImportError("scipy is required for notch filtering. Install it with pip install scipy") from e

        b, a = iirnotch(freq_notch, Q, fs)
        return filtfilt(b, a, signal)
    
    def plot_signal(self, signal):
        plt.plot(signal)

    def _iterall(self, obj):
        pass

    def return_example_recording(self, subject, gesture, recording_idx, channel, plot: False):
        subject_data = self.data_dict[subject]
        gesture_data = subject_data[gesture]
        recording = gesture_data[recording_idx]
        channel_signal = recording['biosignal'][channel]
        active_idx = recording.get('active_gesture_idx', 0)
        ground_truth = recording['ground_truth'][active_idx]

        if plot:
            plt.subplot(2, 1, 1)
            plt.plot(channel_signal)

            plt.subplot(2, 1, 2, sharex=plt.gca())
            plt.plot(ground_truth)
            plt.show()

        return channel_signal, ground_truth


