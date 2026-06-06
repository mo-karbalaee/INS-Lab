import numpy as np
from repeatability.analyse import median_frequency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def get_data_for_model(segments, use_freq = True, normalize = True):

    # want to cut the signal into...
    WINDOW_SIZE = 500  # samples -> 250 ms at 2000 Hz

    label_map = {task: i for i, task in enumerate(segments.keys())}
    data_dict = {}

    for gesture, segments_list in segments.items():
        for seg in segments_list:
            subject = seg['subject']

            biosignal = seg['biosignal']  # shape: (channels, samples)

            data_dict.setdefault(subject, {})
            data_dict[subject].setdefault(gesture, [])

            segment_features = []

            n = biosignal.shape[1]

            for start in range(0, n - WINDOW_SIZE + 1, WINDOW_SIZE):
                end = start + WINDOW_SIZE

                window = biosignal[:, start:end]  # now shape: (channels, 500)

                # extract features

                # Time domain features

                # Mean absolute value
                mav = np.mean(np.abs(window), axis=1)
                # Root mean squared
                rms = np.sqrt(np.mean(window**2, axis=1))
                # Wave length
                wl = np.sum(np.abs(np.diff(window, axis=1)), axis=1)

                if use_freq:
                    # Frequency domain features

                    fft_vals = np.fft.rfft(window, axis=1)
                    freqs = np.fft.rfftfreq(window.shape[1], d=1/2000)
                    power = np.abs(fft_vals) ** 2

                    # Peak frequency
                    peak_freq = freqs[np.argmax(power, axis=1)]
                    # Mean frequency
                    mnf = np.sum(freqs * power, axis=1) / np.sum(power, axis=1)
                    # Median frequeny
                    mdf = np.array([median_frequency(power[ch], freqs) for ch in range(power.shape[0])])

                    features = np.concatenate([mav, rms, wl, peak_freq, mnf, mdf])
                else:
                    features = np.concatenate([mav, rms, wl])

                segment_features.append(features)

            data_dict[subject][gesture].append(segment_features)



    split_data_dict = get_training_testing_from_data(data_dict, label_map)

    return split_data_dict, label_map
        

def get_training_testing_from_data(data_dict, label_map):

    results = {}

    for subject, gestures in data_dict.items():
        x_train = []
        y_train = []

        x_test = []
        y_test = []

        for gesture, segment_list in gestures.items():

            label = label_map[gesture]

            if len(segment_list) == 0:
                continue

            # Split by segment, not by individual windows.
            # Use the last 3 segments for testing when possible.
            if len(segment_list) >= 4:
                n_test_segments = 3
            else:
                n_test_segments = max(1, len(segment_list) // 5)

            train_segments = segment_list[:-n_test_segments]
            test_segments = segment_list[-n_test_segments:]

            # Training
            for seg_windows in train_segments:
                x_train.extend(seg_windows)
                y_train.extend([label] * len(seg_windows))

            # Testing
            for seg_windows in test_segments:
                x_test.extend(seg_windows)
                y_test.extend([label] * len(seg_windows))

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

         # Fit scaler on training data and transform
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        results[subject] = {
        "training": {
            "x": x_train,
            "y": y_train,
        },
        "testing": {
            "x": x_test,
            "y": y_test,
        }
        }

    return results