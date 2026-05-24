import pandas
import numpy as np
from pingouin import intraclass_corr




class analyse_repeatability():
    def __init__(self, segments, data_dict):
        self.segments = segments
        self.data_dict = data_dict
        self.df_repeat = pandas.DataFrame

    def do_something(self):
        pass


    def get_features(self):
        # Plan:
        # for each gesture and participant:
        # get RMS, MAV and WL as features (maybe also iEMG?) Maybe also use frequency domain features?
        # double check frequeny domain features ... completely vibecoded

        fs = 2000

        for gesture, segments_list in self.segments.items():
            for seg in segments_list:
                signal = seg['biosignal']

                # Time domain features

                # Mean absolute value
                mav = np.mean(np.abs(signal), axis=1)
                # Root mean squared
                rms = np.sqrt(np.mean(signal**2, axis=1))
                # Wave length
                wl = np.sum(np.abs(np.diff(signal, axis=1)), axis=1)

                # Frequency domain features
                fft_vals = np.fft.rfft(signal, axis=1)
                freqs = np.fft.rfftfreq(signal.shape[1], d=1/fs)
                power = np.abs(fft_vals) ** 2

                # Peak frequency
                peak_freq = freqs[np.argmax(power, axis=1)]
                # Mean frequency
                mnf = np.sum(freqs * power, axis=1) / np.sum(power, axis=1)
                # Median frequeny
                mdf = np.array([median_frequency(power[ch], freqs) for ch in range(power.shape[0])])
                
                features_ch = {'mav': mav,
                            'rms': rms,
                            'wl': wl,
                            'peak_freq': peak_freq,
                            'mnf': mnf,
                            'mdf': mdf
                }

                features_mean = {'mav': mav.mean(),
                            'rms': rms.mean(),
                            'wl': wl.mean(),
                            'peak_freq': peak_freq.mean(),
                            'mnf': mnf.mean(),
                            'mdf': mdf.mean()
                }

                seg['features_ch'] = features_ch
                seg['features_mean'] = features_mean

    def get_repeat_metrics(self):
        # Plan:
        # need pandas.DataFrame long format 
        # for each features mean, gesture and participant:
        # get ICC, EMS, MDC
        if self.df_repeat.empty:
            print('first generate df using get_df_from_segments')

        data_repeat = []
        for (subject, gesture, feature), df_sub in df.groupby(['subject', 'gesture', 'feature']):
            
            print('test')

    def get_df_from_segments(self):
        # Plan:
        # need: subjects, gestures, trials and features in one df long format
        data = []
        
        for gesture, segments_list in self.segments.items():
            for seg in segments_list:
                subject = seg['subject']
                recording_index = seg['recording_index']
                segment_index = seg['segment_index']
                trial = recording_index * 5 + segment_index
                
                # Features aus features_mean extrahieren
                features_mean = seg.get('features_mean', {})
                
                for feature_name, feature_value in features_mean.items():
                    data.append({
                        'subject': subject,
                        'gesture': gesture,
                        'trial': trial,
                        'feature': feature_name,
                        'value': feature_value
                    })
        
        self.df_repeat = pandas.DataFrame(data)
        return self.df_repeat


def median_frequency(power_row, freqs):
    cumulative = np.cumsum(power_row)
    cutoff = cumulative[-1] / 2
    return freqs[np.searchsorted(cumulative, cutoff)]