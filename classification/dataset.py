from pathlib import Path
import zipfile
import tempfile
import shutil
import pickle as pkl



TASK_LIST = ["Rest", "Thumb", "Pinky", "Pinch", "Power Grasp"]

def get_data_split(path_to_recordings: Path):
    """
    Will split data at: 1046, 1117, 1145
    Will ignore files containing "fail"
    Only processes .pkl files
    """

    path_to_recordings = Path(path_to_recordings)

    # 1. Find the zip file
    zip_files = list(path_to_recordings.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No .zip file found in the given directory.")

    zip_path = zip_files[0]  # assume only one

    # 2. Create raw folders
    raw_dir = path_to_recordings / "raw"
    folder1 = raw_dir / "leon"
    folder2 = raw_dir / "franzi"
    folder3 = raw_dir / "mohammad"

    folder1.mkdir(parents=True, exist_ok=True)
    folder2.mkdir(parents=True, exist_ok=True)
    folder3.mkdir(parents=True, exist_ok=True)

    # 3. Extract zip to temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_path)

        # 4. Collect all valid .pkl files
        pkl_files = [
            p for p in tmp_path.rglob("*.pkl")
            if "fail" not in p.name
        ]

        # 5. Sort into folders based on time code
        for file_path in pkl_files:
            try:
                # Extract time segment: VHI_Recording_YYYYMMDD_HHMMxxxx...
                parts = file_path.stem.split("_")
                time_part = parts[3]  # e.g. 105107537602
                time_val = int(time_part[:4])  # e.g. 1051

                if time_val <= 1046:
                    target = folder1
                elif time_val <= 1117:
                    target = folder2
                else:
                    target = folder3

                shutil.copy2(file_path, target / file_path.name)

            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

        print("finished data sorting\n")


class dataset:
    def __init__(self, path_to_raw: Path):
        """
        Load and organize .pkl recordings from raw folder.
        
        Structure:
            path_to_raw/
                ├── leon/
                ├── franzi/
                └── mohammad/
        
        Output: dict with structure
            {
                'participant_name': {
                    'gesture_0': [recording_dict, ...],
                    'gesture_1': [recording_dict, ...],
                    ...
                },
                ...
            }
        """
        self.path_to_raw = Path(path_to_raw)
        self.recordings = {}
        
        # Find all participant folders
        participant_folders = [p for p in self.path_to_raw.iterdir() if p.is_dir()]
        
        for participant_folder in participant_folders:
            participant_name = participant_folder.name
            self.recordings[participant_name] = self._load_participant_recordings(participant_folder)
    
    def _load_participant_recordings(self, participant_folder: Path) -> dict:
        """
        Load all .pkl files for a single participant and organize by gesture.
        
        Returns:
            {
                'gesture_0': [recording_dicts],
                'gesture_1': [recording_dicts],
                ...
            }
        """
        recordings_by_gesture = {}
        
        # Find all .pkl files
        pkl_files = sorted(participant_folder.glob("*.pkl"))
        
        for pkl_file in pkl_files:
            try:
                recording_dict = self._load_and_analyze_recording(pkl_file)
                
                # Determine active gesture(s)
                task = recording_dict['task']

                if task in TASK_LIST:
                    if task == "Power Grasp":
                        task = "Grasp"
                    elif task == "Thumb":
                        task = "Peace"
                    elif task == "Pinky":
                        task = "Cellphone"

                    recordings_by_gesture.setdefault(task, []).append(recording_dict)
            
            except Exception as e:
                print(f"Error loading {pkl_file.name}: {e}")
        
        return recordings_by_gesture
    
    def _load_and_analyze_recording(self, pkl_file: Path) -> dict:
        """
        Load a .pkl file and extract relevant information.
        
        Returns:
            {
                'filename': str,
                'biosignal': np.ndarray (channels, samples_per_frame, frames),
                'biosignal_timings': np.ndarray,
                'ground_truth': np.ndarray (9, label_frames),
                'ground_truth_timings': np.ndarray,
                'active_gesture_idx': int or None,
                'device_info': dict,
                'recording_time': int,
            }
        """
        import numpy as np
        
        with open(pkl_file, 'rb') as f:
            data = pkl.load(f)
        
        # Determine which gesture channel(s) are active
        ground_truth = data['ground_truth']
        active_gesture_idx = self._find_active_gesture(ground_truth)
        
        return {
            'filename': pkl_file.name,
            'biosignal': data['biosignal'],
            'biosignal_timings': data['biosignal_timings'],
            'ground_truth': data['ground_truth'],
            'ground_truth_timings': data['ground_truth_timings'],
            'active_gesture_idx': active_gesture_idx,
            'device_info': data.get('device_information', {}),
            'recording_time': data.get('recording_time', None),
            'task': data.get('task', None),
        }
    
    def _find_active_gesture(self, ground_truth):
        """
        Determine which gesture channel (0-8) is primarily active.
        
        Returns:
            int: index of the active gesture (0-8), or None if all zeros
        """
        import numpy as np
        
        # Count non-zero activity per channel
        activity_per_channel = np.sum(np.abs(ground_truth) > 1e-6, axis=1)
        
        if np.all(activity_per_channel == 0):
            return None  # Rest or no gesture
        
        # Return the index of the most active channel
        return int(np.argmax(activity_per_channel))
    
    def get_data(self):
        """Return the organized recordings dictionary."""
        return self.recordings
    
    def summary(self):
        """Print a summary of loaded recordings."""
        print("Dataset Summary:")
        print("=" * 60)
        for participant, gestures in self.recordings.items():
            print(f"\n{participant}:")
            for gesture, recordings in gestures.items():
                print(f"  {gesture}: {len(recordings)} recordings")
        



if __name__ == "__main__":
    path_to_recordings_base = r"C:\Users\leonv\AIBE_LAB\aibe_ins_lab_recordings"
    
    # Option 1: Unzip and split raw data (RUN ONCE WHEN YOU HAVE JUST CLONED THE REPO!!!!!!)
    get_data_split(path_to_recordings_base)
    





    # Option 2: Load organized recordings (probably irrelevant now, just example)
    #path_to_raw = Path(path_to_recordings_base) / "raw"
    #ds = dataset(path_to_raw)
    #ds.summary()
    
    # Access data
    #data = ds.get_data()
    # data['leon']['gesture_0'] -> list of recording dicts
