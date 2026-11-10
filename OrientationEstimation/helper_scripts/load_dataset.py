import os
from pathlib import Path
import h5py
import numpy as np
import time

path = 'bleb'

def load_dataset(path):
    with h5py.File(path, 'r') as file:
        # -=- binaural signal -=- #
        # we can split this into train/validation/test beforehand if needed, or we would need to determine indices beforehand
        signals = np.asarray(
            file['signals'],
            dtype=np.float32
        )

        # -=- orientation -=- # probably need separate for listener and speakers
        angles_deg = np.asarray(
            file['orientation_deg'],
            dtype=np.float32
        ).reshape(-1)

        angles_rad = np.asarray(
            file['orientation_rad'],
            dtype=np.float32
        ).reshape(-1)
        angles = (angles_deg, angles_rad)

        # -=- labels -=- #
        listeners = np.asarray(
            file['listener_name'],
            dtype=np.str_
        )
        talkers = np.asarray(
            file['talker_name'],
            dtype=np.str_
        )
        class_index = np.asarray( # (classification)
            file['class_index'],
            dtype=np.uint16
        )
        class_center_deg = np.asarray( # (classification)
            file['class_center_deg'],
            dtype=np.float32
        )
        labels = (listeners, talkers, class_index, class_center_deg)

        # -=- meta data -=-
        ear_spacing_m = np.asarray(
            file['ear_spacing_m'],
            dtype=np.float32
        )
        signal_duration_s = np.asarray(
            file['signal_duration_s'],
            dtype=np.float32
        )
        meta_data = (ear_spacing_m, signal_duration_s)

        meta_dict = {
            'sampling_rate_hz': int(file['sampling_rate_hz']),
            'sound_speed_m_s': int(file['sound_speed_m_s']),
            'num_examples': int(file['num_examples']),
            'num_classes': int(file['num_classes']),
            'label_mode': str(file['label_mode']),
            'source_type': str(file['source_type']),
            'acoustic_condition': str(file['acoustic_condition']),
            'source_distance_m': float(file['source_distance_m']),
            'source_elevation_deg': float(file['source_elevation_deg']),

            # -=- extra -=- #
            'orientation_convention': str(file['orientation_convention']),
            'class_index_convention': str(file['class_index_convention']),
        }

    return signals, angles, labels, meta_data, meta_dict
