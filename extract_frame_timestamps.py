import h5py
import os
from utils.args import args
import numpy as np
import pandas as pd

def read_hdf5_rgb_timestamps(
    filepath
):
    """
    Read and return data emg from an HDF5 file
    """

    with h5py.File(filepath, 'r') as hdf:
        frame_timespamps = hdf['eye-tracking-video-world']['frame_timestamp']['time_s'][:,0]

    frame_times = pd.DataFrame(frame_timespamps, columns=['time_s'])
    frame_times['frame_name'] = np.arange(1,len(frame_times)+1)

    return frame_times

def save_data(
    data,
    filepath
):
    """
    Save data to a pickle file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok = True)
    data.to_pickle(filepath)


if __name__ == '__main__':
    source_filepath = args.source_filepath
    dest_filepath = args.dest_filepath

    rgb_df = read_hdf5_rgb_timestamps(source_filepath)
    save_data(rgb_df, dest_filepath)