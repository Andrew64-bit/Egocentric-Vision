import numpy as np
import pickle
import h5py
import sys
import os
from utils.args import args
import pandas as pd
from scipy.signal import butter, filtfilt

def rectify_emg(emg_data):
    """
    Rectify EMG data by taking the absolute value.
    """
    return np.abs(emg_data)

def low_pass_filter(emg_data, cutoff=5, fs=200):
    """
    Apply a low-pass filter to the EMG data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_emg = filtfilt(b, a, emg_data, axis=0)
    return filtered_emg

def normalize_emg(emg_data):
    """
    Normalize EMG data to the range [-1, 1].
    """
    emg = np.array([np.array(x) for x in emg_data])
    min_val = np.min(emg, axis=0)
    max_val = np.max(emg, axis=0)
    normalized_emg = 2 * ((emg - min_val) / (max_val - min_val)) - 1
    return normalized_emg

def process_emg_data(emg_df,emg):
    """
    Process EMG data as described.
    """
    emg_df1 = emg_df[emg].values
    emg_rectified = rectify_emg(emg_df1)


    times = emg_df['time_s']
    fs = len(emg_df) / (times[len(emg_df) - 1] - times[0])
    # Apply low-pass filter
    emg_filtered = low_pass_filter(emg_rectified, fs=fs)
    # Normalize data
    emg_normalized = normalize_emg(emg_filtered)
    
    return emg_normalized








def read_hdf5_emg(
    filepath
):
    """
    Read and return data emg from an HDF5 file
    """

    with h5py.File(filepath, 'r') as hdf:

        emg_left_data = hdf['myo-left']['emg']['data'][:]
        emg_left_time = hdf['myo-left']['emg']['time_s'][:]
        
        emg_right_data = hdf['myo-right']['emg']['data'][:]
        emg_right_time = hdf['myo-right']['emg']['time_s'][:]

    df_left = pd.DataFrame(emg_left_data, columns=[f'emg_left_ch{i+1}' for i in range(emg_left_data.shape[1])])
    df_left['time_s'] = emg_left_time
    df_left['left_emg'] = df_left[[f'emg_left_ch{i+1}' for i in range(emg_left_data.shape[1])]].values.tolist()
    df_left = df_left[['left_emg','time_s']]
    df_left = df_left.sort_values('time_s')
    df_left['left_emg'] = df_left['left_emg'].apply(np.array)

    emg_processed = process_emg_data(df_left,'left_emg')
    emg_processed = emg_processed.tolist()
    df_left['left_emg'] = emg_processed


    df_right = pd.DataFrame(emg_right_data, columns=[f'emg_right_ch{i+1}' for i in range(emg_right_data.shape[1])])
    df_right['time_s'] = emg_right_time
    df_right['right_emg'] = df_right[[f'emg_right_ch{i+1}' for i in range(emg_right_data.shape[1])]].values.tolist()
    df_right = df_right[['right_emg','time_s']]
    df_right = df_right.sort_values('time_s')
    df_right['right_emg'] = df_right['right_emg'].apply(np.array)

    emg_processed = process_emg_data(df_right,'right_emg')
    emg_processed = emg_processed.tolist()
    df_right['right_emg'] = emg_processed


    result_df = pd.merge_asof(df_left.sort_values('time_s'), df_right.sort_values('time_s'), on='time_s', direction='nearest')

    return result_df

def read_activity_data(
    filepath,
    device_name,
    stream_name
):
    """
    Extract activity data including labels, timestamps, and metadata.
    """

    with h5py.File(filepath, 'r') as h5_file:

        activities = [[x.decode('utf-8') for x in row] for row in h5_file[device_name][stream_name]['data']]
        times = np.squeeze(np.array(h5_file[device_name][stream_name]['time_s']))
    
    return activities, times

def process_activities(
    activities,
    times,
    exclude_bad = True
):
    """
    Process activity data to combine start/stop entries and exclude bad labels if specified.
    """

    processed_activities = []

    for i, (label, start_stop, validity, notes) in enumerate(activities):

        if exclude_bad and validity in ['Bad', 'Maybe']:
            continue

        if start_stop == 'Start':

            processed_activities.append({
                'description': label,
                'start_time': times[i]
            })
        
        elif start_stop == 'Stop' and processed_activities:

            processed_activities[-1]['end_time'] = times[i]

    df = pd.DataFrame(processed_activities)
    return df


def merge_emg_activities(
    emg_df,
    activities
):
    """
    Segment EMG data based on activity start and end times.
    """
    activities['emg_data'] = None  # Inizializza la colonna 'emg_data'
    for index, row in activities.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        
        filtered_emg = emg_df[(emg_df['time_s'] >= start_time) & (emg_df['time_s'] <= end_time)]
        
        activities.at[index, 'emg_data'] = filtered_emg[['time_s', 'left_emg', 'right_emg']].to_dict('records')
    
    return activities

def resample_emg(emg, freq = 10):
    size = len(emg)
    duration = emg['time_s'][size-1] - emg['time_s'][0]
    n_samples = int(duration * freq)

    idx = np.linspace(0, size-1, n_samples)
    idx = idx.astype(int)

    df_downsample = emg.iloc[idx]
    return df_downsample

def augment_data(emg_df, subactions_sec = 10):
    
    df_copy = emg_df.copy()
    for i in range(len(emg_df)):

        segments = []

        data = emg_df.loc[i, 'emg_data']
        dim = data[-1]['time_s'] - data[0]['time_s']
        num_segments = int(dim / subactions_sec)
        if (num_segments >= 1 and num_segments <= 4) and ((dim % subactions_sec) > 0 and (dim % subactions_sec) < (subactions_sec - (subactions_sec/10 * (num_segments + 1)))):
            dim_segments = dim / num_segments
            start_points = [(data[0]['time_s'] + (i*dim_segments)) for i in range(num_segments)]
            size = dim_segments

        else:
            num_segments = int(np.ceil(dim / subactions_sec))
            start_points = np.linspace(data[0]['time_s'],data[-1]['time_s'] - subactions_sec,num_segments)
            size = subactions_sec

        for s in start_points:
            start = s
            end = s + size
            filtered_emg = [entry for entry in data if entry['time_s'] >= start and entry['time_s'] <= end]
            segments.append(filtered_emg)

        df_copy.at[i, 'emg_data'] = segments
    return df_copy











activities_to_classify = {
        'Get/replace items from refrigerator/cabinets/drawers': 0,
        'Peel a cucumber': 1,
        'Clear cutting board': 2,
        'Slice a cucumber': 3,
        'Peel a potato': 4,
        'Slice a potato': 5,
        'Slice bread': 6,
        'Spread almond butter on a bread slice': 7,
        'Spread jelly on a bread slice': 8,
        'Open/close a jar of almond butter': 9,
        'Pour water from a pitcher into a glass': 10,
        'Clean a plate with a sponge': 11,
        'Clean a plate with a towel': 12,
        'Clean a pan with a sponge': 13,
        'Clean a pan with a towel': 14,
        'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
        'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
        'Stack on table: 3 each large/small plates, bowls': 17,
        'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
        'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}


def label_data(emg):
    activities_renamed = {
    'Open a jar of almond butter': 'Open/close a jar of almond butter',
    'Get items from refrigerator/cabinets/drawers': 'Get/replace items from refrigerator/cabinets/drawers',
    }

    emg['description'] = emg['description'].map(lambda x: activities_renamed[x] if x in activities_renamed else x)
    emg['description'] = emg['description'].map(lambda x: activities_renamed[x] if x in activities_renamed else x)
    
    emg['description_class'] = emg['description'].map(activities_to_classify).astype(int)
    emg['description_class'] = emg['description'].map(activities_to_classify).astype(int) 
    return emg

def save_data(
    data,
    filepath
):
    """
    Save data to a pickle file.
    """

    os.makedirs(os.path.dirname(filepath), exist_ok = True)

    data.to_pickle(filepath)

def main():
    source_filepath = args.source_filepath
    dest_filepath = args.dest_filepath
    
    # Left & right
    emg_df = read_hdf5_emg(source_filepath)
    emg_df = resample_emg(emg_df)

    activities, activities_times = read_activity_data(source_filepath, 'experiment-activities', 'activities')
    processed_activities = process_activities(activities, activities_times)
    merge_df = merge_emg_activities(emg_df, processed_activities)
    segments_df = augment_data(merge_df)
    final_df = label_data(segments_df)

    save_data(final_df, dest_filepath)
    #print(f'data successfully saved to {dest_filepath}')

if __name__ == '__main__':
    main()