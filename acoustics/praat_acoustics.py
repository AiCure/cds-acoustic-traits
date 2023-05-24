# This code computes various vocal acoustic traits using the Praat library
# It does not require a model path as Parselmouth uses Praat under the hood and relies on 
# no machine learning models

from threading import Thread, Lock
from pathlib import Path
from collections import deque
import glob
import numpy as np
from aicurelib.util.video_io_util import reencode_audio_to_wav
import pandas as pd
import os
import parselmouth

class video_queue:
    def __init__(self, parent_dir, out_dir, dataset_name=None):
        if dataset_name == None:
            self.dataset_name = parent_dir.split('/')[1]
        else:
            self.dataset_name = dataset_name
        self.out_dir = out_dir
        # paths = glob.glob(f'{parent_dir}/*.mp4')
        paths = glob.glob(f'{parent_dir}/**/*.mp4', recursive=True) # added in case videos are located in subdirectory
        self.video_paths = deque(paths)
        # Fix paths for windows
        paths = [s.replace('\\', '/') for s in paths]
        self.video_ids = deque([''.join(filename.split('.')[:-1]) for filename in list(map(lambda x: x.split('/')[-1], paths))])
        for k in acoustics_map.keys():
            Path(f'{out_dir}/{self.dataset_name}/{k}').mkdir(parents=True, exist_ok=True)
        self.num_videos = len(self.video_ids)


def compute_formant_fn(formant_number):
    def f(path):
        return compute_formants(path, formant_number)
    return f


def compute_formants(path, formant_number=1):
    """
        Using parselmouth library fetching formants
        Args:
            path: (.wav) audio file location
        Returns:
            (list) list of formants for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    formants = sound_pat.to_formant_burg(time_step=.001)
    start = formants.t_bins()[:, 0]
    end = formants.t_bins()[:, 1]
    return extract_formant_fn(formants, formant_number), start, end


def extract_formant_fn(formants, formant_number):
    l = [formants.get_value_at_time(formant_number, formants.get_time_from_frame_number(1+_)) for _ in range(formants.n_frames)]
    return [np.NaN if _ in ['', 'nan'] else _ for _ in l]


def compute_intensity(path):
    """
        Using parselmouth library fetching Intensity
        Args:
            path: (.wav) audio file location
        Returns:
            (list) list of Intensity for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    intensity = sound_pat.to_intensity(time_step=.001)
    start = intensity.t_bins()[:, 0]
    end = intensity.t_bins()[:, 1]
    return intensity.values[0], start, end

def compute_hnr(path):
    """
        Using parselmouth library fetching Harmonicity
        Args:
            path: (.wav) audio file location
        Returns:
            (list) list of Harmonicity for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    harmonicity = sound_pat.to_harmonicity_ac(time_step=.001)
    start = harmonicity.t_bins()[:, 0]
    end = harmonicity.t_bins()[:, 1]
    values = np.where(harmonicity.values==-200, np.NaN, harmonicity.values).transpose()
    return values.reshape(values.shape[0],), start, end


def compute_pitch(path):
    """
    Using parselmouth library fetching fundamental frequency
    Args:
        path: (.wav) audio file location
    Returns:
        (list) list of fundamental frequency for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    pitch = sound_pat.to_pitch(time_step=.001)
    pitch_values = pitch.selected_array['frequency']
    start = pitch.t_bins()[:, 0]
    end = pitch.t_bins()[:, 1]
    return list(pitch_values), start, end


def generate_acoustic_dataframe(audio_path, f, fkey):
    try:
        data, fs, fe = f(audio_path)
        return pd.DataFrame({fkey : data,
                             f'{fkey}_frame_start' : fs,
                             f'{fkey}_frame_end' : fe,
                             'error_reason' : ['pass' for _ in range(len(data))]})
    except:
        return None
    
    
def compute_acoustic_traits(video_path, video_id=None, num_left=0, num_videos=1, thread_id=None):
    # TODO: Is there a way to store meta information from the model, such as step size
    audio_path = f'{video_path[:-4]}.wav'
    delete_audio_file = not os.path.isfile(audio_path)
    output = {}

    try:
        # extract audio from video 
        reencode_audio_to_wav(video_path, audio_path)
    except Exception:
        # failed to extract audio from video
        return output

    try:
        # generate acoustic traits
        for k,f in acoustics_map.items():
            output[k] = generate_acoustic_dataframe(audio_path, f, k)
    except Exception:
        # failed to generate acoustic traits
        pass
    
    try:
        if delete_audio_file:
            os.remove(audio_path)
    except Exception:
        pass

    return output


def process_videos_from_queue(q, lock, thread_id, output_dir, output_format='csv'):
    while len(q.video_ids) > 0:
        with lock:
            video_path = q.video_paths.popleft()
            video_id = q.video_ids.popleft()
            num_left = len(q.video_ids)
            num_videos = q.num_videos
        skip = False
        for k in acoustics_map.keys():
            if Path(f'{output_dir}/{q.dataset_name}/{k}/{video_id}.csv').is_file():
                skip = True
        if skip:
            continue
        # TODO update this to handle a dictionary of dataframes
        output = compute_acoustic_traits(video_path, video_id, num_left, num_videos, thread_id)
        for k,df in output.items():
            if df is not None:
                if output_format == 'parquet':
                    df.to_parquet(f'{output_dir}/{q.dataset_name}/{k}/{video_id}.parquet', index=False)
                else:
                    df.to_csv(f'{output_dir}/{q.dataset_name}/{k}/{video_id}.csv', index=False)


def process_directory(video_dir, output_dir, num_threads=1, output_format='csv'):
    lock = Lock()
    q = video_queue(video_dir, output_dir)
    
    threads = list()
    for thread_id in range(num_threads):
        thread = Thread(target=process_videos_from_queue, 
                        args=(q, lock, thread_id, output_dir, output_format))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

acoustics_map = {
    'intensity' : compute_intensity,
    'ff' : compute_pitch,
    'f1' : compute_formant_fn(1),
    'f2' : compute_formant_fn(2),
    'f3' : compute_formant_fn(3),
    'f4' : compute_formant_fn(4),
    'hnr' : compute_hnr
}