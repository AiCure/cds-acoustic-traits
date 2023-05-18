# %% imports
import parselmouth
import librosa

# %%
test_file = '../tests/data/actor_audio/321-Describe_Pic_2.wav'

# %% compute intensity
sound_pat = parselmouth.Sound(test_file)
intensity = sound_pat.to_intensity(time_step=.001)
intensity_center_times = [intensity.get_time_from_frame_number(i+1) for i in range(intensity.n_frames)]
print(intensity_center_times[1])
print(len(intensity.values[0]))

# %% compute pitch
sound_pat = parselmouth.Sound(test_file)
pitch = sound_pat.to_pitch(time_step=.001)
pitch_values = pitch.selected_array['frequency']
pitch_center_times = [pitch.get_time_from_frame_number(i+1) for i in range(pitch.n_frames)]
print(pitch_center_times[1])
print(len(list(pitch_values)))

# %%
aud_dur = librosa.get_duration(path=test_file)
print(aud_dur)
# %%
