# %% imports
import parselmouth
import librosa

# %%
test_file = '../tests/data/actor_audio/321-Describe_Pic_2.wav'

# %% compute intensity
sound_pat = parselmouth.Sound(test_file)
intensity = sound_pat.to_intensity(time_step=.001)
intensity_center_times = [intensity.get_time_from_frame_number(i+1) for i in range(intensity.n_frames)]
print(len(intensity.t_bins()[:, 0]))
print(len(intensity.values[0]))

# %% compute pitch
sound_pat = parselmouth.Sound(test_file)
pitch = sound_pat.to_pitch(time_step=.001)
pitch_values = pitch.selected_array['frequency']
pitch_center_times = [pitch.get_time_from_frame_number(i+1) for i in range(pitch.n_frames)]
print(pitch_center_times[1])
print(len(list(pitch_values)))

# %% compute formants
sound_pat = parselmouth.Sound(test_file)
formants = sound_pat.to_formant_burg(time_step=.001)
start = formants.t_bins()[:, 0]
end = formants.t_bins()[:, 1]

def build_formant(formants, formant_number):
    return [formants.get_value_at_time(formant_number, formants.get_time_from_frame_number(1+_)) for _ in range(formants.n_frames)]

f1 = build_formant(formants, 1)

# %% harmonicity
sound_pat = parselmouth.Sound(test_file)
harmonicity = sound_pat.to_harmonicity_ac(time_step=.001)

# %% mfcc
sound_pat = parselmouth.Sound(test_file)
mfcc = sound_pat.to_mfcc(time_step=.001, number_of_coefficients=12)

# %% jitter
sound_pat = parselmouth.Sound(test_file)
pointProcess = parselmouth.praat.call(sound_pat, "To PointProcess (periodic, cc)", 80, 500)
jitter = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

# %%
aud_dur = librosa.get_duration(path=test_file)
print(aud_dur)
# %%
