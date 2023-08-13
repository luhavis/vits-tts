import soundfile
from glob import glob
import librosa
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm


files = glob("datasets/*.wav")

for file in tqdm(files):
    sample_rate, data = wavfile.read(file)
    y, sr = librosa.load(file, sr=sample_rate)
    resample_data = librosa.resample(y=y, orig_sr=sr, target_sr=22050)
    soundfile.write(file, data=resample_data, samplerate=22050, format="wav", endian="LITTLE", subtype="PCM_16")
