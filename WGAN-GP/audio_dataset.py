import os
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import librosa
import scipy.io.wavfile as wavf
import scipy.signal

# Convert audio to spectrogram
n_fft = 2048  # FFT window size
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel frequency bands
win_length = 1024  # Window length
sr = 16000

class AudioDataset(Dataset):
	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.file_list = os.listdir(folder_path)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		file_name = self.file_list[index]
		file_path = os.path.join(self.folder_path, file_name)
		return self.wav2spec(file_path)

	def wav2spec(self, file_path, max_pad_len=20):
		# read data
		fs, data = wavf.read(file_path)
		# resample
		data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=sr, res_type="scipy")
		# zero padding
		if len(data) > sr:
			raise ValueError("data length cannot exceed padding length.")
		elif len(data) < sr:
			embedded_data = np.zeros(sr)
			offset = np.random.randint(low = 0, high = sr - len(data))
			embedded_data[offset:offset+len(data)] = data
		elif len(data) == sr:
			# nothing to do here
			embedded_data = data
			pass

		# Compute spectrogram using Short-Time Fourier Transform (STFT)
		D = librosa.stft(embedded_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
		spectrogram = np.abs(D)

		# Convert spectrogram to Mel scale
		mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
		mel_spectrogram = np.dot(mel_basis, spectrogram)

		# Apply logarithm to the spectrogram
		log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
		return torch.tensor(log_mel_spectrogram).float()

def invert_spectrogram(log_mel_spectrogram):
	mel_spectrogram = librosa.db_to_amplitude(log_mel_spectrogram)
	mel_spectrogram = np.array(mel_spectrogram, dtype=np.float32)

	S = librosa.feature.inverse.mel_to_stft(mel_spectrogram)
	y = librosa.griffinlim(S)

	# Normalize the audio waveform
	y /= np.max(np.abs(y))
	return y