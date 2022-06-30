import torch
import torchaudio
from pathlib import Path
import torchaudio.transforms as T
from torch.utils.data import Dataset


class EcoData(Dataset):

    def __init__(self, path: list, length: int = 60, lwin: int = 12, ext: str = "WAV", n_fft: int = 1024):

        self.path = path
        self.length = length
        self.n_fft = n_fft
        self.lwin = lwin
        self.files = []

        for p in path:
            print(p)
            folder_files = list(Path(p).rglob("*.{}".format(ext)))
            print(len(folder_files))
            self.files += folder_files

    def __getitem__(self, index):
        file_ = self.files[index]
        x, sr = torchaudio.load(file_)
        resampling = 22050
        if x.shape[0] == 2:
          x = torch.mean(x, dim=0, keepdim=True)
        audio_len = self.length * resampling
        xr = T.Resample(sr, resampling)(x)
        xr = xr[:, 0:audio_len]

        win = self.lwin*resampling
        xw = torch.reshape(xr,(audio_len//win,win))
        Sxx = T.Spectrogram(      
            n_fft=self.n_fft,
            win_length=None,
            hop_length=None,
            center=True,
            pad_mode="reflect",
            power=2.0)(xw)
        return Sxx, xr, resampling

    def __len__(self):
        return len(self.files)