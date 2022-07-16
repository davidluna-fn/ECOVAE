import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def get_spectrogram(waveform, n_fft = 1024, win_len = None, hop_len = None, power = 2.0):
  spectrogram = T.Spectrogram(
      n_fft=n_fft,
      win_length=win_len,
      hop_length=hop_len,
      center=True,
      pad_mode="reflect",
      power=power,
  )
  return spectrogram(waveform)

def showTest(grid):
  fig, ax = plt.subplots(nrows=2,ncols=4,figsize=(20,5))
  ax = ax.ravel()
    
  for i in range(grid.shape[0]):
    ax[i].imshow(grid[i,:,:].cpu().detach().numpy(),vmin=0,vmax=1)
    ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.subplots_adjust(wspace=-0.8, hspace=0.1)
  plt.show()

  return fig

def showgrid(t1,t2):
    fig, ax = plt.subplots(nrows=2,ncols=5,figsize=(20,7))
    for i in range(t1.shape[0]):
        ax[0,i].pcolormesh(t1[i,:,:].cpu().numpy(),vmin=0,vmax=1)
        ax[1,i].pcolormesh(t2[i,:,:].cpu().detach().numpy(),vmin=0,vmax=1)
        ax[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax[1,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig

def inverse_data(data, n_fft):
    ISxx = T.InverseSpectrogram(
    n_fft=n_fft,
    win_length=None,
    hop_length=None,
    center=True,
    pad_mode="reflect")(data.type(torch.complex64))


    return ISxx.reshape((ISxx.shape[0] * ISxx.shape[1] ))

def testModel(model, iterator):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    (valid, _,_,_) = next(iterator)
    
    valid = torch.unsqueeze(valid,1)
  
    valid = valid.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    fig = showgrid(valid[:,0,:,:],valid_reconstructions[:,0,:,:])
    recon_error = F.mse_loss(valid, valid_reconstructions)

    return fig, recon_error