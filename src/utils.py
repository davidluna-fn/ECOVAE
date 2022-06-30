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
  fig, ax = plt.subplots(nrows=2,ncols=5,figsize=(20,5))
  ax = ax.ravel()
    
  for i in range(grid.shape[0]):
    ax[i].imshow(grid[i,:,:].cpu().detach().numpy(),vmin=0,vmax=1)
    ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.subplots_adjust(wspace=-0.8, hspace=0.1)
  plt.show()

  return fig

def show_spectrogram(grid):
    fig, ax = plt.subplots(nrows=1,ncols=5,figsize=(15,5))
    for i in range(grid.shape[0]):
        try:
            ax[i].imshow(grid[i,:,:].cpu().numpy(),vmin=0,vmax=1)
            #ax[i].axes('off')
        except:
            ax[i].imshow(grid[i,:,:].cpu().detach().numpy(),vmin=0,vmax=1)
            #ax[i].axes('off')
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()

    return fig


def testModel(model, iterator):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    (valid, _,_) = next(iterator)
    
    valid = torch.unsqueeze(valid,1)
  
    valid = valid.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    #xx = torch.cat((valid[:,0,:,:],valid_reconstructions[:,0,:,:]),0)
    fig1 = show_spectrogram(valid[:,0,:,:])
    fig2 = show_spectrogram(valid_reconstructions[:,0,:,:])
    

    #mgrid = make_grid(xx, nrow=5, pad_value=20)
    #fig = showTest(mgrid)
    

    recon_error = F.mse_loss(valid, valid_reconstructions)

    return fig1, fig2, recon_error