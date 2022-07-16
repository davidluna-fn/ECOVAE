import torch
import argparse
import numpy as np
from models import VQVAE
from utils import testModel
from dataset import EcoData
import torch.optim as optim
from six.moves import xrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import wandb


class VQVAETrainer:
    def __init__(self, folder_path,length=60,lwin=12,ext='WAV',n_fft=1028, device='cuda', wandb=False,load_checkpoint = None):
        self.folder_path = folder_path
        self.dataset = EcoData(folder_path, length=length, 
                               lwin=lwin, ext=ext, n_fft=n_fft)
        self.partitions = [round(len(self.dataset)*0.9), 
                           len(self.dataset)-round(len(self.dataset)*0.9)]

        self.train, self.test = random_split(self.dataset,self.partitions,
                                generator=torch.Generator().manual_seed(1024))

        self.num_training_updates = len(self.train)
        self.device = device
        self.wandb = wandb
        self.load_checkpoint = load_checkpoint

    def run(self, checkpoints, batch_size, num_hiddens, 
            num_embeddings, embedding_dim, commitment_cost, 
            decay, learning_rate, num_epochs):

        print(f'run trained \t Device: {self.device}')

        train_dataloader = DataLoader(self.train, batch_size=batch_size, shuffle = True)
        test_dataloader  = DataLoader(self.test, batch_size=batch_size, shuffle = True)

        model = VQVAE(num_hiddens, num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(self.device)

        if self.load_checkpoint != None:
            file_checkpoint = torch.load(self.load_checkpoint)
            model.load_state_dict(file_checkpoint['state_dict'])


        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1 )

        if self.wandb:
            wandb.watch(model, F.mse_loss, log="all", log_freq=1)


        model.train()
        test_iter = iter(self.test)

        for epoch in range(num_epochs):
            train_iter = iter(train_dataloader)
            for i in xrange(len(train_iter)):
                try:
                    (data, _,_,_) = next(train_iter)
                except Exception as e:
                    print(e)
                    continue
                
                for w in range(data.shape[1]):
                    tdata = data[:,w,:,:]  
                    tdata = torch.unsqueeze(tdata,1)
                    tdata = tdata.to(self.device)

                    optimizer.zero_grad()
                    vq_loss, data_recon, perplexity = model(tdata)
      
                    recon_error = F.mse_loss(data_recon, tdata, reduction='mean')
                    loss = recon_error + vq_loss
                    loss.backward()

                    optimizer.step()
              
                if self.wandb:
                    wandb.log({"loss":loss.item(),
                               "perplexity":perplexity.item(),
                               "recon_error": recon_error,
                               "vq_loss": vq_loss})

                print(f'epoch: {epoch} of {num_epochs} \t iteration: {(i+1)}... of {len(train_iter)} \t loss: {np.round(loss.item(),7)} \t recon_error: {np.round(recon_error.item(),7)} \t vq_loss: {np.round(vq_loss.item(),7)}')

                torch.cuda.empty_cache()


                if (i+1) % 10 == 0:
                    try:
                        fig, test_error = testModel(model, test_iter)
                        if self.wandb:
                            images = wandb.Image(fig, caption= f"recon_error: {np.round(test_error.item(),4)}")
                            wandb.log({"examples": images})
                            torch.save({'state_dict': model.state_dict(),
                                    'epoch': epoch,
                                    'iteration': i,
                                    'loss': np.round(loss.item(),7),
                                    'recon_error': np.round(recon_error.item(),7), 
                                    'vq_loss': np.round(vq_loss.item(),7)},
                                    f'{checkpoints}/{wandb.run.name}.pth')
                        else:
                            torch.save({'state_dict': model.state_dict(),
                                    'epoch': epoch,
                                    'iteration': i,
                                    'loss': np.round(loss.item(),7),
                                    'recon_error': np.round(recon_error.item(),7), 
                                    'vq_loss': np.round(vq_loss.item(),7)},
                                    f'{checkpoints}/ecovae_model.pth')

                        plt.close('all')

                        
                    except:
                        test_iter = iter(self.test)
                        fig, test_error = testModel(model, test_iter)
                        if self.wandb:
                            images = wandb.Image(fig, caption= f"recon_error: {np.round(test_error.item(),4)}")
                            wandb.log({"examples": images})
                            torch.save({'state_dict': model.state_dict(),
                                    'epoch': epoch,
                                    'iteration': i,
                                    'loss': np.round(loss.item(),7),
                                    'recon_error': np.round(recon_error.item(),7), 
                                    'vq_loss': np.round(vq_loss.item(),7)},
                                    f'{checkpoints}/{wandb.run.name}.pth')

                        else:
                            torch.save({'state_dict': model.state_dict(),
                                    'epoch': epoch,
                                    'iteration': i,
                                    'loss': np.round(loss.item(),7),
                                    'recon_error': np.round(recon_error.item(),7), 
                                    'vq_loss': np.round(vq_loss.item(),7)},
                                    f'{checkpoints}/ecovae_model.pth')

                        plt.close('all')

            scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', nargs='+', required=True)
    parser.add_argument('--audio_len', type=int, default=60)
    parser.add_argument('--lwin', type=int, default=12)
    parser.add_argument('--ext', type=str, default='WAV')
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--checkpoints_path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--num_hiddens', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_embeddings', type=int, default=64)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--wandb', type=bool, default=bool,required=False)
    parser.add_argument('--load_checkpoint', type=str, default=None,required=False)
    args = parser.parse_args()
    

    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.finish()
        wandb.init(project="ecovae", config=args)


    vqvae_trainer = VQVAETrainer(args.folder_path,
                                args.audio_len,
                                args.lwin,
                                args.ext,
                                args.n_fft, 
                                args.device,
                                args.wandb,
                                args.load_checkpoint)

    vqvae_trainer.run(args.checkpoints_path, 
                     args.batch_size, 
                     args.num_hiddens, 
                     args.num_embeddings, 
                     args.embedding_dim, 
                     args.commitment_cost, 
                     args.decay, 
                     args.learning_rate, 
                     args.num_epochs)

    if args.wandb:
        wandb.finish()


    print('End training')


if __name__ == '__main__':
    main()
