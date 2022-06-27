import torch
import argparse
from pathlib import Path
from src.dataset import EcoData
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.models import VQVAE
import torch.optim as optim
from six.moves import xrange
import numpy as np
from src.utils import testModel



class VQVAETrainer:
    def __init__(self, folder_path,length=60,lwin=12,ext='WAV',n_fft=1024, device):
        self.folder_path = folder_path
        self.dataset = EcoData(folder_path, length=length, 
                               lwin=lwin, ext=ext, n_fft=n_fft)
        self.partitions = [round(len(self.dataset)*0.9), 
                           len(self.dataset)-round(len(self.dataset)*0.9)]

        self.train, self.test = random_split(self.dataset,self.partitions,
                                generator=torch.Generator().manual_seed(1024))

        self.num_training_updates = len(self.train)
        self.device = device

    def run(self, checkpoints, batch_size, num_hiddens, 
            num_embeddings, embedding_dim, commitment_cost, 
            decay, learning_rate, num_epochs, num_training_updates ):

        train_dataloader = DataLoader(self.train, batch_size=batch_size, shuffle = False)
        test_dataloader  = DataLoader(self.test, batch_size=batch_size)

        model = VQVAE(num_hiddens, num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)


        model.train()
        iterator = iter(self.test)

        for epoch in range(num_epochs):
            for i in xrange(num_training_updates):
                try:
                    (data, _,_) = next(iter(train_dataloader))
                except Exception as e:
                    print(e)
                    continue
                
                for w in range(data.shape[1]):
                    tdata = data[:,w,:,:]  
                    tdata = torch.unsqueeze(tdata,1)
                    tdata = tdata.to(self.device)

                    optimizer.zero_grad()
                    vq_loss, data_recon, perplexity = model(tdata)
      
                    recon_error = F.mse_loss(data_recon, tdata)
                    loss = recon_error + vq_loss
                    loss.backward()

                    optimizer.step()


                print(f'epoch: {epoch} of {num_epochs} \t iteration: {(i+1)}... of {num_training_updates} \t loss: {np.round(loss.item(),7)} \t recon_error: {np.round(recon_error.item(),7)} \t vq_loss: {np.round(vq_loss.item(),7)}')

                torch.cuda.empty_cache()


                if (i+1) % 10 == 0:
                    fig, test_error = testModel(model, iterator)
                    torch.save(model.state_dict(),f'./{checkpoints}/model.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
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
    parser.add_argument('device', type=str)
    args = parser.parse_args()

    if not args.device:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vqvae_trainer = VQVAETrainer(args.folder_path,
                                args.audio_len,
                                args.lwin,
                                args.ext,
                                args.n_fft, 
                                args.device)
    vqvae_trainer.run(args.checkpoints_path, 
                     args.batch_size, 
                     args.num_hiddens, 
                     args.num_embeddings, 
                     args.embedding_dim, 
                     args.commitment_cost, 
                     args.decay, 
                     args.learning_rate, 
                     args.num_epochs, 
                     args.num_training_updates)


if __name__ == '__main__':
    main()
