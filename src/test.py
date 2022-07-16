import torch
import argparse
from models import VQVAE
from utils import testModel
from dataset import EcoData
from six.moves import xrange
from torch.utils.data import DataLoader



class VQVAETest:
    def __init__(self, folder_path,length=60,lwin=12,ext='WAV',n_fft=1024, device='cuda'):
        self.folder_path = folder_path
        self.dataset = EcoData(folder_path, length=length, 
                               lwin=lwin, ext=ext, n_fft=n_fft)
        self.device = device

    def get_features(self,checkpoints,batch_size, num_hiddens, 
            num_embeddings, embedding_dim, commitment_cost, 
            decay, learning_rate):

        data_files = DataLoader(self.dataset, batch_size=batch_size, shuffle = False)

        model = VQVAE(num_hiddens, num_embeddings, embedding_dim, 
                      commitment_cost, decay).to(self.device)

        file_checkpoint = torch.load(checkpoints)
        model.load_state_dict(file_checkpoint['state_dict'])
        

        model.eval()
        iter_train = iter(data_files)
        feat = torch.empty((1, 900)).to('cpu')
        print(f'feat size1: {feat.shape}')

        for i in xrange(len(data_files)):
            (data, _,_, file_) = next(iter_train)
            features = torch.zeros((1, 900)).to('cpu')
            for w in range(data.shape[1]):
                tdata = data[:,w,:,:]  
                tdata = torch.unsqueeze(tdata,1)

                print(f'tdata size:  {tdata.shape}')
                tdata = tdata.to(self.device)

                vq_output_eval = model._pre_vq_conv(model._encoder(tdata))
                _, valid_quantize, _, encodings = model._vq_vae(vq_output_eval)


                features +=  valid_quantize.mean(dim=1).reshape(batch_size, 30*30).to('cpu')
                torch.cuda.empty_cache()
            
            features = features/5
            

            feat = torch.cat((feat,features),dim=0) 
            print(f'{i} \t features: {features.shape} \t name: {file_}')


        print(f'feat size: {feat.shape}')

        return feat 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', nargs='+', required=True)
    parser.add_argument('--audio_len', type=int, default=60)
    parser.add_argument('--lwin', type=int, default=12)
    parser.add_argument('--ext', type=str, default='WAV')
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--checkpoints_path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num_hiddens', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_embeddings', type=int, default=64)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, required=False)
    args = parser.parse_args()

    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae_test = VQVAETest(args.folder_path,
                                args.audio_len,
                                args.lwin,
                                args.ext,
                                args.n_fft, 
                                args.device)

    vqvae_test.get_features(args.checkpoints_path, 
                     args.batch_size, 
                     args.num_hiddens, 
                     args.num_embeddings, 
                     args.embedding_dim, 
                     args.commitment_cost, 
                     args.decay, 
                     args.learning_rate)

    print('Termino el for')


if __name__ == '__main__':
    main()
