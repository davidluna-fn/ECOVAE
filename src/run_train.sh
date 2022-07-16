!python train.py   \
    --folder_path '/content/drive/MyDrive/Datasets/Grabaciones_Camilo_2/Sitio1/AudioMoth' \
                  '/content/drive/MyDrive/Datasets/Grabaciones_Camilo_2/Sitio2/AudioMoth' \
                  '/content/drive/MyDrive/UdeA/Datasets/Grabaciones_Camilo_2/Sitio1/a/Data' \
                  '/content/drive/MyDrive/UdeA/Datasets/Grabaciones_Camilo_2/Sitio1/b/Data' \
                  '/content/drive/MyDrive/UdeA/Datasets/GrabacionesCamilo/G21A' \
                  '/content/drive/MyDrive/UdeA/Datasets/GrabacionesCamilo/G21S' \
    --audio_len 60 \
    --lwin 12 \
    --ext 'WAV' \
    --n_fft 1028 \
    --checkpoints_path '/content/drive/MyDrive/UdeA/Codes/ECOVAE/checkpoints' \
    --batch-size 16 \
    --num_epochs 10 \
    --num_hiddens 64 \
    --embedding_dim 512 \
    --num_embeddings 256 \
    --commitment_cost 0.25 \
    --decay 0.99 \
    --learning_rate 1e-2 \
    --wandb True \