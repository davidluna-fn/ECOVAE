python train.py   \
    --folder_path '/content/drive/MyDrive/Datasets/Grabaciones_Camilo_2/Sitio1/AudioMoth' \
    --audio_len 60 \
    --lwin 12 \
    --ext 'WAV' \
    --n_fft 1024 \
    --checkpoints_path '/content/drive/MyDrive/UdeA/Codes/ECOVAE/checkpoints' \
    --batch-size 64 \
    --num_epochs 3 \
    --num_hiddens 64 \
    --embedding_dim 128 \
    --num_embeddings 64 \
    --commitment_cost 0.25 \
    --decay 0.99 \
    --learning_rate 1e-3 \