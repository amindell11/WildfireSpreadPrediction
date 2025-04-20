# Parameters shared by both Conv. Autoencoder and SegFormer
epochs_try = [100, 200] 
learning_rate_try = [0.0005, 0.00001]
dropout_rate_try = [0.10]
batchsize_try = [128]


# Parameters specific to SegFormer
hidden_sizes_try = [[32, 64, 160, 256],[64, 128, 320, 512]]
depths_try = [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 18, 3], [3, 6, 40, 3]]
decoder_hidden_size_try = [256]
num_encoder_blocks_try = [1]
