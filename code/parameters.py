epochs_try = [100] #callback
learning_rate_try = [0.001, 0.0005, 0.00001]
hidden_sizes_try = [[32, 64, 160, 256],[64, 128, 320, 512]]
depths_try = [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 18, 3], [3, 8, 27, 3], [3, 6, 40, 3]]
decoder_hidden_size_try = [256, 768]
num_encoder_blocks_try = [1, 2, 4]

dropout_rate_try = [0.10]
batchsize_try = [128]


# epochs_try = [1] #callback
# learning_rate_try = [0.001]
# dropout_rate_try = [0]
# batchsize_try = [100]