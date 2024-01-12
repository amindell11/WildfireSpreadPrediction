from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import Input, Dropout, Add, LeakyReLU
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import evaluate


import constants
import utils
import parameters
import deeplab

from transformers import TFSegformerForSemanticSegmentation

for epoch in parameters.epochs_try:
    num_epochs = epoch
    for lr in parameters.learning_rate_try:
        learningrate = lr
        for dr in parameters.dropout_rate_try:
            dropoutrate = dr
            for batchs in parameters.batchsize_try:
                batchsize = batchs
                             
                # experiment = wandb.init(
                # project="Transformer",
                # config={
                #     "learning_rate": learningrate,
                #     "architecture": "Semantic Segformer",
                #     "dataset": "NDWS",
                #     "optimizer": "adam",
                #     "epochs": num_epochs,
                #     "batch_size": batchsize,
                #     "dropout_rate": dropoutrate
                #     }
                # )
                # print(f"Starting Training: \
                #         Learning Rate: {lr} \
                #         Epochs: {num_epochs} \
                #         Batch Size: {batchsize} \
                #         Dropout: {dropoutrate} \ ")    
                
                

                # config = wandb.config


                side_length = 64 #length of the side of the square you select (so, e.g. pick 64 if you don't want any random cropping)
                

                dataset = utils.get_dataset(
                    constants.file_pattern,
                    data_size=64,
                    sample_size=side_length,
                    batch_size=config.batch_size,
                    num_in_channels=12,
                    compression_type=None,
                    clip_and_normalize=True,
                    clip_and_rescale=False,
                    random_crop=False,
                    center_crop=False)

                
                dataset_test = utils.get_dataset(
                    constants.file_pattern_test,
                    data_size=64,
                    sample_size=side_length,
                    batch_size = config.batch_size,
                    num_in_channels=12,
                    compression_type=None,
                    clip_and_normalize = True,
                    clip_and_rescale=False,
                    random_crop=False,
                    center_crop=False
                )

                dataset_evaluate = utils.get_dataset(
                    constants.file_pattern_evaluate,
                    data_size=64,
                    sample_size=side_length,
                    batch_size = 100,
                    num_in_channels=12,
                    compression_type=None,
                    clip_and_normalize = True,
                    clip_and_rescale=False,
                    random_crop=False,
                    center_crop=False
                )                                

                metric = evaluate.load(keras.metrics.AUC(curve = 'PR'))
                
                dataset.element_spec
                
                
                
                
                # experiment.finish()            




