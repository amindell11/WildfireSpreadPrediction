from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import Input, Dropout, Add, LeakyReLU
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import evaluate
import tensorflow as tf

import constants
import utils
import parameters
import deeplab

from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

from transformers.keras_callbacks import KerasMetricCallback
from sklearn.metrics import average_precision_score
from pdb import set_trace

#metric=evaluate.load("arthurvqin/pr_auc")

for epoch in parameters.epochs_try:
    num_epochs = epoch
    for lr in parameters.learning_rate_try:
        learningrate = lr
        for hiddens in parameters.hidden_sizes_try:
            hiddensize = hiddens
            for depth in parameters.depths_try:
                depths = depth
                for decoderhiddens in parameters.decoder_hidden_size_try:
                    decoderhiddensize = decoderhiddens
                    for numenc in parameters.num_encoder_blocks_try:
                        numencoderblocks = numenc
                        
                        experiment = wandb.init(
                        project="WildfirePropagation23-24v2",
                        config={
                            "learning_rate": learningrate,
                            "architecture": "SegFormer",
                            "dataset": "NDWS",
                            "optimizer": "adam",
                            "epochs": num_epochs,
                            "batch_size": 128,
                            "hidden_size": hiddensize,
                            "depth": depths,
                            "decoder_hidden_size": decoderhiddensize,
                            "num_encoder_blocks": numencoderblocks
                            }
                        )
                        print(f"Starting Training: \
                                Learning Rate: {lr} \
                                Epochs: {num_epochs} \
                                Batch Size: 128 \
                                Hidden Size: {hiddensize} \
                                Depth: {depths} \
                                Decoder Hidden Size: {decoderhiddensize} \
                                Number of Encoder Blocks: {numencoderblocks} \ ")    
                        
                        

                        config = wandb.config

                        metric = evaluate.load("mean_iou")
                        m = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
                        configg = SegformerConfig(
                            num_channels = 12,
                            image_size = 64,
                            num_encoder_blocks=config.num_encoder_blocks,
                            depths=config.depth,
                            hidden_sizes=config.hidden_size,
                            decoder_hidden_size=config.decoder_hidden_size
                            )
                        model = TFSegformerForSemanticSegmentation(
                            configg
                        )


                        model_checkpoint = "nvidia/mit-b0"  # pre-trained model from which to fine-tune



                        side_length = 64


                        dataset = utils.get_dataset(
                            constants.cloud_file_pattern,
                            data_size=64,
                            sample_size=side_length,
                            batch_size=128,
                            num_in_channels=12,
                            compression_type=None,
                            clip_and_normalize=True,
                            clip_and_rescale=False,
                            random_crop=False,
                            center_crop=False,
                            transformer_shape=True
                            )


                        dataset_test = utils.get_dataset(
                            constants.cloud_file_pattern_test,
                            data_size=64,
                            sample_size=side_length,
                            batch_size = 128,
                            num_in_channels=12,
                            compression_type=None,
                            clip_and_normalize = True,
                            clip_and_rescale=False,
                            random_crop=False,
                            center_crop=False,
                            transformer_shape=True
                        )

                        dataset_evaluate = utils.get_dataset(
                            constants.cloud_file_pattern_evaluate,
                            data_size=64,
                            sample_size=side_length,
                            batch_size = 100,
                            num_in_channels=12,
                            compression_type=None,
                            clip_and_normalize = True,
                            clip_and_rescale=False,
                            random_crop=False,
                            center_crop=False,
                            transformer_shape=True
                        )                                

                        def compute_metrics(eval_pred):
                            logits, labels = eval_pred
                            # logits are of shape (batch_size, num_labels, height, width), so
                            # we first transpose them to (batch_size, height, width, num_labels)
                            logits = tf.transpose(logits, perm=[0, 2, 3, 1])
                            # scale the logits to the size of the label
                            logits_resized = tf.image.resize(
                                logits,
                                size=tf.shape(labels)[1:],
                                method="bilinear",
                            )
                            # compute the prediction labels and compute the metric
                            pred_labels = tf.argmax(logits_resized, axis=-1)
                            
                            m.update_state(labels, pred_labels)
                            # metrics = metric.compute(
                            #     predictions=pred_labels,
                            #     references=labels,
                            #     num_labels=2,
                            #     ignore_index=-1
                            #)
                            return {"val_iou": m.result().numpy()}




                        metric_callback = KerasMetricCallback(
                            metric_fn=compute_metrics,
                            eval_dataset=dataset_test,
                            batch_size=128,
                            label_cols=["labels"],
                        )

                        print(dataset_test.element_spec)

                        #print(model.config)
                        opt = keras.optimizers.Adam(learning_rate = config.learning_rate)

                        model.compile(optimizer=opt)

                        callbacks = [metric_callback, WandbMetricsLogger(log_freq="epoch")]

                        model.fit(
                            dataset,
                            validation_data=dataset_test,
                            callbacks=callbacks,
                            epochs=config.epochs,
                        )
                        results = model.evaluate(dataset_evaluate)
                        
                        experiment.finish()   