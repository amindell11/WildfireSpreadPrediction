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



experiment = wandb.init(
project="WildfireAblation",
config={
    "learning_rate": 0.0005,
    "architecture": "SegFormer",
    "dataset": "NDWS",
    "optimizer": "adam",
    "epochs": 25,
    "batch_size": 128
    }
)
print(f"Starting Training: \
        Learning Rate: 0.0005 \
        Epochs: 25 \
        Batch Size: 128 \ ")    



config = wandb.config

metric = evaluate.load("mean_iou")

configg = SegformerConfig(
    num_channels = 11
    image_size = 64,
    )
model = TFSegformerForSemanticSegmentation(
    configg
)




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
    
    
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=2,
        ignore_index=-1
    )
    
    return {"val_" + k: v for k, v in metrics.items()}




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