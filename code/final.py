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



metric = evaluate.load("mean_iou")

config = SegformerConfig(
    num_channels = 12,
    image_size = 64
    
    )
model = TFSegformerForSemanticSegmentation(
    config
)


model_checkpoint = "nvidia/mit-b0"  # pre-trained model from which to fine-tune



side_length = 64


dataset = utils.get_dataset(
    constants.file_pattern,
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
    constants.file_pattern_test,
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
    constants.file_pattern_evaluate,
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
    
    print(pred_labels.shape)
    print(labels.shape)
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=1,
        ignore_index=-1
    )
    # add per category metrics as individual key-value pairs
    mean_accuracy = metrics.pop("mean_accuracy").tolist()
    mean_iou = metrics.pop("mean_iou").tolist()

    metrics.update(
        {f"accuracy": v for i, v in enumerate(mean_accuracy)}
    )
    metrics.update({f"iou": v for i, v in enumerate(mean_iou)})
    return {"val_" + k: v for k, v in metrics.items()}





metric_callback = KerasMetricCallback(
    metric_fn=compute_metrics,
    eval_dataset=dataset_test,
    batch_size=128,
    label_cols=["labels"],
)

print(dataset_test.element_spec)

#print(model.config)
opt = keras.optimizers.Adam(learning_rate = 0.0001)

model.compile(optimizer=opt)

callbacks = [metric_callback]
#callbacks = [metric_callback, WandbMetricsLogger(log_freq="epoch")]


model.fit(
    dataset,
    validation_data=dataset_test,
    callbacks=callbacks,
    epochs=2,
)
eval_loss = model.evaluate(dataset_evaluate)