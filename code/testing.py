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

metric = evaluate.load("arthurvqin/pr_auc")

#metric=keras.metrics.AUC(curve = 'PR')

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
    probabilities = tf.nn.softmax(logits_resized, axis=-1)
    
    # compute the prediction labels and compute the metric
    pred_labels = probabilities[:,:,:,1]
    
    pred_labels = tf.reshape(pred_labels, shape=(-1,))
    
    labels = tf.reshape(labels, shape=(-1,))
    #pred_labels = tf.argmax(probabilities, axis=-1)
    metrics = metric.compute(
        prediction_scores= pred_labels,
        references=labels,
    )
    # add per category metrics as individual key-value pairs
    per_category_auc = metrics.pop("pr_auc").tolist()

    metrics.update({f"auc_pr": v for i, v in enumerate(per_category_auc)})
    return {"val_" + k: v for k, v in metrics.items()}


metric_callback = KerasMetricCallback(
    metric_fn=compute_metrics,
    eval_dataset=dataset_evaluate,
    batch_size=128,
    label_cols=["labels"],
)


#print(model.config)
opt = keras.optimizers.Adam(learning_rate = 0.001)

model.compile(optimizer=opt)
                
callbacks = [metric_callback] 

model.fit(dataset_evaluate, epochs=2, validation_data=dataset_test, callbacks=callbacks, shuffle=True, verbose=True)


