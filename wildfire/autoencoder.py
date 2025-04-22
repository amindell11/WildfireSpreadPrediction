from tensorflow import keras
from keras.layers import Conv2D, Input, Dropout, Add, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model

import wandb
from wandb.keras import WandbMetricsLogger

from . import constants
from . import utils
from . import parameters

def build_autoencoder(input_shape, dropout_rate):
    input_img = Input(shape=input_shape)

    # First skip connection
    skip_conv2D_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    skip_dropout_1 = Dropout(dropout_rate)(skip_conv2D_1)
    enc_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    enc_dropout1 = Dropout(dropout_rate)(enc_conv1)
    enc_merge_1 = Add()([enc_dropout1, skip_dropout_1])

    # ResBlock 1
    skip = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(enc_merge_1)
    skip = Dropout(dropout_rate)(skip)
    x = LeakyReLU()(enc_merge_1)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([x, skip])

    # ResBlock 2
    skip = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    skip = Dropout(dropout_rate)(skip)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([x, skip])

    # Upsample and ResBlock 3
    x = UpSampling2D((2, 2))(x)
    skip = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    skip = Dropout(dropout_rate)(skip)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([x, skip])

    # Upsample and ResBlock 4
    x = UpSampling2D((2, 2))(x)
    skip = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    skip = Dropout(dropout_rate)(skip)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([x, skip])

    output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, output)

def load_datasets(config, side_length=64):
    kwargs = {
        "data_size": 64,
        "sample_size": side_length,
        "batch_size": config.batch_size,
        "num_in_channels": 12,
        "compression_type": None,
        "clip_and_normalize": True,
        "clip_and_rescale": False,
        "random_crop": False,
        "center_crop": False,
        "transformer_shape": False
    }

    dataset = utils.get_dataset(constants.cloud_file_pattern, **kwargs)
    dataset_test = utils.get_dataset(constants.cloud_file_pattern_test, **kwargs)
    dataset_eval = utils.get_dataset(constants.cloud_file_pattern_evaluate, batch_size=100, **kwargs)
    return dataset, dataset_test, dataset_eval

def train_and_evaluate_model(model, dataset, dataset_test, dataset_eval, config):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='BinaryCrossentropy',
        metrics=[keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)]
    )

    model.fit(
        dataset,
        epochs=config.epochs,
        validation_data=dataset_test,
        callbacks=[WandbMetricsLogger(log_freq="epoch")]
    )

    results = model.evaluate(dataset_eval)
    return results

def hyper_param_sweep():
    for epoch in parameters.epochs_try:
        for lr in parameters.learning_rate_try:
            for dr in parameters.dropout_rate_try:
                for batchsize in parameters.batchsize_try:
                    experiment = wandb.init(
                        project="WildfirePropagation23-24",
                        config={
                            "learning_rate": lr,
                            "architecture": "Convolutional Autoencoder",
                            "dataset": "NDWS",
                            "optimizer": "adam",
                            "epochs": epoch,
                            "batch_size": batchsize,
                            "dropout_rate": dr
                        }
                    )
                    config = wandb.config
                    print(f"Training - LR: {lr}, Epochs: {epoch}, Batch Size: {batchsize}, Dropout: {dr}")

                    dataset, dataset_test, dataset_eval = load_datasets(config)
                    model = build_autoencoder(input_shape=(64, 64, 12), dropout_rate=config.dropout_rate)
                    train_and_evaluate_model(model, dataset, dataset_test, dataset_eval, config)

                    experiment.finish()
