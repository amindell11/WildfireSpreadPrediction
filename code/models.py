from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import Input, Dropout, Add, LeakyReLU
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model


def cnn_autoencoder(input_img, dropout_rate):
    #First skip connection
    skip_conv2D_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    skip_dropout_1 = Dropout(dropout_rate)(skip_conv2D_1)
    enc_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    enc_dropout1 = Dropout(dropout_rate)(enc_conv1)
    enc_merge_1 = Add()([enc_dropout1, skip_dropout_1])


    #First ResBlock(Maxpool)
    skip_resblock1_conv = Conv2D(32, (3, 3), strides = (2,2), activation = 'relu', padding = 'same')(enc_merge_1)
    skip_resblock1_dropout = Dropout(dropout_rate)(skip_resblock1_conv)
    resblock1_leaky1 = LeakyReLU()(enc_merge_1)
    resblock1_dropout1 = Dropout(dropout_rate)(resblock1_leaky1)
    resblock1_maxpool = MaxPooling2D((2, 2), padding='same')(resblock1_dropout1)
    resblock1_leaky2 = LeakyReLU()(resblock1_maxpool)
    resblock1_dropout2 = Dropout(dropout_rate)(resblock1_leaky2)
    resblock1_conv = Conv2D(32, (3 ,3), activation = 'relu', padding = 'same') (resblock1_dropout2)
    resblock1_dropout3 = Dropout(dropout_rate)(resblock1_conv)
    resblock1_merge = Add()([resblock1_dropout3, skip_resblock1_dropout])

    #Second ResBlock(Maxpool)
    skip_resblock2_conv = Conv2D(32, (3, 3), strides=(2,2), activation = 'relu', padding = 'same')(resblock1_merge)
    skip_resblock2_dropout = Dropout(dropout_rate)(skip_resblock2_conv)
    resblock2_leaky1 = LeakyReLU()(resblock1_merge)
    resblock2_dropout1 = Dropout(dropout_rate)(resblock2_leaky1)
    resblock2_maxpool1 = MaxPooling2D((2, 2), padding='same')(resblock2_dropout1)
    resblock2_leaky2 = LeakyReLU()(resblock2_maxpool1)
    resblock2_dropout2 = Dropout(dropout_rate)(resblock2_leaky2)
    resblock2_conv = Conv2D(32, (3 ,3), activation = 'relu', padding = 'same') (resblock2_dropout2)
    resblock2_dropout3 = Dropout(dropout_rate)(resblock2_conv)
    resblock2_merge = Add()([resblock2_dropout3, skip_resblock2_dropout])

    enc_upsample1 = UpSampling2D((2, 2))(resblock2_merge)

    #ResBlock(Conv2D)
    skip_resblock3_conv = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(enc_upsample1)
    skip_resblock3_dropout = Dropout(dropout_rate)(skip_resblock3_conv)
    resblock3_leaky1 = LeakyReLU()(enc_upsample1)
    resblock3_dropout1 = Dropout(dropout_rate)(resblock3_leaky1)
    resblock3_conv2D = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(resblock3_dropout1)
    resblock3_leaky2 = LeakyReLU()(resblock3_conv2D)
    resblock3_dropout2 = Dropout(dropout_rate)(resblock3_leaky2)
    resblock3_conv = Conv2D(16, (3 ,3), activation = 'relu', padding = 'same') (resblock3_dropout2)
    resblock3_dropout3 = Dropout(dropout_rate)(resblock3_conv)
    resblock3_merge = Add()([resblock3_dropout3, skip_resblock3_dropout])

    enc_upsample2 = UpSampling2D((2,2))(resblock3_merge)

    #ResBlock(Conv2D)
    skip_resblock4_conv = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(enc_upsample2)
    skip_resblock4_dropout = Dropout(dropout_rate)(skip_resblock4_conv)
    resblock4_leaky1 = LeakyReLU()(enc_upsample2)
    resblock4_dropout1 = Dropout(dropout_rate)(resblock4_leaky1)
    resblock4_conv2D = Conv2D(16, (3, 3), activation = 'relu', padding='same')(resblock4_dropout1)
    resblock4_leaky2 = LeakyReLU()(resblock4_conv2D)
    resblock4_dropout2 = Dropout(dropout_rate)(resblock4_leaky2)
    resblock4_conv = Conv2D(16, (3 ,3), activation = 'relu', padding = 'same') (resblock4_dropout2)
    resblock4_dropout3 = Dropout(dropout_rate)(resblock4_conv)
    resblock4_merge = Add()([resblock4_dropout3, skip_resblock4_dropout])

    enc_output = Conv2D(1, (3, 3), activation = 'sigmoid', padding='same')(resblock4_merge) 
    return enc_output

def vision_transformer(input_img, ):
    
    return