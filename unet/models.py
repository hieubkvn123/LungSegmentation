import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def build_unet_model():
    inputs = Input(shape=(256, 256, 3))

    ## Encoding path ##
    conv_1_1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv_1_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv_1_1)

    pool = MaxPooling2D(pool_size=(2,2))(conv_1_2)

    conv_2_1 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool)
    conv_2_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv_2_1)

    pool = MaxPooling2D(pool_size=(2,2))(conv_2_2)

    conv_3_1 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(pool)
    conv_3_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv_3_1)

    pool = MaxPooling2D(pool_size=(2,2))(conv_3_2)

    conv_4_1 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(pool)
    conv_4_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(conv_4_1)

    pool = MaxPooling2D(pool_size=(2,2))(conv_4_2)

    conv_5_1 = Conv2D(1024, kernel_size=3, padding='same', activation='relu')(pool)
    conv_5_2 = Conv2D(1024, kernel_size=3, padding='same', activation='relu')(conv_5_1)

    ## Decoding path ##
    up1 = Conv2DTranspose(512, kernel_size=2, strides=(2,2), padding='same', activation='relu')(conv_5_2)
    up1 = Concatenate(axis=3)([conv_4_2, up1])

    upconv_1_1 = Conv2D(512, kernel_size=2, padding='same', activation='relu')(up1)
    upconv_1_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(upconv_1_1)

    up2 = Conv2DTranspose(256, kernel_size=2, strides=(2,2), padding='same', activation='relu')(upconv_1_2)
    up2 = Concatenate(axis=3)([conv_3_2, up2])

    upconv_2_1 = Conv2D(256, kernel_size=2, padding='same', activation='relu')(up2)
    upconv_2_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(upconv_2_1)

    up3 = Conv2DTranspose(128, kernel_size=2, strides=(2,2), padding='same', activation='relu')(upconv_2_2)
    up3 = Concatenate(axis=3)([conv_2_2, up3])

    upconv_3_1 = Conv2D(128, kernel_size=2, padding='same', activation='relu')(up3)
    upconv_3_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(upconv_3_1)

    up4 = Conv2DTranspose(64, kernel_size=2, strides=(2,2), padding='same', activation='relu')(upconv_3_2)
    up4 = Concatenate(axis=3)([conv_1_2, up4])

    upconv_4_1 = Conv2D(64, kernel_size=2, padding='same', activation='relu')(up4)
    upconv_4_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(upconv_4_1)
    upconv_4_3 = Conv2D(2, kernel_size=3, padding='same', activation='relu')(upconv_4_2)
    output = Softmax(axis=3)(upconv_4_3)

    model = Model(inputs=inputs, outputs=output, name='UNet-Lung-Segmentation')
    return model
