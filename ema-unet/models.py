import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

class EMA_Unet:
    def __init__(self, momentum=0.999):
        self.student = EMA_Unet.build_unet_model()
        self.teacher = EMA_Unet.build_unet_model()

        self.momentum = momentum
        # Initially, params_t = params_s
        self.teacher.set_weights(self.student.get_weights())
    
    def update_ema_params(self, step, ema=False):
        alpha = min(self.momentum, 1 - (1/(1 + step)))
        if(step == 1 or not ema):
            self.teacher.set_weights(self.student.get_weights())
        else:
            for i, s_layer in enumerate(self.student.layers):
                s_params = s_layer.weights
                
                if(len(s_params) > 0):
                    updated_params = []
                    for j in range(len(s_params)):
                        t_params = self.teacher.layers[i].weights
                        updated_params.append(alpha * t_params[j] + (1 - alpha) * s_params[j])

                    self.teacher.layers[i].set_weights(updated_params)

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


    @staticmethod
    def build_unet_model():
        inputs = Input(shape=(256, 256, 3))
        init = 'he_uniform' # l1(2e-4)
        reg  = l1(2e-04)

        ## Encoding path ##
        conv_1_1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(inputs)
        conv_1_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_1_1)
        conv_1_2 = Dropout(0.5)(conv_1_2)

        pool = MaxPooling2D(pool_size=(2,2))(conv_1_2)

        conv_2_1 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_2_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_2_1)
        conv_2_2 = Dropout(0.5)(conv_2_2)

        pool = MaxPooling2D(pool_size=(2,2))(conv_2_2)

        conv_3_1 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_3_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_3_1)
        conv_3_2 = Dropout(0.5)(conv_3_2)

        pool = MaxPooling2D(pool_size=(2,2))(conv_3_2)

        conv_4_1 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_4_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_4_1)
        conv_4_2 = Dropout(0.5)(conv_4_2)

        pool = MaxPooling2D(pool_size=(2,2))(conv_4_2)

        conv_5_1 = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_5_2 = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_5_1)
        conv_5_2 = Dropout(0.5)(conv_5_2)

        ## Decoding path ##
        # up1 = Conv2DTranspose(512, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_5_2)
        up1 = UpSampling2D(size=(2,2))(conv_5_2)
        up1 = Concatenate(axis=3)([conv_4_2, up1])

        upconv_1_1 = Conv2D(512, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up1)
        upconv_1_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_1_1)
        upconv_1_2 = Dropout(0.5)(upconv_1_2)

        # up2 = Conv2DTranspose(256, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_1_2)
        up2 = UpSampling2D(size=(2,2))(upconv_1_2)
        up2 = Concatenate(axis=3)([conv_3_2, up2])

        upconv_2_1 = Conv2D(256, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up2)
        upconv_2_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_2_1)
        upconv_2_2 = Dropout(0.5)(upconv_2_2)

        # up3 = Conv2DTranspose(128, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_2_2)
        up3 = UpSampling2D(size=(2,2))(upconv_2_2)
        up3 = Concatenate(axis=3)([conv_2_2, up3])

        upconv_3_1 = Conv2D(128, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up3)
        upconv_3_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_3_1)
        upconv_3_2 = Dropout(0.5)(upconv_3_2)

        # up4 = Conv2DTranspose(64, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_3_2)
        up4 = UpSampling2D(size=(2,2))(upconv_3_2)
        up4 = Concatenate(axis=3)([conv_1_2, up4])

        upconv_4_1 = Conv2D(64, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up4)
        upconv_4_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_1)
        upconv_4_2 = Dropout(0.5)(upconv_4_2)
        output = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_2)

        model = Model(inputs=inputs, outputs=output, name='UNet-Lung-Segmentation')
        return model
