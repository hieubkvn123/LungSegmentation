import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

class EMA_Unet:
    def __init__(self, momentum=0.999):
        self.student = EMA_Unet.build_unet_model()
        self.teacher = EMA_Unet.build_unet_model()

        self.momentum = momentum

    def update_ema_params(self, step):
        alpha = self.momentum #min(self.momentum, 1 - (1/(1 + step)))

        for i, s_layer in enumerate(self.student.layers):
            s_params = s_layer.weights
            
            if(len(s_params) > 0):
                updated_params = []
                for j in range(len(s_params)):
                    t_params = self.teacher.layers[i].weights
                    updated_params.append(alpha * t_params[j] + (1 - alpha) * s_params[j])

                self.teacher.layers[i].set_weights(updated_params)

    @staticmethod
    def build_unet_model():
        inputs = Input(shape=(256, 256, 3))
        init = 'he_normal' # l1(2e-4)
        reg  = l1(2e-04)

        ## Encoding path ##
        conv_1_1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(inputs)
        conv_1_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_1_1)

        pool = MaxPooling2D(pool_size=(2,2))(conv_1_2)

        conv_2_1 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_2_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_2_1)

        pool = MaxPooling2D(pool_size=(2,2))(conv_2_2)

        conv_3_1 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_3_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_3_1)

        pool = MaxPooling2D(pool_size=(2,2))(conv_3_2)

        conv_4_1 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_4_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_4_1)

        pool = MaxPooling2D(pool_size=(2,2))(conv_4_2)

        conv_5_1 = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(pool)
        conv_5_2 = Conv2D(1024, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_5_1)

        ## Decoding path ##
        # up1 = Conv2DTranspose(512, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(conv_5_2)
        up1 = UpSampling2D(size=(2,2))(conv_5_2)
        up1 = Concatenate(axis=3)([conv_4_2, up1])

        upconv_1_1 = Conv2D(512, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up1)
        upconv_1_2 = Conv2D(512, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_1_1)

        # up2 = Conv2DTranspose(256, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_1_2)
        up2 = UpSampling2D(size=(2,2))(upconv_1_2)
        up2 = Concatenate(axis=3)([conv_3_2, up2])

        upconv_2_1 = Conv2D(256, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up2)
        upconv_2_2 = Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_2_1)

        # up3 = Conv2DTranspose(128, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_2_2)
        up3 = UpSampling2D(size=(2,2))(upconv_2_2)
        up3 = Concatenate(axis=3)([conv_2_2, up3])

        upconv_3_1 = Conv2D(128, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up3)
        upconv_3_2 = Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_3_1)

        # up4 = Conv2DTranspose(64, kernel_size=2, strides=(2,2), padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_3_2)
        up4 = UpSampling2D(size=(2,2))(upconv_3_2)
        up4 = Concatenate(axis=3)([conv_1_2, up4])

        upconv_4_1 = Conv2D(64, kernel_size=2, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(up4)
        upconv_4_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_1)
        output = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same', kernel_initializer=init, kernel_regularizer=reg)(upconv_4_2)

        model = Model(inputs=inputs, outputs=output, name='UNet-Lung-Segmentation')
        return model
