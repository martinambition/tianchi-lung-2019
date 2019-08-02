from resnet_helper import Resnet3DBuilder
from keras.optimizers import Adam
from keras.layers import Input, Conv3D, Dense, BatchNormalization, Add, Flatten, Concatenate, AveragePooling3D, GlobalMaxPooling3D, Activation
from keras.models import Model
from keras.metrics import categorical_accuracy
from config import *

class Resnet:

    def __init__(self):
        pass

#     def get_model(self,learning_rate):
#         inputs = (CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL)
#         model = Resnet3DBuilder.build_resnet_50(inputs,CLASSIFY_OUTPUT_CHANNEL)
#         model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=[categorical_accuracy])
#         return model
    def conv_bn_relu(self,x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', apply_relu=True):
        x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if apply_relu:
            x = Activation('relu')(x)
        return x

    def bottleneck(self,x, shrinkage=False):
        print('resnet block, shrinkage:{}'.format(shrinkage))
        print(x.get_shape())

        input_filters = x.get_shape()[4].value
        keep_filters = input_filters // 2 if shrinkage else input_filters // 4
        output_filters = input_filters * 2 if shrinkage else input_filters
        first_strides = (2, 2, 2) if shrinkage else (1, 1, 1)

        residual = self.conv_bn_relu(x, filters=keep_filters, kernel_size=(1, 1, 1), strides=first_strides)
        residual = self.conv_bn_relu(residual, filters=keep_filters, kernel_size=(3, 3, 3))
        residual = self.conv_bn_relu(residual, filters=output_filters, kernel_size=(1, 1, 1), apply_relu=False)

        if shrinkage:
            x = self.conv_bn_relu(x, filters=output_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), apply_relu=False)

        print(residual.get_shape())
        print(x.get_shape())
        x = Add()([residual, x])
        x = Activation('relu')(x)

        return x

    def get_model(self,learning_rate):
        inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))

        x = self.conv_bn_relu(inputs, RESNET_INITIAL_FILTERS)

        print('base')
        print(x.get_shape())

        for i in range(RESNET_BLOCKS):
            x = self.bottleneck(x, shrinkage=(i % RESNET_SHRINKAGE_STEPS == 0))

        print('top')
        x = GlobalMaxPooling3D()(x)
        print(x.get_shape())

        x = Dense(5, activation='softmax')(x)
        print(x.get_shape())

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model