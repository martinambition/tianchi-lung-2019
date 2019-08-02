from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization
from keras.models import Model
from keras.metrics import categorical_accuracy
from config import *
class SimpleVgg():
    def __init__(self):
        self.use_batchnom = False

    def get_model(self,learning_rate):
        inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
        x = inputs

        x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = BatchNormalization()(x)

        x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = BatchNormalization()(x)

        x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = BatchNormalization()(x)

        x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = BatchNormalization()(x)

        x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
        x = GlobalMaxPooling3D()(x)

        x = Dense(32, activation='relu')(x)
        #x = Dropout(0.5)(x)
        x = Dense(CLASSIFY_OUTPUT_CHANNEL, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        #optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE)
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=[categorical_accuracy])

        return model