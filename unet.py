from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout,BatchNormalization
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K
from config import  *
from skimage import morphology, measure, segmentation
from keras.utils import multi_gpu_model
# from visual_utils import VisualUtil
import numpy as np

SMOOTH = 1.0

class UNet():
    def __init__(self):
        pass

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return 1 - UNet.dice_coef(y_true, y_pred)

    @staticmethod
    def metrics_true_sum(y_true, y_pred):
        return K.sum(y_true)

    @staticmethod
    def metrics_pred_sum(y_true, y_pred):
        return K.sum(y_pred)

    @staticmethod
    def metrics_pred_max(y_true, y_pred):
        return K.max(y_pred)

    @staticmethod
    def metrics_pred_min(y_true, y_pred):
        return K.min(y_pred)

    @staticmethod
    def metrics_pred_mean(y_true, y_pred):
        return K.mean(y_pred)

#     def get_model(self,learning_rate =TRAIN_SEG_LEARNING_RATE ,enable_drop_out=False):
#         inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

#         conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#         conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#         pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
#         conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#         conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#         pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
#         conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#         conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#         pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
#         conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#         conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#         drop4 = Dropout(0.5)(conv4)
#         pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

#         conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#         conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#         drop5 = Dropout(0.5)(conv5)

#         up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling3D(size=(2, 2, 2))(drop5))
#         merge6 = concatenate([drop4, up6], axis=3)
#         conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#         conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

#         up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling3D(size=(2, 2, 2))(conv6))
#         merge7 = concatenate([conv3, up7], axis=3)
#         conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#         conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

#         up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling3D(size=(2, 2, 2))(conv7))
#         merge8 = concatenate([conv2, up8], axis=3)
#         conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#         conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

#         up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling3D(size=(2, 2, 2))(conv8))
#         merge9 = concatenate([conv1, up9], axis=3)
#         conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#         conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#         conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#         conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)

#         model = Model(inputs=inputs, outputs=conv10)
#         model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=UNet.dice_coef_loss,
#                   metrics=[UNet.dice_coef, UNet.metrics_true_sum, UNet.metrics_pred_sum,
#                            UNet.metrics_pred_max, UNet.metrics_pred_min,
#                            UNet.metrics_pred_mean])
#         return model
    def get_complex_model(self,enable_drop_out=False):
        inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

        conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        
        conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

        conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=(2, 2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=(2, 2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=(2, 2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=(2, 2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv3D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=UNet.dice_coef_loss,
                  metrics=[UNet.dice_coef, UNet.metrics_true_sum, UNet.metrics_pred_sum,
                           UNet.metrics_pred_max, UNet.metrics_pred_min,
                           UNet.metrics_pred_mean])
        return model
    def get_1024_model(self,enable_drop_out=False):
        inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))
    
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
        pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)
        
        conv5_1 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(pool5)
        conv5_1 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(conv5_1)
        
        up6_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_1), conv5], axis=-1)
        conv6_1 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(up6_1)
        conv6_1 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6_1)
        
        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_1), conv4], axis=-1)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)
    
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=UNet.dice_coef_loss,
                      metrics=[UNet.dice_coef, UNet.metrics_true_sum, UNet.metrics_pred_sum,
                               UNet.metrics_pred_max, UNet.metrics_pred_min,
                               UNet.metrics_pred_mean])
    
        return model
    def get_model_with_bn(self,learning_rate =TRAIN_SEG_LEARNING_RATE ,enable_drop_out=False,enable_bn=True):
        inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        if enable_bn:
            conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        if enable_bn:
            conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        if enable_bn:
            conv2 = BatchNormalization()(conv2)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        if enable_bn:
            conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        if enable_bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        if enable_bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        if enable_bn:
            conv4 = BatchNormalization()(conv4)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        if enable_bn:
            conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        if enable_bn:
            conv5 = BatchNormalization()(conv5)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
        if enable_bn:
            conv5 = BatchNormalization()(conv5)

        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=-1)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        if enable_bn:
            conv6 = BatchNormalization()(conv6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
        if enable_bn:
            conv6 = BatchNormalization()(conv6)
            
        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        if enable_bn:
            conv7 = BatchNormalization()(conv7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
        if enable_bn:
            conv7 = BatchNormalization()(conv7)

        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        if enable_bn:
            conv8 = BatchNormalization()(conv8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
        if enable_bn:
            conv8 = BatchNormalization()(conv8)
            
        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        if enable_bn:
            conv9 = BatchNormalization()(conv9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
        if enable_bn:
            conv9 = BatchNormalization()(conv9)
        conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)  
        model = Model(inputs=inputs, outputs=conv10)
#         model = multi_gpu_model(model, gpus=3)
        model.compile(optimizer=Adam(lr=learning_rate), loss=UNet.dice_coef_loss,
                      metrics=[UNet.dice_coef, UNet.metrics_true_sum, UNet.metrics_pred_sum,
                               UNet.metrics_pred_max, UNet.metrics_pred_min,
                               UNet.metrics_pred_mean])

        return model
    def get_model(self,learning_rate =TRAIN_SEG_LEARNING_RATE ,enable_drop_out=False):
        inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = concatenate([UpSampling3D(size=(2, 2, 2))(drop5), drop4], axis=-1)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
        conv6 = Dropout(0.5)(conv6)
        
        up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
        conv7 = Dropout(0.5)(conv7)
        
        up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
#         model = multi_gpu_model(model, gpus=3)
        model.compile(optimizer=Adam(lr=learning_rate), loss=UNet.dice_coef_loss,
                      metrics=[UNet.dice_coef, UNet.metrics_true_sum, UNet.metrics_pred_sum,
                               UNet.metrics_pred_max, UNet.metrics_pred_min,
                               UNet.metrics_pred_mean])

        return model
    def get_evaluator(self,generator,name):
        return UNetEvaluator(generator,name)

class UNetEvaluator(Callback):
    def __init__(self,generator,name):
        self.counter = 0
        self.generator =generator
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        #if self.counter % TRAIN_SEG_EVALUATE_FREQ == 0:
        self.do_evaluate(self.model)

    def do_evaluate(self,model):
        print('Model evaluating')
        if callable(self.generator):
            X, y_true = next(self.generator('val',self.name))
        else:
             X, y_true = next(self.generator.flow_segmentation('val'))
        y_true = y_true.astype(np.float64)
        y_pred = model.predict(X)
        #X, y_true, y_pred = X[:, :, :,:, 0], y_true[:, :, :,:, 0], y_pred[:, :, :, :,0]
        intersection = y_true * y_pred
        recall = (np.sum(intersection) + SMOOTH) / (np.sum(y_true) + SMOOTH)
        precision = (np.sum(intersection) + SMOOTH) / (np.sum(y_pred) + SMOOTH)
        print('Average recall {:.4f}, precision {:.4f}'.format(recall, precision))

        for threshold in range(0, 10, 2):
            threshold = threshold / 10.0
            pred_mask = (y_pred > threshold).astype(np.uint8)
            intersection = y_true * pred_mask
            recall = (np.sum(intersection) + SMOOTH) / (np.sum(y_true) + SMOOTH)
            precision = (np.sum(intersection) + SMOOTH) / (np.sum(pred_mask) + SMOOTH)
            print("Threshold {}: recall {:.4f}, precision {:.4f}".format(threshold, recall, precision))
            print(str(np.sum(pred_mask))+'/'+str(np.sum(y_true))+'/'+
                   str(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3]))

        #regions = measure.regionprops(measure.label(y_pred))
        #print('Num of pred regions {}'.format(len(regions)))

        # if DEBUG_PLOT_WHEN_EVALUATING_SEG:
        #     VisualUtil.plot_comparison(X, y_true, y_pred)
        #     VisualUtil.plot_slices(X)
        #     VisualUtil.plot_slices(y_true)
        #     VisualUtil.plot_slices(y_pred)