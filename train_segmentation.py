from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from config import *
from unet import UNet
from data_generator import  DataGenerator
import time
from glob import glob
import random
import os
import numpy as np

# def flow(mode='train',name="lung", batch_size=TRAIN_BATCH_SIZE):
#     PAHT= PREPROCESS_GENERATOR_LUNG_PATH if name =="lung" else PREPROCESS_GENERATOR_MEIASTINAL_PATH
#     files = glob(PAHT+'/*_x_'+mode+'.npy')
#     #random.seed(9)
#     while True:
#         idx = random.randint(0, len(files) - 1)
#         file = files[idx]
#         name = os.path.splitext(os.path.basename(file))[0]
#         id = name.split('_')[0]

#         X = np.load(file)
#         y = np.load(PAHT+ '/'+id+'_y_'+mode+'.npy')
        
#         #Ignore negative sample
#         if np.sum(y) == 0 and random.random() < 0.8:
#             continue
# #         s = random.randint(0, 8)
# #         yield X[s:s+8,...],y[s:s+8,...]
#         yield X, y
        
def seg_train(name,learning_rate,init_weight=None):
    print('start seg_train')
    net = UNet()
    model = net.get_model(learning_rate,enable_drop_out=False)
    if not init_weight == None:
        model.load_weights(init_weight)
    model.summary()
    generator =  DataGenerator(name=name)

    run = '{}-{}-{}'.format(name, time.localtime().tm_hour, time.localtime().tm_min)
    log_dir = SEG_LOG_DIR.format(run)
    check_point = log_dir + '/' + name + '_checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

    print("seg train round {}".format(run))
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=False)
    checkpoint = ModelCheckpoint(filepath=check_point, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=TRAIN_EARLY_STOPPING, verbose=1)
    evaluator = net.get_evaluator(generator,name)
    model.fit_generator(generator.flow_segmentation(mode='train'), steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        validation_data=generator.flow_segmentation(mode='val'), validation_steps=TRAIN_VALID_STEPS,
                        epochs=TRAIN_EPOCHS, verbose=1,
                        callbacks=[tensorboard, checkpoint ,early_stopping, evaluator]) #
#     model.fit_generator(flow('train', name), steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
#                     validation_data=flow('val',name), validation_steps=TRAIN_VALID_STEPS,
#                     epochs=TRAIN_EPOCHS, verbose=1,
#                     callbacks=[tensorboard, checkpoint, early_stopping, evaluator])

if __name__ == '__main__':
    seg_train('lung',TRAIN_SEG_LEARNING_RATE)
