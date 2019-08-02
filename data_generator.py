import pandas as pd
import os
import pickle
import numpy as np
import random
import h5py
from glob import glob
from config import *
class DataGenerator(object):
    def __init__(self,name="lung"):
        self.name = name
        self.h5path = PREPROCESS_PATH_LUNG if name == "lung" else PREPROCESS_PATH_MEIASTINAL
        self.meta_dict =  self.get_meta_dict()
        self.records = self.get_ct_records()
        self.train_set,self.val_set = self.split_train_val()


    def split_train_val(self,ratio=0.8):
        record_len =self.records.shape[0]
        train_record = self.records[:int(record_len * ratio)]
        val_record = self.records[int(record_len * ratio):]
        return train_record, val_record

    def get_meta_dict(self):
        cache_file = '{}/all_meta_cache.meta'.format(PREPROCESS_PATH)
        if os.path.exists(cache_file):
            print('get meta_dict from cache')
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        meta_dict = {}
        for f in glob('{}/*.meta'.format(PREPROCESS_PATH_META)):
            seriesuid = os.path.splitext(os.path.basename(f))[0]

            with open(f, 'rb') as f:
                meta = pickle.load(f)
                meta_dict[meta['seriesuid']] = meta
        # cache it
        with open(cache_file, 'wb') as f:
            pickle.dump(meta_dict, f)

        return meta_dict

    def get_ct_records(self):
        numpy_files = glob(self.h5path + "/*.h5")
        fields = ['img_numpy_file', 'origin', 'spacing', 'shape']

        def fill_info(seriesuid):
            seriesuid = str(seriesuid)
            data = [None] * len(fields)
            matching = [s for s in numpy_files if seriesuid in s]

            if len(matching)>0:
                data[0] = matching[0]

            if seriesuid in self.meta_dict:
                t = self.meta_dict[seriesuid]
                data[1:] = [t['origin'], t['spacing'], t['shape']]

            return pd.Series(data, index=fields)

        records = pd.read_csv(ANNOTATION_FILE)

        if self.name =="lung":
            #records =  records[(records['label']==1) | (records['label']==5)]
            records =  records[(records['label']==5)]
        else:
            records = records[records['label'] > 5]

        records[fields] = records['seriesuid'].apply(fill_info)
        records.dropna(inplace=True)

        print('ct record size {}'.format(records.shape))
        return records

    def get_positive(self,record, shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH),random_offset=(0,0,0)):
        '''
        Get positive sample
        :param record: one focus record
        :param shape:
        :return: one positve sample,(block,mask)
        '''
        if not ENABLE_RANDOM_OFFSET:
            random_offset= (0,0,0)
        mask = np.zeros(shape)
        with h5py.File(record['img_numpy_file'], 'r') as hf:
            W, H, D = hf['img'].shape[0], hf['img'].shape[1], hf['img'].shape[2]

            #DiameterX
            diameter = np.array([record['diameterX'],record['diameterY'],record['diameterZ']])
            radius = np.ceil(diameter/record['spacing']/2).astype(int)
            upper_z = 2
            orgin_coord = np.array([record['coordX'], record['coordY'], record['coordZ']+upper_z])
            
            orgin_coord = np.abs((orgin_coord - record['origin']) / record['spacing'])
            coord = orgin_coord + random_offset

            x, y, z = int(coord[0] - shape[0] // 2), int(coord[1] - shape[1] // 2), int(coord[2] - shape[2] // 2)

            x, y, z = max(x, 0), max(y, 0), max(z, 0)
            x, y, z = min(x, W - shape[0] - 1), min(y, H - shape[1] - 1), min(z, D - shape[2] - 1)

            block = hf['img'][x:x + shape[0], y:y + shape[1], z:z + shape[2]]

            # cub_coord = np.array([INPUT_WIDTH // 2, INPUT_HEIGHT // 2, INPUT_DEPTH // 2])

            real_coord = (orgin_coord - np.array([x, y, z])).astype(int)


            min_cor = np.clip(real_coord - radius,0,None)
            max_cor = real_coord + radius + 1# Add one  
            if max_cor[0]>INPUT_WIDTH:
                max_cor[0] = INPUT_WIDTH
            if max_cor[1]>INPUT_HEIGHT:
                max_cor[1] = INPUT_HEIGHT
            if max_cor[2]>INPUT_DEPTH:
                max_cor[2] = INPUT_DEPTH

            mask[min_cor[0]:max_cor[0],
            min_cor[1]:max_cor[1],
            min_cor[2]:max_cor[2]] = 1.0
            # print(f"Found Positive:{(x,y,z)},{(x+shape[0],y+shape[1],z+shape[2])}")
            return block,mask


    def get_negative(self,slice_records,shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH)):
        '''
        Get negative sample
        :param slice_records: one CT related records
        :param shape:
        :return: negative sample,(block,mask)
        '''
        first_record = slice_records.iloc[0]
        W, H, D = first_record['shape'][0],first_record['shape'][1],first_record['shape'][2]
        mask = np.zeros((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH))
        block = np.zeros((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH))
        #All the coordZ seems too low.
        
        focus_coords = np.array([slice_records['coordX'].values, slice_records['coordY'].values, slice_records['coordZ'].values])
        focus_coords = focus_coords.transpose(1, 0)
        origin  = first_record['origin']
        spacing = first_record['spacing']
        focus_coords = np.abs((focus_coords - origin) / spacing)
        focus_dim = np.array([slice_records['diameterX'].values, slice_records['diameterY'].values, slice_records['diameterZ'].values])
        focus_dim = focus_dim.transpose(1, 0)
        focus_size = focus_dim/spacing

        focus_start_coords = focus_coords - focus_size//2
        focus_end_coords = focus_coords + focus_size // 2

        #Get ramdom negative
        with h5py.File(first_record['img_numpy_file'], 'r') as hf:
            while True:
                x, y, z = random.randint(0, W - shape[0] - 1), random.randint(0, H - shape[1] - 1), random.randint(0, D - shape[
                    2] - 1)
                if not self.check_overlap((x,y,z),(x+shape[0],y+shape[1],z+shape[2]),
                                      focus_start_coords,focus_end_coords):
                    block = hf['img'][x:x + shape[0], y:y + shape[1], z:z + shape[2]]
                    #print(f"Found Negative:{(x,y,z)},{(x+shape[0],y+shape[1],z+shape[2])}")
                    if np.sum(block!=-ZERO_CENTER) > 0:
                        break
        return block, mask

    def check_overlap(self,start,end, focus_start_coords,focus_end_coords):
        for i in range(len(focus_start_coords)):
            cub_start = focus_start_coords[i]
            cub_end = focus_end_coords[i]
            if self.check_cub_overlap(start,end,cub_start,cub_end):
                #print(f'Found Collision,{start},{end},{cub_start},{cub_end}')
                return True
        return False

    def check_cub_overlap(self,cub_start,cub_end, focus_start,focus_end):
        x_min = cub_start[0]
        x_max = cub_end[0]
        y_min = cub_start[1]
        y_max = cub_end[1]
        z_min = cub_start[2]
        z_max = cub_end[2]

        x_min2 = focus_start[0]
        x_max2 = focus_end[0]
        y_min2 = focus_start[1]
        y_max2 = focus_end[1]
        z_min2 = focus_start[2]
        z_max2 = focus_end[2]
        #print('Box2 min %.2f, %.2f, %.2f' % (x_min2, y_min2, z_min2))
        #print('Box2 max %.2f, %.2f, %.2f' % (x_max2, y_max2, z_max2))
        isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                       or (x_min <= x_max2 and x_min >= x_min2) \
                       or (x_min <= x_min2 and x_max >= x_max2) \
                       or (x_min >= x_min2 and x_max <= x_max2) \
                       ) \
                      and ((y_max >= y_min2 and y_max <= y_max2) \
                           or (y_min <= y_max2 and y_min >= y_min2) \
                           or (y_min <= y_min2 and y_max >= y_max2) \
                           or (y_min >= y_min2 and y_max <= y_max2) \
                           ) \
                      and ((z_max >= z_min2 and z_max <= z_max2) \
                           or (z_min <= z_max2 and z_min >= z_min2) \
                           or (z_min <= z_min2 and z_max >= z_max2) \
                           or (z_min >= z_min2 and z_max <= z_max2) \
                           )
        return isColliding

    def flow_segmentation(self,mode = 'train',batch_size = TRAIN_BATCH_SIZE):
        idx = 0
        records = self.train_set if mode =='train' else self.val_set
        shape = (INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH)
        X = np.zeros((batch_size, *shape, INPUT_CHANNEL))
        y = np.zeros((batch_size, *shape, OUTPUT_CHANNEL))
        y_class = np.zeros((batch_size, CLASSIFY_OUTPUT_CHANNEL))

    

        while True:
            for b in range(batch_size):
                #Random select
                idx = random.randint(0, records.shape[0] - 1)
                record = records.iloc[idx]
                is_positive_sample = random.random() <= TRAIN_SEG_POSITIVE_SAMPLE_RATIO
                random_offset = np.array([
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET)
                ])
                if is_positive_sample:
                    X[b, :, :, :, 0],y[b, :, :, :, 0] =  self.get_positive(record,shape,random_offset)
                    y_class[b, label_softmax[record['label']]] = 1
                else:
                    #Get all the focus records for one CT
                    focus_records = records.loc[records['seriesuid'] == record['seriesuid']]
                    if focus_records.empty:
                        print(record['seriesuid'])
                    X[b, :, :, :, 0], y[b, :, :, :, 0] = self.get_negative(focus_records,shape)
                    y_class[b, 0] = 1

            # rotate
            # for b in range(batch_size):
            #     _perm = np.random.permutation(3)
            #     X[b, :, :, :, 0] = np.transpose(X[b, :, :, :, 0], _perm)
            #     y[b, :, :, :, 0] = np.transpose(y[b, :, :, :, 0], _perm)

            yield X.astype(np.float16), y.astype(np.float16)#, y_class

    def flow_classfication(self, mode='train', batch_size=TRAIN_BATCH_SIZE):
        idx = 0
        records = self.train_set if mode == 'train' else self.val_set
        shape = (CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH)
        X = np.zeros(
            (batch_size, *shape, CLASSIFY_INPUT_CHANNEL))
        y = np.zeros((batch_size, CLASSIFY_OUTPUT_CHANNEL))



        while True:
            for b in range(batch_size):
                idx = random.randint(0, records.shape[0] - 1)
                record = records.iloc[idx]
                is_positive_sample = random.random() <= TRAIN_CLASSIFY_POSITIVE_SAMPLE_RATIO
                random_offset = np.array([
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET)
                ])
                if is_positive_sample:
                    X[b, :, :, :, 0], _ = self.get_positive(record,shape,random_offset)
                    y[b, label_softmax[record['label']]] = 1

                else:
                    # Get all the focus records for one CT
                    focus_records = records.loc[records['seriesuid'] == record['seriesuid']]
                    if focus_records.empty:
                        print(record['seriesuid'])
                    X[b, :, :, :, 0], _ = self.get_negative(focus_records,shape)
                    y[b, 0] = 1

            yield X, y

