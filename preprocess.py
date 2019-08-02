from config import *
import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure, segmentation
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from glob import glob
import h5py
import scipy
import os
import pickle
import pandas as pd
from tqdm import tqdm
class Preprocess():
    def __init__(self):
        pass

    #CT_PATH, PREPROCESS_PATH_LUNG
    def handle(self,ct_path,out_path):
        self.anotations = pd.read_csv(ANNOTATION_FILE)
        print('start preprocess')
        self.ct_files = glob(ct_path)
        
        self.lung_path = os.path.join(out_path,'lung')
        self.mediastinal_path = os.path.join(out_path,'mediastinal')
        self.meta_path = os.path.join(out_path,'meta')
        if not os.path.exists(self.lung_path):
            os.makedirs(self.lung_path)
        if not os.path.exists(self.mediastinal_path):
            os.makedirs(self.mediastinal_path)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)
        
        handled_ids = set([f[-9:-3] for f in glob('{}/*.h5'.format(self.lung_path))])
        print('{} total, {} processed'.format(len(self.ct_files), len(handled_ids)))

        counter = 0
        for f in tqdm(self.ct_files):
            seriesuid = os.path.splitext(os.path.basename(f))[0]
            if seriesuid in handled_ids:
                print('{} handled'.format(seriesuid))
                continue
            # anno = self.anotations.loc[self.anotations['seriesuid'] == int(seriesuid)]
            # if anno.empty or anno[(anno['label']==1) | (anno['label']==5)].empty:
            #     continue
            counter += 1
            print('{} process {}'.format(counter, f))

            itk_img = sitk.ReadImage(f)
            img = sitk.GetArrayFromImage(itk_img)  # (depth, height, width)
            img = np.transpose(img, (2, 1, 0))  # (width, height, depth)

            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            #Resample to 1:1:1
            img, new_spacing = self.resample(img, spacing)

            new_img_1 = img.copy()
            new_img_2 = img.copy()

            #Generate Lung Image
            lung_img = self.extract_lung_img_3d(new_img_1)
            lung_img = self.normalize(lung_img,LUNG_MIN_BOUND,LUNG_MAX_BOUND,zero_center=True)
            lung_img = lung_img.astype(np.float16)
            #Generate Mediastinal Image
            mediastinal_img =self.normalize(new_img_2,CHEST_MIN_BOUND,CHEST_MAX_BOUND,zero_center=True)
            mediastinal_img = mediastinal_img.astype(np.float16)

            meta = {
                'seriesuid': seriesuid,
                'shape': new_img_1.shape,
                'origin': origin,
                'spacing': new_spacing
            }
            self.save_to_numpy(seriesuid, lung_img,mediastinal_img, meta)

        print('all preprocess done')

    # Resample to 1mm, 1mm, 1mm
    def resample(self,image, spacing, new_spacing=[1, 1, 1]):
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing

    def normalize(self,img,lower,upper,zero_center =False):
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        if zero_center:
            img  = img - ZERO_CENTER
        return img

    def normalize_all(self,imgs):
        for i in range(imgs.shape[2]):
            imgs[:, :, i] = self.normalize(imgs[:, :, i])

    def extract_mediastinal_img(self,imgs):
        return np.clip(imgs,CHEST_MIN_BOUND,CHEST_MAX_BOUND)

    def extract_lung_img_3d(self,imgs):
        ret = np.zeros(imgs.shape)
        for i in range(imgs.shape[2]):
            ret[:,:,i] = self.extract_lung_img_2D(imgs[:,:,i])
        return ret

    def extract_lung_img_2D(self, im, plot=False):
        binary = im < -550
        cleared = segmentation.clear_border(binary)
        label_image = measure.label(cleared)
        areas = [r.area for r in measure.regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0

        selem = morphology.disk(2)
        binary = morphology.binary_erosion(binary, selem)

        selem = morphology.disk(10)
        binary = morphology.binary_closing(binary, selem)

        # #?
        # selem = morphology.disk(10)
        # binary = morphology.binary_dilation(binary, selem)

        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

        get_high_vals = binary == 0
        im[get_high_vals] = LUNG_MIN_BOUND
        return im



    def save_to_numpy(self,seriesuid, lung_img,mediastinal_img, meta):
       
        with h5py.File(os.path.join(self.lung_path,seriesuid+'.h5'), 'w') as hf:
            hf.create_dataset('img', data=lung_img)
        with h5py.File(os.path.join(self.mediastinal_path,seriesuid+'.h5'), 'w') as hf:
            hf.create_dataset('img', data=mediastinal_img)
        with open(os.path.join(self.meta_path,seriesuid+'.meta'), 'wb') as f:
            pickle.dump(meta, f)
    
if __name__ =="__main__" :
    p = Preprocess()
    p.handle()