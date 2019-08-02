from config import *
from unet import UNet
from resnet import Resnet
from skimage import morphology, measure, segmentation,filters
import scipy.ndimage
import glob
import os
import pickle
import h5py
import numpy as np
import pandas as pd
from config import *
from tqdm import tqdm
import SimpleITK as sitk

def find_all_sensitive_point():
    files = glob.glob(TEST_FOLDER+"/*.mhd")
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ']
    found_record = pd.DataFrame(columns=columns)
    for index,file in enumerate(tqdm(files)):
        seriesuid = os.path.splitext(os.path.basename(file))[0]
        itk_img = sitk.ReadImage(file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        img_array = np.transpose(img_array, (2, 1, 0)) # (x, y, z)
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        centers = find_sensitive_point_from_one_lung(img_array)
        center_in_world = centers*spacing
        for cindex in range(center_in_world.shape[0]):
            c_in_w = center_in_world[cindex]
            new_row = pd.DataFrame([[ seriesuid,c_in_w[0],c_in_w[1],c_in_w[2] ]], columns=columns)
            found_record = found_record.append(new_row, ignore_index=True)
    found_record.to_csv('./output/sensitive_point.csv',index=False)
    
def find_sensitive_point_from_one_lung(ret_img):
    area_threshold = 10000
    threshold = 299
    temp_img = ret_img.copy()

    #Clear Bound
    mask = temp_img>threshold
    #mask = morphology.binary_erosion(mask, selem=np.ones((2, 1, 1)))#binary_opening  dilation
    mask = morphology.binary_dilation(mask, selem=np.ones((2, 2, 2)))#binary_opening  dilation
    #edges = filters.hessian(mask)
    mask = scipy.ndimage.binary_fill_holes(mask)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    for r in regions:
        if r.area>area_threshold:
            for c in r.coords:
                temp_img[c[0], c[1], c[2]] = 0
    #
    mask = temp_img==300
    mask = morphology.dilation(mask, np.ones([3, 3, 3]))
    mask = morphology.dilation(mask, np.ones([3, 3, 3]))
    mask = morphology.erosion(mask, np.ones([3, 3, 3]))
    centers = []
    for prop in regions:
        B = prop.bbox
        if B[3] - B[0] > 2 and B[4] - B[1] > 2 and B[5] - B[2] > 2: # ignore too small focus
            x = int((B[3] + B[0]) / 2.0)
            y = int((B[4] + B[1]) / 2.0)
            z = int((B[5] + B[2]) / 2.0)
            centers.append(np.array([x, y, z]))
    return np.array(centers)

def predict_test(name='lung',mode='test',seg_model_path=SEG_LUNG_TRAIN_WEIGHT,class_model_path=CLASS_LUNG_TRAIN_WEIGHT,
                 seg_thresh_hold=0.8,limit = [0,0]):
    detect_net = UNet()
    class_net = Resnet()

    detect_model = detect_net.get_model(0.1)
    detect_model.load_weights(seg_model_path)
    class_model = class_net.get_model(0.1)
    class_model.load_weights(class_model_path)

    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    df = pd.DataFrame(columns=columns)
    for img, meta in get_files(name,mode):
        count = 0
        cubs = []
        cub_sizes = []
        for w in range(limit[0], img.shape[0]-limit[0], 32):
            for h in range(limit[1], img.shape[1]-limit[1], 32):
                for d in range(0, img.shape[2], 32):
                    if d + INPUT_DEPTH > img.shape[2]:
                        d = img.shape[2] - INPUT_DEPTH
                    if h + INPUT_HEIGHT > img.shape[1]:
                        h = img.shape[1] - INPUT_HEIGHT
                    if w + INPUT_WIDTH > img.shape[0]:
                        w = img.shape[0] - INPUT_WIDTH
                    cub = img[w:w + INPUT_WIDTH, h:h + INPUT_HEIGHT, d:d + INPUT_DEPTH]
                    
                    if np.all(cub == ZERO_CENTER):
                        continue
                    
                    #batch_cub = cub[np.newaxis, ..., np.newaxis]
                    cubs.append(cub)
                    cub_sizes.append([w, h, d])
        for k in range(0,len(cub_sizes),16):
            t = 16
            if k + 16>= len(cub_sizes):
                t = len(cub_sizes) - k 
            
            batch_cub = np.array(cubs[k:t+k])
            batch_cub_sizes = cub_sizes[k:t+k]
            
            batch_cub = batch_cub[..., np.newaxis]
            pre_y_batch = detect_model.predict(batch_cub)
            for k in range(pre_y_batch.shape[0]):
                pre_y = pre_y_batch[k, :, :, :, 0] > seg_thresh_hold
                #print('predicted pix:'+ str(np.sum(pre_y)))
                if np.sum(pre_y) > 0:
                    crops, crop_centers,diameter,bboxes = crop_for_class(img, pre_y, np.array(batch_cub_sizes[k]))
                    #print('find:'+str(len(crop_centers)))
                    for i, center in enumerate(crop_centers):
                        crop = crops[i]
                        crop_cub = crop[np.newaxis,...,np.newaxis]
                        class_type = class_model.predict(crop_cub)
                        class_type= class_type[0]
                        index = np.argmax(class_type)
                        if index >0 :
                            #print('Add one')
                            location =  meta['origin']+center
                            new_row = pd.DataFrame([[meta['seriesuid'],location[0],location[1],location[2],
                                                          label_softmax_reverse[index],class_type[index]]], columns=columns)
                            df = df.append(new_row, ignore_index=True)
        df.to_csv('./output/predict_'+name+'_'+mode+'.csv', index=False)
    print('finished')
    
    
def predict_box(start,class_model,columns,df):
    step_w = CLASSIFY_INPUT_WIDTH
    step_h = CLASSIFY_INPUT_HEIGHT
    step_d = CLASSIFY_INPUT_DEPTH
    test_files_path = TEST_FOLDER+ "/*.mhd"
    test_files = glob.glob(test_files_path)
    total_step = len(test_files)
    print("total:"+str(total_step))
    pbar = tqdm(total=total_step)
    count =0
    for img, meta in get_test_file():
        pbar.update(1)
        for w in range(start[0], img.shape[0], step_w):
            for h in range(start[1], img.shape[1], step_h):
                for d in range(start[2], img.shape[2], step_d):
                    if d + step_d > img.shape[2]:
                        d = img.shape[2] - step_d - 1
                    if h + step_h > img.shape[1]:
                        h = img.shape[1] - step_h - 1
                    if w + step_w > img.shape[0]:
                        w = img.shape[0] - step_w - 1
                    
                    if count % 16 == 0:
                        X = np.zeros((16, CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
                        seriesuids = []
                        points = []
                    location = meta['origin'] + np.array([w + step_w / 2, h + step_h / 2, d + step_d / 2])
                    seriesuids.append(meta['seriesuid'])
                    points.append(location)
                    X[count%16, :, :, :, 0] = img[w:w + step_w, h:h + step_h, d:d + step_d]
                    
                    if (count % 16) == 15:
                        class_type = class_model.predict(X)
                        for k in range(class_type.shape[0]):
                            cur_class = class_type[k]
                            index = np.argmax(cur_class)
                            if index>0 and cur_class[index] > 0.5:
                                new_row = pd.DataFrame([[seriesuids[k], points[k][0], points[k][1], points[k][2],
                                                         label_softmax_reverse[index], cur_class[index]]],
                                                       columns=columns)
                                df = df.append(new_row, ignore_index=True)
                    
                    count= count+1
    return df
                    

def predict_test_only_classification():
    net = Resnet()
    name = 'Resnet'
    model = net.get_model()
    model.load_weights(CLASS_MODEL_PATH)

    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    df = pd.DataFrame(columns=columns)

    #Use two round to detect. same step,different start point
    print('Round 1')
    df = predict_box(np.array([16, 16, 16]), model, columns, df)
    print('Round 2')
    df = predict_box(np.array([0, 0, 0]), model, columns, df)

    df.to_csv('./output/result_only_class.csv', index=False)

def crop_roi_cub(cub,orign):
#     centers = []
#     size = 8
#     for w in range(0, 64, size):
#             for h in range(0, 64, size):
#                 for d in range(0, 64, size):
#                     small_cub = cub[w:w + size, h:h + size, d:d + size]
#                     binary = small_cub > np.percentile(small_cub,80)
#                     labels = measure.label(binary)
#                     regions = measure.regionprops(labels)
#                     labels = [(r.area, r.bbox) for r in regions]
                    
#                     if len(labels)>0:
#                         labels.sort(reverse=True)
#                         B = labels[0][1]
#                         #if B[3] - B[0] > 2 and B[4] - B[1] > 2 and B[5] - B[2] > 2: # ignore too small focus
#                         x = int((B[3] + B[0]) / 2.0)
#                         y = int((B[4] + B[1]) / 2.0)
#                         z = int((B[5] + B[2]) / 2.0)
#                         centers.append(np.array([x+w, y+h, z+d])+orign)
#     return centers
#     xs,ys,zs =np.where(cub > np.mean())
#     centers=[]
#     for i in range(len(xs)):
#         x = xs[i]
#         y = ys[i]
#         z = zs[i]
#         centers.append(np.array([x, y, z])+orign)
#     return centers
    binary = cub > 0
#     binary = morphology.dilation(binary, np.ones([2, 2, 2]))
#     binary = morphology.dilation(binary, np.ones([3, 3, 3]))
#     binary = morphology.erosion(binary, np.ones([2, 2, 2]))
    labels = measure.label(binary)
    regions = measure.regionprops(labels)
    centers = []
    for prop in regions:
        if prop.area > 100:
            B = prop.bbox
            #if B[3] - B[0] > 2 and B[4] - B[1] > 2 and B[5] - B[2] > 2: # ignore too small focus
            x = int((B[3] + B[0]) / 2.0)
            y = int((B[4] + B[1]) / 2.0)
            z = int((B[5] + B[2]) / 2.0)
            centers.append(np.array([x, y, z])+orign)
    return centers

def crop_for_class(img_arr,pre_y,orign,mean_val=-0.25):
    class_boundary = np.array([CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH])
#     pre_y = morphology.dilation(pre_y, np.ones([3, 3, 3]))
#     pre_y = morphology.dilation(pre_y, np.ones([3, 3, 3]))
#     pre_y = morphology.erosion(pre_y, np.ones([3, 3, 3]))
    labels = measure.label(pre_y, connectivity=2)
    regions = measure.regionprops(labels)
    centers = []
    bboxes= []
    spans = []
    crops= []
    crop_centers = []
    for prop in regions:
        B = prop.bbox
        if B[3] - B[0] > 2 and B[4] - B[1] > 2 and B[5] - B[2] > 2: # ignore too small focus
            x = int((B[3] + B[0]) / 2.0)
            y = int((B[4] + B[1]) / 2.0)
            z = int((B[5] + B[2]) / 2.0)
            span = np.array([int(B[3] - B[0]), int(B[4] - B[1]), int(B[5] - B[2])])
            
            bcub = img_arr[B[0]+orign[0]:B[3]+orign[0],B[1]+orign[1]:B[4]+orign[1],B[2]+orign[2]:B[5]+orign[2]]
#             if np.mean(bcub) < mean_val:
#                 continue
            
            spans.append(span)
            centers.append(np.array([x, y, z]))
            bboxes.append(B)
    for idx, bbox in enumerate(bboxes):
        crop = np.zeros(class_boundary, dtype=np.float32)
        crop_center = centers[idx]
        crop_center = crop_center + orign
        half = class_boundary / 2
        crop_center = check_center(class_boundary, crop_center, img_arr.shape)
        crop = img_arr[int(crop_center[0] - half[0]):int(crop_center[0] + half[0]), \
               int(crop_center[1] - half[1]):int(crop_center[1] + half[1]), \
               int(crop_center[2] - half[2]):int(crop_center[2] + half[2])]
        
        crops.append(crop)
        crop_centers.append(crop_center)
    return crops,crop_centers,spans,bboxes

def generate_detect_result(name='lung',mode='test',model_path=SEG_LUNG_TRAIN_WEIGHT,thresh_hold=0.8,limit = [0,0]):
    detect_net = UNet()
    detect_model = detect_net.get_model()
    detect_model.load_weights(model_path)
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ' ,'diameterX','diameterY','diameterZ']
    df = pd.DataFrame(columns=columns)
    for img, meta in get_files(name,mode):
        for w in range(limit[0], img.shape[0]-limit[0], INPUT_WIDTH):
            for h in range(limit[1], img.shape[1]-limit[0], INPUT_HEIGHT):
                for d in range(0, img.shape[2], INPUT_DEPTH):
                    if d + INPUT_DEPTH > img.shape[2]:
                        d = img.shape[2] - INPUT_DEPTH
                    if h + INPUT_HEIGHT > img.shape[1]:
                        h = img.shape[1] - INPUT_HEIGHT
                    if w + INPUT_WIDTH > img.shape[0]:
                        w = img.shape[0] - INPUT_WIDTH
                    
                    cub = img[w:w + INPUT_WIDTH, h:h + INPUT_HEIGHT, d:d + INPUT_DEPTH]
                    
                    batch_cub = cub[np.newaxis, ..., np.newaxis]
                    pre_y = detect_model.predict(batch_cub)
                    pre_y = pre_y[0, :, :, :, 0] > thresh_hold
                    #print('predicted pix:'+ str(np.sum(pre_y)))
                    if np.sum(pre_y) > 0:
                        crops, crop_centers,diameter,bboxes = crop_for_class(img, pre_y, np.array([w, h, d]),mean_val)
                        print('find:'+str(len(crop_centers)))
                        for i, center in enumerate(crop_centers):
                            #location = meta['origin']+center
                            location = center
                            print(center)
                            
                            new_row = pd.DataFrame([[meta['seriesuid'], location[0], location[1], location[2],diameter[i][0],diameter[i][1],diameter[i][2]]],columns=columns)
                            df = df.append(new_row, ignore_index=True)
                               
    df.to_csv('./output/predict_'+name+'_'+mode+'.csv', index=False)
    print('finished')

def check_detect_result_accuracy(name='lung',model_path=SEG_LUNG_TRAIN_WEIGHT,thresh_hold=0.8,limit = [0,0]):
    mode='train' 
    df = pd.read_csv(ANNOTATION_FILE)
    detect_net = UNet()
    detect_model = detect_net.get_model()
    detect_model.load_weights(model_path)
    count = 0
    postive_focus= []
    negative_focus = []
    total_focus = 0
    postive_focus_set  =set()
    for img, meta in get_files(name,mode):
        if count == 10:
            break
        count+=1
        seriesuid = meta['seriesuid']
        origin = meta['origin']
        
        if name == 'lung':
            focus_records = df[(df['seriesuid'] == int(seriesuid)) & ((df['label'] == 1) | (df['label'] == 5))]
        else:
            focus_records = df[(df['seriesuid'] == int(seriesuid)) & (df['label'] > 5 )]
        
        total_focus += focus_records.shape[0]
        focus_records['coordX'] = focus_records['coordX'] - origin[0] 
        focus_records['coordY'] = focus_records['coordY'] - origin[1] 
        focus_records['coordZ'] = focus_records['coordZ'] - origin[2]
        focus_records['radiusZ'] = focus_records['diameterZ']//2 
        focus_records['radiusY'] = focus_records['diameterY']//2 
        focus_records['radiusX'] = focus_records['diameterX']//2 
        
        step = 32
        for w in range(limit[0], img.shape[0]-limit[0], step):
            for h in range(limit[1], img.shape[1]-limit[0], step):
                for d in range(0, img.shape[2], step):
                    if d + INPUT_DEPTH > img.shape[2]:
                        d = img.shape[2] - INPUT_DEPTH
                    if h + INPUT_HEIGHT > img.shape[1]:
                        h = img.shape[1] - INPUT_HEIGHT
                    if w + INPUT_WIDTH > img.shape[0]:
                        w = img.shape[0] - INPUT_WIDTH
                    cub = img[w:w + INPUT_WIDTH, h:h + INPUT_HEIGHT, d:d + INPUT_DEPTH]
                    mean_val = np.percentile(cub,80)
                    batch_cub = cub[np.newaxis, ..., np.newaxis]
                    pre_y = detect_model.predict(batch_cub)
                    pre_y = pre_y[0, :, :, :, 0] > thresh_hold
                    
                    if np.sum(pre_y) > 0:
                        crops, crop_centers,diameter,bboxes = crop_for_class(img, pre_y, np.array([w, h, d]))
                        #crop_centers_roi = crop_roi_cub(cub,np.array([w, h, d]))
                        #print("Found ROI",len(crop_centers_roi))
                        #crop_centers = crop_centers_roi + crop_centers 
                        for i, center in enumerate(crop_centers):
                            found_focus = False
                            distances = []
                            for fi,focus in focus_records.iterrows():
                                anno_focus_center =  np.array([focus['coordX'],focus['coordY'] ,focus['coordZ'] ])
                                #distances.append(np.linalg.norm(center-anno_focus_center))
                                if center[2] >= (focus['coordZ'] - focus['radiusZ']) and center[2]  <= (focus['coordZ'] +  focus['radiusZ']):
                                    if center[0] >= (focus['coordX'] - focus['radiusX']) and center[0]  <= (focus['coordX'] +  focus['radiusX']):
                                        if center[1] >= (focus['coordY'] - focus['radiusY']) and center[1]  <= (focus['coordY'] +  focus['radiusY']):
                                            
                                            postive_focus_set.add(str(seriesuid)+'_'+str(fi)+'_'+str(focus['label']))
                                            found_focus = True
                            if found_focus:
                                postive_focus.append(center)
                            else:
                                #print(min(distances))
                                negative_focus.append(center)
    
    print('Found Right Focus:'+str(len(postive_focus_set)))
    print('Found Wrong Focus:'+str(len(negative_focus)))
    print('Total Ground-truth Focus:'+str(total_focus))
    print('finished')
    return postive_focus_set
    
def check_center(size,crop_center,image_shape):
    '''
    @size：所切块的大小
    @crop_center：待检查的切块中心
    @image_shape：原图大小
    Return：检查修正后切块中心
    '''
    half=size/2
    margin_min=crop_center-half#检查下界
    margin_max=crop_center+half-image_shape#检查上界
    for i in range(3):#如有超出，对中心进行修正
        if margin_min[i]<0:
            crop_center[i]=crop_center[i]-margin_min[i]
        if margin_max[i]>0:
            crop_center[i]=crop_center[i]-margin_max[i]
    return crop_center

def get_files(focus_type,mode):
    orgin_folder = TRAIN_FOLDER if mode == 'train' else TEST2_FOLDER
    process_parent_folder = PREPROCESS_PATH if mode == 'train' else TEST2_PROCESS_PATH
    processed_folder = process_parent_folder+'/lung' if focus_type == 'lung' else process_parent_folder+'/mediastinal'
    
    test_files = orgin_folder + "/*.mhd"
    files = glob.glob(test_files)
    print('total:'+str(len(files)))
    for index,file in enumerate(files):
        seriesuid = os.path.splitext(os.path.basename(file))[0]
        print('process:'+str(index)+', seriesuid:'+seriesuid)
        h5_file = processed_folder+"/"+seriesuid+".h5"
        meta_file = process_parent_folder+'/meta'+"/"+seriesuid+".meta"
       
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        ret_img = None
        with h5py.File(h5_file, 'r') as hf:
            ret_img = hf['img'].value
            
        yield ret_img, meta

if __name__ == '__main__':
    generate_false_positive()
