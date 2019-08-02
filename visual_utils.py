# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PLOT_NUM = 16
class VisualUtil():
    
    @staticmethod
    def plot_all_slices(img, title='', box=None):
        print(title)
        img.shape[2]
        fig, axs = plt.subplots(img.shape[2]//4,4, figsize=(16, img.shape[2]//4*4), sharex=True, sharey=True)
        for index,ax in enumerate(axs.flat):
            ax.imshow(np.transpose(img[:,:,index],(1,0)), cmap='gray')
    @staticmethod
    def plot_all_slices_with_mask(img,focus_records, title=''):
        print(title)
        fig, axs = plt.subplots(img.shape[2]//4,4, figsize=(16, img.shape[2]//4*4), sharex=True, sharey=True)
        for index,ax in enumerate(axs.flat):
            ax.imshow(np.transpose(img[:,:,index],(1,0)), cmap='gray')
            for fi,focus in focus_records.iterrows():
                radiusZ = focus['diameterZ']//2 
                radiusY = focus['diameterY']//2 
                radiusX = focus['diameterX']//2 
                #if index == focus['coordZ']:
                if index >= (focus['coordZ'] - radiusZ) and index <= (focus['coordZ'] + radiusZ):
                    ax.add_patch(patches.Rectangle((focus['coordX']-radiusX, focus['coordY']-radiusY),
                                                   focus['diameterX'],focus['diameterY'], 
                                                   linewidth=1,edgecolor='r',facecolor='none'))
       
    @staticmethod
    def plot_slices(img, title='', box=None):
        print(title)

        fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
        c, c_step = 0, img.shape[2] // PLOT_NUM
        #
        # c = img.shape[2] // 4
        # c_step = c_step // 2
        for ax in axs.flat:
            ax.imshow(img[:,:,c], cmap='gray')

            if box:
                ax.add_patch(patches.Rectangle((box['x'], box['y']),box['w'] * 4,box['h'] * 4, linewidth=1,edgecolor='r',facecolor='none'))
            c += c_step

        axs[0,0].set(title=title)
        plt.show()

    @staticmethod
    def plot_middle_slices_comparison(imgs):
        shape = None
        for img in imgs:
            if shape is None:
                shape = img.shape
            else:
                if shape != img.shape:
                    print('plot_middle_slices_comparison with images have different size, former {}, now {}'.format(shape, img.shape))
                    return

        l = len(imgs)
        row = 3
        fig, axs = plt.subplots(row, l, figsize=(10, 15), sharex=True, sharey=True)
        for r in range(row):
            for i in range(l):
                offset = (r - 1) * 3
                depth = int(imgs[i].shape[2] / 2 + offset)
                axs[r][i].imshow(imgs[i][:, :, depth], cmap='gray')

        plt.show()

    @staticmethod
    def plot_comparison(X, y, pred, title='', box=None):
        print(title)

        assert X.shape[2] == y.shape[2] == pred.shape[2]
        z = X.shape[2] // 2

        fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharex=True, sharey=True)
        axs[0].imshow(X[:,:,z], cmap='gray')
        axs[1].imshow(y[:,:,z], cmap='gray')
        axs[2].imshow(pred[:,:,z], cmap='gray')

        if box:
            rec = patches.Rectangle((box['x'], box['y']), box['w'] * 4, box['h'] * 4, linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(rec)
            axs[1].add_patch(rec)
            axs[2].add_patch(rec)

        axs[0].set(title='X')
        axs[1].set(title='y')
        axs[2].set(title='pred')
        plt.show()
