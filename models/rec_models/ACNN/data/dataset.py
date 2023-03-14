# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data
from common.utils import *
import numpy as np
from torchvision import transforms as T
import ipdb
import scipy.io as sio
import cv2

class Mridata(data.Dataset):

    def __init__(self, root, mask_smp_path, slice_num, transforms=None, test=False):
        self.test = test
        self.dir_list = os.listdir(root)
        self.slice_num = slice_num
        self.dir_index = list(range(1,18)) if not self.test else list(range(18,20))        
        self.imgs = [os.path.join(root, str(index), img) for index in self.dir_index for img in os.listdir(os.path.join(root, str(index)))]
        self.fix_list()

        self.mask_smp = cv2.resize(sio.loadmat(mask_smp_path)['smp_mask'], dsize=(256, 320), interpolation=cv2.INTER_CUBIC)

        self.fft = fft2
        self.ifft = ifft2
        self.scale = 0.1
        self.rng = np.random.RandomState(0)
    
    def fix_list(self):
        for img in self.imgs:
            index = int(img.split('/')[-1].split('.')[0])
            boundary = self.slice_num
            if index <= boundary or index > 256 - boundary:
                self.imgs.remove(img)

    def __getitem__(self, index):
        success = False
        while not success:
            try:
                data, label, sen_mat, dc_fft = self.getitem(index)
                success = True
            except Exception as e:
                print(e)
                print(self.imgs[index])
                index = self.rng.choice(range(0, self.__len__()))
        return data, label, sen_mat, dc_fft
    
    def read_mat(self, img_path):
        mat = sio.loadmat(img_path)['single_slice']
        mat = np.transpose(mat, (2, 0, 1))
        real_mat = mat.real
        imag_mat = mat.imag
        mat_input = np.concatenate((real_mat[:,:,:,np.newaxis], imag_mat[:,:,:,np.newaxis]), 3)
        label = mat_input
        data = self.ifft(self.fft(t.from_numpy(mat_input)) * t.from_numpy(self.mask_smp.astype(np.float32)[:,:,np.newaxis]))
        #data = self.ifft(self.fft(t.from_numpy(mat_input)) * t.from_numpy(self.mask_smp.reshape(320,256,1))) #Not work
        #data = self.ifft(self.fft(t.from_numpy(mat_input)) * t.from_numpy(self.mask_smp.astype(np.float32).reshape(320,256,1))) #Not work

        label = t.from_numpy(label)
        
        data = data * self.scale
        label = label * self.scale

        dc_fft = data[:,160,128,:]     
        dc_fft = t.cat((dc_fft[:,0], dc_fft[:,1]), 0)        

        sen_mat = t.tensor([0])
        if self.test:
            sen_index = img_path.rfind('/')
            sen_path = img_path[:sen_index] + 's' + img_path[sen_index:]
            sen_mat = sio.loadmat(sen_path)['sensitivity']        
            sen_mat = np.transpose(sen_mat, (2, 0, 1))
            real_mat = sen_mat.real
            imag_mat = sen_mat.imag
            sen_mat = np.concatenate((real_mat[:,:,:,np.newaxis], imag_mat[:,:,:,np.newaxis]), 3)
            sen_mat = t.from_numpy(sen_mat)
        return data, label, sen_mat, dc_fft

    def getitem(self, index):
        if os.path.exists('/tmp/debug'):
            ipdb.set_trace()
        img_path = self.imgs[index]
        slide_index = img_path.rfind('/')
        tmp_path = img_path[:slide_index]
        slice_index = int(img_path.split('/')[-1].split('.')[0])
        
        data_list = []
        label_list = []
        sen_mat_list = []
        dc_fft_list = []
        boundary = self.slice_num // 2
        for i in range(-boundary, boundary + 1):
            if slice_index + i <= 0:
                new_index = 1
            elif slice_index + i > 256 :
                new_index = 256
            else:
                new_index = slice_index + i

            i_path = tmp_path + '/' + str(new_index) + '.mat'
            data, label, sen_mat, dc_fft = self.read_mat(i_path)
            data_list.append(data)
            label_list.append(label)
            sen_mat_list.append(sen_mat)
            dc_fft_list.append(dc_fft)

        multi_data = t.cat(data_list, 0)
        multi_label = t.cat(label_list, 0)
        multi_sen_mat = t.cat(sen_mat_list, 0)
        multi_dc_fft = t.cat(dc_fft_list, 0)
        return multi_data, multi_label, multi_sen_mat, multi_dc_fft

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
	dataset = Mridata("/data/dutia/python_mri", 1)

