from __future__ import print_function, division

import torch
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import math

import random
import numpy as np
import warnings
import torch.utils.data as data
from PIL import Image
plt.ion()    # interactive mode
warnings.filterwarnings('ignore')

# patch_dim = 32
# gap = 8
# jitter = 8
# gray_portion = .30
# reuse_image_count = 4

patch_order_arr = [
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (1, 2, 0, 3),
    (1, 2, 3, 0),
    (1, 3, 0, 2),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (2, 0, 3, 1),
    (2, 1, 0, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 0, 1, 2),
    (3, 0, 2, 1),
    (3, 1, 0, 2),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0)
]

class PlainDataset(Dataset):

    def __init__(self, split, labelled_root_dir, preTransform=None, postTransform=None):
        self.labelled_root_dir = labelled_root_dir
        # Output of pretransform should be PIL images
        self.preTransform = preTransform
        self.postTransform = postTransform
        self.split = split
        self.labelled_data_dir = labelled_root_dir + '/' + split
        self.dataset = datasets.ImageFolder(self.labelled_data_dir, self.preTransform) 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        plain_img, plain_class = self.dataset[index]
        if self.postTransform:
            sample = self.postTransform(plain_img)
        else:
            sample = transforms.ToTensor(plain_img)
        return sample, plain_class

class PrimeRotationDataset(Dataset):

    def __init__(self, split, labelled_root_dir, preTransform=None, postTransform=None):
        self.labelled_root_dir = labelled_root_dir
        # Output of pretransform should be PIL images
        self.preTransform = preTransform
        self.postTransform = postTransform
        self.split = split
        self.labelled_data_dir = labelled_root_dir + '/' + split
        self.dataset = datasets.ImageFolder(self.labelled_data_dir, self.preTransform) 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_PIL, _ = self.dataset[index]
        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_img = img_PIL.rotate(rot_angle)
        if self.postTransform:
            sample = self.postTransform(rot_img)
        else:
            sample = transforms.ToTensor(rot_img)
        return sample, rot_class

class PrimePatchDataset(Dataset):
    
    def __init__(self, split, root_paths, patch_dim, gap, jitter, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split

        self.patch_dim = patch_dim
        self.gap = gap
        self.jitter = jitter

        self.margin = math.ceil(self.patch_dim/2.0) + self.jitter
        self.min_width = 2*self.patch_dim + 2*self.jitter + 2*self.gap

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)
    
    def prep_patch(self, image):

        # for some patches, randomly downsample to as little as 100 total pixels
        # 说是要下采样，结果放大后又缩小？迷惑行为
        # 可能可以添加抖动
        if(random.random() < .33):
            pil_patch = Image.fromarray(image)
            original_size = pil_patch.size
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            pil_patch = pil_patch.resize((randpix, randpix)) 
            pil_patch = pil_patch.resize(original_size) 
            np.copyto(image, np.array(pil_patch))

        # randomly drop all but one color channel
        # 看起来就是毫无意义的增加学习难度
        # 垃圾玩意，删了
        # chan_to_keep = random.randint(0, 2)
        # for i in range(0, 3):
        #     if i != chan_to_keep:
        #         image[:,:,i] = np.random.randint(0, 255, (self.patch_dim, self.patch_dim), dtype=np.uint8)

    def __getitem__(self, index):
        # [y, x, chan], dtype=uint8, top_left is (0,0)
        patch_loc_arr = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        # image_index = int(math.floor((len(self.dataset) * random.random())))
        # pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
        # pil_image = datasets.ImageFolder(self.image_paths, self.preTransform)
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[1] <= self.min_width or image.shape[0] <= self.min_width:
            return self.__getitem__(index)
        
        patch_direction_label = int(math.floor((8 * random.random())))
        patch_jitter_y = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
        patch_jitter_x = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
                
        while True:
                
            uniform_patch_x_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            uniform_patch_y_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            random_patch_y_coord = uniform_patch_x_coord + patch_loc_arr[patch_direction_label][0] * (self.patch_dim + self.gap) + patch_jitter_y
            random_patch_x_coord = uniform_patch_y_coord + patch_loc_arr[patch_direction_label][1] * (self.patch_dim + self.gap) + patch_jitter_x

            if random_patch_y_coord>=0 and random_patch_x_coord>=0 and random_patch_y_coord+self.patch_dim<image.shape[0] and random_patch_x_coord+self.patch_dim<image.shape[1]:
                break
        
        uniform_patch = image[
            uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
            uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
        ]                
        random_patch = image[
            random_patch_y_coord : random_patch_y_coord + self.patch_dim, 
            random_patch_x_coord : random_patch_x_coord + self.patch_dim
        ]
        # 非必要模块：随机图片抖动
        # self.prep_patch(uniform_patch)
        # self.prep_patch(random_patch)
        if self.preTransform:
            uniform_patch = self.postTransform(uniform_patch)
            random_patch = self.postTransform(random_patch)
        else:
            uniform_patch = transforms.ToTensor(uniform_patch)
            random_patch = transforms.ToTensor(random_patch)

        patch_direction_label = np.array(patch_direction_label).astype(np.int64)
        return uniform_patch, random_patch, patch_direction_label

class JigsawPatchDataset(Dataset):

    def __init__(self, split, root_paths, patch_dim, gap, jitter, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split

        self.patch_dim = patch_dim
        self.gap = gap
        self.jitter = jitter

        self.gray_portion = 0.3
        self.color_shift = 2
        self.margin = math.ceil((2*patch_dim + 2*jitter + 2*self.color_shift + gap)/2)
        self.min_width = 2 * self.margin + 1
        
        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)
    
    def half_gap(self):
        return math.ceil(self.gap/2)

    def random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random()))) - self.jitter

    def random_shift(self):
        return random.randrange(self.color_shift * 2 + 1)

    # crops the patch by self.color_shift on each side
    def prep_patch(self, image, gray):
 
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

        if(gray):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift : self.color_shift + self.patch_dim, 
                self.color_shift : self.color_shift + self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]
        
        return cropped

    def __getitem__(self, index):
        # [y, x, chan], dtype=uint8, top_left is (0,0)
        # image_index = int(math.floor((len(self.image_paths) * random.random())))
        # pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[0] <= self.min_width or image.shape[1] <= self.min_width:
                return self.__getitem__(index)
        
        center_y_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin
        center_x_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin
        patch_coords = [
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.random_jitter() + self.color_shift),
                center_x_coord - (self.patch_dim + self.half_gap() + self.random_jitter() + self.color_shift)
            ),
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.random_jitter() + self.color_shift),
                center_x_coord + self.half_gap() + self.random_jitter() - self.color_shift
            ),
            (
                center_y_coord + self.half_gap() + self.random_jitter() - self.color_shift,
                center_x_coord - (self.patch_dim + self.half_gap() + self.random_jitter() + self.color_shift)
            ),
            (
                center_y_coord + self.half_gap() + self.random_jitter() - self.color_shift,
                center_x_coord + self.half_gap() + self.random_jitter() - self.color_shift
            )
        ]
        
        patch_shuffle_order_label = int(math.floor((24 * random.random())))
        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[patch_shuffle_order_label],patch_coords))]

        patch_a = image[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = image[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = image[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = image[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        gray = random.random() < self.gray_portion
        patch_a = self.prep_patch(patch_a, gray)
        patch_b = self.prep_patch(patch_b, gray)
        patch_c = self.prep_patch(patch_c, gray)
        patch_d = self.prep_patch(patch_d, gray)

        patch_shuffle_order_label = np.array(patch_shuffle_order_label).astype(np.int64)

        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label

class JigsawRotationDataset(Dataset):

    def __init__(self, split, root_paths, patch_dim, jitter, preTransform=None, postTransform=None):
        # (self, image_paths, patch_dim, length, jitter, color_shift, transform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split
        self.patch_dim = patch_dim
        self.jitter = jitter

        self.gray_portion = 0.3
        self.color_shift = 2
        self.image_reused = 0
        self.reuse_image_count = 4

        self.sub_window_width = self.patch_dim + 2*self.jitter + 2*self.color_shift
        self.window_width = 2*self.sub_window_width
        # self.min_image_width = self.window_width
        
        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)

    def random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random())))

    def random_shift(self):
        return random.randrange(self.color_shift * 2 + 1)

    def prep_patch(self, image):
 
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

        if(random.random() < self.gray_portion):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift:self.color_shift+self.patch_dim, 
                self.color_shift:self.color_shift+self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]

        return cropped


    def __getitem__(self, index):
        # [y, x, chan], dtype=uint8, top_left is (0,0)
        # image_index = int(math.floor((len(self.image_paths) * random.random())))
        
        # if self.image_reused == 0:
        #     # pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
        #     # if pil_image.shape[1] > pil_image.shape[0]:
        #     #     self.pil_image = pil_image.resize((self.min_image_width, int(round(pil_image.shape[1]/pil_image.shape[0] * self.min_image_width))))
        #     # else:
        #     #     self.pil_image = pil_image.resize((int(round(pil_image.shape[0]/pil_image.shape[1] * self.min_image_width)), self.min_image_width))
        #     img_PIL, _ = self.dataset[index]
        #     self.image_reused = self.reuse_image_count - 1
        # else:
        #     self.image_reused -= 1
        # 每张图片调用 4 次
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)

        window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
        window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))
        window = image[window_y_coord:window_y_coord+self.window_width, window_x_coord:window_x_coord+self.window_width]
        
        rotation_label = int(math.floor((4 * random.random())))
        order_label = int(math.floor((24 * random.random()))) 
        
        if rotation_label>0:
            window = np.rot90(window, rotation_label).copy()

        patch_coords = [
            (self.random_jitter(), self.random_jitter()),
            (self.random_jitter(), self.sub_window_width + self.random_jitter()),
            (self.sub_window_width + self.random_jitter(), self.random_jitter()),
            (self.sub_window_width + self.random_jitter(), self.sub_window_width + self.random_jitter()),
        ]

        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[order_label],patch_coords))]

        patch_a = window[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = window[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = window[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = window[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        patch_a = self.prep_patch(patch_a)
        patch_b = self.prep_patch(patch_b)
        patch_c = self.prep_patch(patch_c)
        patch_d = self.prep_patch(patch_d)

        # combined_label = np.array(rotation_label * 24 + order_label).astype(np.int64)
        combined_label = np.array(order_label).astype(np.int64)
                
        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, combined_label

### 警告：施工现场 ###
class ContrastiveDataset(Dataset):
    
    def __init__(self, split, root_paths, patch_dim, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split

        self.patch_dim = patch_dim

        self.margin = math.ceil(self.patch_dim/2.0)
        self.min_width = 2*self.patch_dim

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

    def __len__(self):
        return len(self.dataset)
    
    def prep_patch(self, image):

        # for some patches, randomly downsample to as little as 100 total pixels
        # 说是要下采样，结果放大后又缩小？迷惑行为
        # 可能可以添加抖动
        if(random.random() < .33):
            pil_patch = Image.fromarray(image)
            original_size = pil_patch.size
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            pil_patch = pil_patch.resize((randpix, randpix)) 
            pil_patch = pil_patch.resize(original_size) 
            np.copyto(image, np.array(pil_patch))

    def __getitem__(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[1] <= self.min_width or image.shape[0] <= self.min_width:
            return self.__getitem__(index)

        patch_direction_label = np.random.randint(2)
                
        if self.postTransform:
            uniform_patch = self.postTransform(image)
            random_patch = self.postTransform(image)

        return uniform_patch, random_patch, patch_direction_label

### 警告：施工现场 ###
class JointDataset(Dataset):
    
    def __init__(self, split, root_paths, patch_dim, gap, jitter, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split

        self.patch_dim = patch_dim
        self.gap = gap
        self.jitter = jitter

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)

        self.margin = math.ceil(self.patch_dim/2.0) + self.jitter
        self.min_width = 2*self.patch_dim + 2*self.jitter + 2*self.gap

        self.gray_portion = 0.3
        self.color_shift = 2
        self.jigmargin = math.ceil((2*patch_dim + 2*jitter + 2*self.color_shift + gap)/2)
        self.jigmin_width = 2 * self.margin + 1

        self.gray_portion = 0.3
        self.color_shift = 2
        self.image_reused = 0
        self.reuse_image_count = 4

        self.sub_window_width = self.patch_dim + 2*self.jitter + 2*self.color_shift
        self.window_width = 2*self.sub_window_width

    def __len__(self):
        return len(self.dataset)
    
    def half_gap(self):
        return math.ceil(self.gap/2)

    def jigro_random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random())))

    def jigpa_random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random()))) - self.jitter

    def random_shift(self):
        return random.randrange(self.color_shift * 2 + 1)
    
    def prep_patch(self, image):
        # for some patches, randomly downsample to as little as 100 total pixels
        # 说是要下采样，结果放大后又缩小？迷惑行为
        # 可能可以添加抖动
        if(random.random() < .33):
            pil_patch = Image.fromarray(image)
            original_size = pil_patch.size
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            pil_patch = pil_patch.resize((randpix, randpix)) 
            pil_patch = pil_patch.resize(original_size) 
            np.copyto(image, np.array(pil_patch))
    
    def prep_jigpa(self, image, gray):
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)
        if(gray):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift : self.color_shift + self.patch_dim, 
                self.color_shift : self.color_shift + self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]
        
        return cropped
    
    def prep_jigro(self, image):
 
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

        if(random.random() < self.gray_portion):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift:self.color_shift+self.patch_dim, 
                self.color_shift:self.color_shift+self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]

        return cropped

    def get_plain(self, index):
        plain_img, plain_class = self.dataset[index]
        if self.postTransform:
            sample = self.postTransform(plain_img)
        else:
            sample = transforms.ToTensor(plain_img)
        return sample, plain_class
    
    def get_rota(self, index):
        img_PIL, _ = self.dataset[index]
        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_img = img_PIL.rotate(rot_angle)
        if self.postTransform:
            sample = self.postTransform(rot_img)
        else:
            sample = transforms.ToTensor(rot_img)
        return sample, rot_class
    
    def get_patch(self, index):
        patch_loc_arr = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[1] <= self.min_width or image.shape[0] <= self.min_width:
            return self.__getitem__(index)
        
        patch_direction_label = int(math.floor((8 * random.random())))
        patch_jitter_y = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
        patch_jitter_x = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
                
        while True:
                
            uniform_patch_x_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            uniform_patch_y_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            random_patch_y_coord = uniform_patch_x_coord + patch_loc_arr[patch_direction_label][0] * (self.patch_dim + self.gap) + patch_jitter_y
            random_patch_x_coord = uniform_patch_y_coord + patch_loc_arr[patch_direction_label][1] * (self.patch_dim + self.gap) + patch_jitter_x

            if random_patch_y_coord>=0 and random_patch_x_coord>=0 and random_patch_y_coord+self.patch_dim<image.shape[0] and random_patch_x_coord+self.patch_dim<image.shape[1]:
                break
        
        uniform_patch = image[
            uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
            uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
        ]                
        random_patch = image[
            random_patch_y_coord : random_patch_y_coord + self.patch_dim, 
            random_patch_x_coord : random_patch_x_coord + self.patch_dim
        ]
        # 非必要模块：随机图片抖动
        # self.prep_patch(uniform_patch)
        # self.prep_patch(random_patch)
        if self.preTransform:
            uniform_patch = self.postTransform(uniform_patch)
            random_patch = self.postTransform(random_patch)
        else:
            uniform_patch = transforms.ToTensor(uniform_patch)
            random_patch = transforms.ToTensor(random_patch)

        patch_direction_label = np.array(patch_direction_label).astype(np.int64)
        return uniform_patch, random_patch, patch_direction_label
    
    def get_jigpa(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[0] <= self.jigmin_width or image.shape[1] <= self.jigmin_width:
                return self.__getitem__(index)
        
        center_y_coord = int(math.floor((image.shape[0] - self.jigmargin*2) * random.random())) + self.jigmargin
        center_x_coord = int(math.floor((image.shape[1] - self.jigmargin*2) * random.random())) + self.jigmargin
        patch_coords = [
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift),
                center_x_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift)
            ),
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift),
                center_x_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift
            ),
            (
                center_y_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift,
                center_x_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift)
            ),
            (
                center_y_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift,
                center_x_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift
            )
        ]
        
        patch_shuffle_order_label = int(math.floor((24 * random.random())))
        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[patch_shuffle_order_label],patch_coords))]

        patch_a = image[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = image[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = image[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = image[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        gray = random.random() < self.gray_portion
        patch_a = self.prep_jigpa(patch_a, gray)
        patch_b = self.prep_jigpa(patch_b, gray)
        patch_c = self.prep_jigpa(patch_c, gray)
        patch_d = self.prep_jigpa(patch_d, gray)

        patch_shuffle_order_label = np.array(patch_shuffle_order_label).astype(np.int64)

        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label
    
    def get_jigro(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)

        window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
        window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))
        window = image[window_y_coord:window_y_coord+self.window_width, window_x_coord:window_x_coord+self.window_width]
        
        rotation_label = int(math.floor((4 * random.random())))
        order_label = int(math.floor((24 * random.random()))) 
        
        if rotation_label>0:
            window = np.rot90(window, rotation_label).copy()

        patch_coords = [
            (self.jigro_random_jitter(), self.jigro_random_jitter()),
            (self.jigro_random_jitter(), self.sub_window_width + self.jigro_random_jitter()),
            (self.sub_window_width + self.jigro_random_jitter(), self.jigro_random_jitter()),
            (self.sub_window_width + self.jigro_random_jitter(), self.sub_window_width + self.jigro_random_jitter()),
        ]

        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[order_label],patch_coords))]

        patch_a = window[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = window[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = window[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = window[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        patch_a = self.prep_jigro(patch_a)
        patch_b = self.prep_jigro(patch_b)
        patch_c = self.prep_jigro(patch_c)
        patch_d = self.prep_jigro(patch_d)

        # combined_label = np.array(rotation_label * 24 + order_label).astype(np.int64)
        combined_label = np.array(order_label).astype(np.int64)
                
        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, combined_label

    def __getitem__(self, index):
        return self.get_plain(index), self.get_rota(index), self.get_patch(index), self.get_jigpa(index), self.get_jigro(index)

### 警告：施工现场 ###
class DJDataset(Dataset):
    
    def __init__(self, split, root_paths, patch_dim, gap, jitter, preTransform=None, postTransform=None):
        self.root_paths = root_paths
        self.image_paths = root_paths + '/' + split
        self.valid_paths = root_paths + '/valid'

        self.patch_dim = patch_dim
        self.gap = gap
        self.jitter = jitter

        self.preTransform = preTransform
        self.postTransform = postTransform
        self.dataset = datasets.ImageFolder(self.image_paths, self.preTransform)
        self.validataset = datasets.ImageFolder(self.valid_paths, self.preTransform)

        self.margin = math.ceil(self.patch_dim/2.0) + self.jitter
        self.min_width = 2*self.patch_dim + 2*self.jitter + 2*self.gap

        self.gray_portion = 0.3
        self.color_shift = 2
        self.jigmargin = math.ceil((2*patch_dim + 2*jitter + 2*self.color_shift + gap)/2)
        self.jigmin_width = 2 * self.margin + 1

        self.gray_portion = 0.3
        self.color_shift = 2
        self.image_reused = 0
        self.reuse_image_count = 4

        self.sub_window_width = self.patch_dim + 2*self.jitter + 2*self.color_shift
        self.window_width = 2*self.sub_window_width

    def __len__(self):
        return len(self.dataset)
    
    def half_gap(self):
        return math.ceil(self.gap/2)

    def jigro_random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random())))

    def jigpa_random_jitter(self):
        return int(math.floor((self.jitter * 2 * random.random()))) - self.jitter

    def random_shift(self):
        return random.randrange(self.color_shift * 2 + 1)
    
    def prep_patch(self, image):
        # for some patches, randomly downsample to as little as 100 total pixels
        # 说是要下采样，结果放大后又缩小？迷惑行为
        # 可能可以添加抖动
        if(random.random() < .33):
            pil_patch = Image.fromarray(image)
            original_size = pil_patch.size
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            pil_patch = pil_patch.resize((randpix, randpix)) 
            pil_patch = pil_patch.resize(original_size) 
            np.copyto(image, np.array(pil_patch))
    
    def prep_jigpa(self, image, gray):
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)
        if(gray):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift : self.color_shift + self.patch_dim, 
                self.color_shift : self.color_shift + self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]
        
        return cropped
    
    def prep_jigro(self, image):
 
        cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

        if(random.random() < self.gray_portion):
            pil_patch = Image.fromarray(image)
            pil_patch = pil_patch.convert('L')
            pil_patch = pil_patch.convert('RGB')
            np.copyto(cropped, np.array(pil_patch)[
                self.color_shift:self.color_shift+self.patch_dim, 
                self.color_shift:self.color_shift+self.patch_dim, 
                :])
        else:
            shift = [self.random_shift() for _ in range(6)]
            cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
            cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
            cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]

        return cropped

    def get_plain(self, index):
        plain_img, plain_class = self.dataset[index]
        if self.postTransform:
            sample = self.postTransform(plain_img)
        else:
            sample = transforms.ToTensor(plain_img)
        return sample, plain_class

    def get_valid(self, index):
        index = index % len(self.validataset)
        plain_img, plain_class = self.validataset[index]
        if self.postTransform:
            sample = self.postTransform(plain_img)
        else:
            sample = transforms.ToTensor(plain_img)
        return sample, plain_class
    
    def get_rota(self, index):
        img_PIL, _ = self.dataset[index]
        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_img = img_PIL.rotate(rot_angle)
        if self.postTransform:
            sample = self.postTransform(rot_img)
        else:
            sample = transforms.ToTensor(rot_img)
        return sample, rot_class
    
    def get_patch(self, index):
        patch_loc_arr = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[1] <= self.min_width or image.shape[0] <= self.min_width:
            return self.__getitem__(index)
        
        patch_direction_label = int(math.floor((8 * random.random())))
        patch_jitter_y = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
        patch_jitter_x = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
                
        while True:
                
            uniform_patch_x_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            uniform_patch_y_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin - int(round(self.patch_dim/2.0))
            random_patch_y_coord = uniform_patch_x_coord + patch_loc_arr[patch_direction_label][0] * (self.patch_dim + self.gap) + patch_jitter_y
            random_patch_x_coord = uniform_patch_y_coord + patch_loc_arr[patch_direction_label][1] * (self.patch_dim + self.gap) + patch_jitter_x

            if random_patch_y_coord>=0 and random_patch_x_coord>=0 and random_patch_y_coord+self.patch_dim<image.shape[0] and random_patch_x_coord+self.patch_dim<image.shape[1]:
                break
        
        uniform_patch = image[
            uniform_patch_x_coord : uniform_patch_x_coord + self.patch_dim, 
            uniform_patch_y_coord : uniform_patch_y_coord + self.patch_dim
        ]                
        random_patch = image[
            random_patch_y_coord : random_patch_y_coord + self.patch_dim, 
            random_patch_x_coord : random_patch_x_coord + self.patch_dim
        ]
        # 非必要模块：随机图片抖动
        # self.prep_patch(uniform_patch)
        # self.prep_patch(random_patch)
        if self.preTransform:
            uniform_patch = self.postTransform(uniform_patch)
            random_patch = self.postTransform(random_patch)
        else:
            uniform_patch = transforms.ToTensor(uniform_patch)
            random_patch = transforms.ToTensor(random_patch)

        patch_direction_label = np.array(patch_direction_label).astype(np.int64)
        return uniform_patch, random_patch, patch_direction_label
    
    def get_jigpa(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)
        # If image is too small, try another image
        if image.shape[0] <= self.jigmin_width or image.shape[1] <= self.jigmin_width:
                return self.__getitem__(index)
        
        center_y_coord = int(math.floor((image.shape[0] - self.jigmargin*2) * random.random())) + self.jigmargin
        center_x_coord = int(math.floor((image.shape[1] - self.jigmargin*2) * random.random())) + self.jigmargin
        patch_coords = [
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift),
                center_x_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift)
            ),
            (
                center_y_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift),
                center_x_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift
            ),
            (
                center_y_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift,
                center_x_coord - (self.patch_dim + self.half_gap() + self.jigpa_random_jitter() + self.color_shift)
            ),
            (
                center_y_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift,
                center_x_coord + self.half_gap() + self.jigpa_random_jitter() - self.color_shift
            )
        ]
        
        patch_shuffle_order_label = int(math.floor((24 * random.random())))
        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[patch_shuffle_order_label],patch_coords))]

        patch_a = image[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = image[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = image[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = image[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        gray = random.random() < self.gray_portion
        patch_a = self.prep_jigpa(patch_a, gray)
        patch_b = self.prep_jigpa(patch_b, gray)
        patch_c = self.prep_jigpa(patch_c, gray)
        patch_d = self.prep_jigpa(patch_d, gray)

        patch_shuffle_order_label = np.array(patch_shuffle_order_label).astype(np.int64)

        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label
    
    def get_jigro(self, index):
        img_PIL, _ = self.dataset[index]
        image = np.array(img_PIL)

        window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
        window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))
        window = image[window_y_coord:window_y_coord+self.window_width, window_x_coord:window_x_coord+self.window_width]
        
        rotation_label = int(math.floor((4 * random.random())))
        order_label = int(math.floor((24 * random.random()))) 
        
        if rotation_label>0:
            window = np.rot90(window, rotation_label).copy()

        patch_coords = [
            (self.jigro_random_jitter(), self.jigro_random_jitter()),
            (self.jigro_random_jitter(), self.sub_window_width + self.jigro_random_jitter()),
            (self.sub_window_width + self.jigro_random_jitter(), self.jigro_random_jitter()),
            (self.sub_window_width + self.jigro_random_jitter(), self.sub_window_width + self.jigro_random_jitter()),
        ]

        patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[order_label],patch_coords))]

        patch_a = window[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
        patch_b = window[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
        patch_c = window[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
        patch_d = window[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

        patch_a = self.prep_jigro(patch_a)
        patch_b = self.prep_jigro(patch_b)
        patch_c = self.prep_jigro(patch_c)
        patch_d = self.prep_jigro(patch_d)

        # combined_label = np.array(rotation_label * 24 + order_label).astype(np.int64)
        combined_label = np.array(order_label).astype(np.int64)
                
        if self.postTransform:
            patch_a = self.postTransform(patch_a)
            patch_b = self.postTransform(patch_b)
            patch_c = self.postTransform(patch_c)
            patch_d = self.postTransform(patch_d)

        return patch_a, patch_b, patch_c, patch_d, combined_label

    def __getitem__(self, index):
        return self.get_plain(index), self.get_valid(index), self.get_rota(index), self.get_patch(index), self.get_jigpa(index), self.get_jigro(index)