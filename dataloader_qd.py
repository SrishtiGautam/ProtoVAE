import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
# from torchvision import transforms
import pickle
import numpy as np
from PIL import Image
import torch


# from libcpab.libcpab.pytorch import cpab


class CUB11(Dataset):

    def __init__(self, root_dir, mode='train', im_size=(224, 224), transform=None, save_img=False):

        self.im_size = im_size
        self.mode = mode
        self.root_dir = root_dir  # root_dir should be the one within which the images, attributes, parts folders are present
        self.transform = transform
        self.save_img = save_img
        if self.save_img:
            self.dict_name = 'CUB11_data_dict.pkl'
        else:
            self.dict_name = 'CUB11_data_dict_noimg.pkl'

        try:
            pkl_file = open(self.root_dir + self.dict_name, 'rb')
        except:
            self.build_data_dict()

        pkl_file = open(self.root_dir + self.dict_name, 'rb')
        self.data = pickle.load(pkl_file)
        pkl_file.close()

    def build_data_dict(self):
        print('Building the complete data dictionary')
        data_dict = {}
        path_file = open(self.root_dir + 'images.txt', 'r')
        mode_file = open(self.root_dir + 'train_test_split.txt', 'r')
        parts_file = open(self.root_dir + 'parts/part_locs.txt', 'r')
        label_file = open(self.root_dir + 'image_class_labels.txt', 'r')

        # Reading and dividing all the indices on the basis of whether they are for training
        idx_content = mode_file.readlines()
        train_idx = np.array([int(line.split()[0]) for line in idx_content if line.split()[1] == '1'])
        test_idx = np.array([int(line.split()[0]) for line in idx_content if line.split()[1] == '0'])
        mode_file.close()

        label_content = label_file.readlines()
        labels = np.array([int(line.split()[1]) - 1 for line in label_content])
        label_file.close()

        path_content = path_file.readlines()
        if self.save_img:
            images = np.array(
                [np.array(Image.open(self.root_dir + '/images/' + line.split()[1]).resize(self.im_size).convert('RGB'))
                 for line in path_content])
        paths = [self.root_dir + '/images/' + line.split()[1] for line in path_content]
        path_file.close()

        parts_content = parts_file.readlines()
        parts_file.close()
        parts = []
        vis_parts = []
        cur_part_locs = []
        cur_part_vis = []
        for line in parts_content:
            line_content = line.split()
            cur_part_locs += [float(line_content[2]), float(line_content[3])]
            cur_part_vis += [float(line_content[4])]
            if line_content[1] == '15':
                parts.append(cur_part_locs)
                vis_parts.append(cur_part_vis)
                cur_part_locs = []
                cur_part_vis = []
        parts = np.array(parts)
        vis_parts = np.array(vis_parts)

        data_dict['train'] = train_idx
        data_dict['test'] = test_idx
        data_dict['y'] = labels
        if self.save_img:
            data_dict['images'] = images
        data_dict['part_locs'] = parts
        data_dict['part_vis'] = vis_parts
        data_dict['path'] = paths
        f = open(self.root_dir + self.dict_name, 'wb')
        pickle.dump(data_dict, f)
        f.close()

        return

    def set_mode(self, new_mode):
        self.mode = new_mode

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):
        true_idx = self.data[self.mode][idx] - 1
        norm_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # image, y, part_locs = np.moveaxis(self.data['images'][true_idx], -1, 0)/255.0, self.data['y'][true_idx], self.data['part_locs'][true_idx]
        y, part_locs, vis_locs = self.data['y'][true_idx], self.data['part_locs'][true_idx], self.data['part_vis'][
            true_idx]
        image = Image.open(self.data['path'][true_idx]).resize(self.im_size).convert('RGB')
        # orig_size = np.array(Image.open(self.data['path'][true_idx]).size)
        # change_arr = 50.0 * np.tile(orig_size / np.array([224, 224]), 15)
        # part_locs = part_locs / change_arr
        # print (image.max())
        if not self.transform is None:
            image = self.transform(image)
        # image = torch.as_tensor(image).float()
        return image, torch.as_tensor(y)  # , torch.as_tensor(vis_locs).float()


class QuickDraw(Dataset):

    def __init__(self, ncat=10, mode='train', root_dir=None):

        self.mode = mode
        self.root_dir = root_dir  # root_dir should be the one within which the '.npy' are present, categories.txt should also be present

        cat_file = open(self.root_dir + 'categories.txt', 'r')
        self.cat = cat_file.readlines()
        self.cat = [x.rstrip('\n') for x in self.cat]
        if ncat == 10:
            self.cat = ['apple', 'banana', 'carrot', 'grapes', 'ant', 'cat', 'dog', 'cow', 'lion', 'frog']
            self.cat.sort()
            print('Selected categories', self.cat)
        elif ncat == 20:
            self.cat = ['apple', 'banana', 'carrot', 'grapes', 'ant', 'cat', 'dog', 'cow', 'lion', 'frog', 'camel',
                        'airplane', 'broccoli', 'bus', 'butterfly', 'cactus', 'camera', 'calculator', 'alarm clock',
                        'ambulance']
            self.cat.sort()
            print('Selected categories', self.cat)
            # Organize categories
        # --- Write code here ---

        self.dict_name = 'info_' + str(len(self.cat)) + '.pkl'
        try:
            pkl_file = open(self.root_dir + self.dict_name, 'rb')
        except:
            self.build_info_dict()

        pkl_file = open(self.root_dir + self.dict_name, 'rb')
        self.info = pickle.load(pkl_file)
        pkl_file.close()

    def build_info_dict(self):
        # This is a useless function now
        print('Building the info dictionary. selecting images for training and testing from organized categories')
        info_dict = {}
        info_dict['train'] = {}
        info_dict['test'] = {}
        # Select images for training, testing
        # --- Code here ---
        for i in range(len(self.cat)):
            cur_cat = self.cat[i]
            cur_samp = np.load(self.root_dir + cur_cat + '_small.npy')
            n_samp = cur_samp.shape[0]
            idx = np.arange(0, n_samp)
            np.random.shuffle(idx)
            info_dict['train'][cur_cat] = idx[:8000]
            info_dict['test'][cur_cat] = idx[8000:]

        info_dict['train']['samp_per_class'] = 8000
        info_dict['test']['samp_per_class'] = 2000
        f = open(self.root_dir + self.dict_name, 'wb')
        pickle.dump(info_dict, f)
        f.close()
        return

    def set_mode(self, new_mode):
        self.mode = new_mode

    def __len__(self):
        return len(self.cat) * self.info[self.mode]['samp_per_class']

    def __getitem__(self, idx):
        samp_per_class = self.info[self.mode]['samp_per_class']
        y = idx // samp_per_class
        cat = self.cat[y]
        class_idx = idx % samp_per_class
        if self.mode == 'test':
            class_idx = self.info['train'][
                            'samp_per_class'] + class_idx  # Ensure picking from samples not from training data
        arr = np.load(self.root_dir + cat + '_small.npy')
        image = arr[class_idx].reshape([28, 28])
        image = torch.as_tensor(image).float() / 255.0
        image = (image-0.5)/0.5
        image = image.unsqueeze(0)
        # image = np.repeat(image, 3, axis=0)
        return image, torch.as_tensor(y)  # , torch.as_tensor(vis_locs).float()






