import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
import io
from torchvision import transforms
import cv2
import pdb
import PIL
import numpy as np
from einops import rearrange
from scipy.spatial import ConvexHull
import random
import json
from math import floor
import os
import warnings
warnings.filterwarnings("ignore")

class Portrait(Dataset):
    def __init__(self, root, size = 256, base_res = 256, flip=False, 
                interpolation="bicubic", dilate=False, convex_hull=True):
        super().__init__()
        self.size = size
        self.root = root
        self.base_res = base_res
        self.error_img = json.load(open(f'{root}/error_img.json'))
        
        self.lmk_path = f'{root}/landmark/landmark_256.pkl'
        self.landmarks = pickle.load(open(self.lmk_path,'rb'))
        
        self.src_list = os.listdir(f'{root}/source') 
        self.src_list = [x for x in self.src_list if x not in self.error_img['source']]
        self.src_list.sort()
        self.tgt_list = os.listdir(f'{root}/target') 
        self.tgt_list = [x for x in self.tgt_list if x not in self.error_img['target']]
        self.tgt_list.sort()
        
        print(f'len(self.src_list): {len(self.src_list)}')
        self.affine_thetas = json.load(open(f'{root}/affine_theta.json'))
        
        self.interpolation = {"linear": PIL.Image.LINEAR,
                        "bilinear": PIL.Image.BILINEAR,
                        "bicubic": PIL.Image.BICUBIC,
                        "lanczos": PIL.Image.LANCZOS,
                        }[interpolation]
        self.convex_hull = convex_hull
        
                
        all_indices = np.arange(0, 68)
        self.landmark_indices = {
            # 'face': all_indices[:17].tolist() + all_indices[17:27].tolist(),
            'l_eye': all_indices[36:42].tolist(),
            'r_eye': all_indices[42:48].tolist(),
            'nose': all_indices[27:36].tolist(),
            'mouth': all_indices[48:68].tolist(),
        }
        self.dilate = dilate
        if dilate:
            self.dilate_kernel = np.ones((11, 11), np.uint8)

    def __len__(self):
        return len(self.src_list) * len(self.tgt_list) # 9 * 1039 = 9351
    
    def __getitem__(self, index): # index: 0 - 9350
        src_index = floor(index / len(self.tgt_list)) # 0 - 8
        tgt_index = index - len(self.tgt_list) * src_index # 0 - 1038
        batch = {}
        
        for type in ['source', 'target']:
            if type == 'source':
                image = Image.open(os.path.join(f'{self.root}/align/{type}', self.src_list[src_index])).convert('RGB')
                affine_theta = np.array(self.affine_thetas[type][self.src_list[src_index]], dtype=np.float32)
                landmark = torch.tensor(self.landmarks[type][self.src_list[src_index]]) / self.base_res
            elif type == 'target':
                image = Image.open(os.path.join(f'{self.root}/align/{type}', self.tgt_list[tgt_index])).convert('RGB')
                affine_theta = np.array(self.affine_thetas[type][self.tgt_list[tgt_index]], dtype=np.float32)
                landmark = torch.tensor(self.landmarks[type][self.tgt_list[tgt_index]]) / self.base_res
            
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)   

            if type == 'source':
                batch['mask_organ_src'] = self.mask_organ_src(landmark)
                batch['image_src'] = image
                # batch['image_src']'s identity 
                batch['affine_theta_src'] = affine_theta
                batch['src'] = self.src_list[src_index]
            else:
                batch['image'] = image
                # batch['image']'s identity 
                batch['landmark'] = landmark
                batch['affine_theta'] = affine_theta
                batch['target'] = self.tgt_list[tgt_index]
                
        if self.convex_hull:
            mask_dict = self.extract_convex_hulls(batch['landmark'])
            batch.update(mask_dict)
        return batch
    
    def mask_organ_src(self, landmark):
        mask_organ = []
        for key, indices in self.landmark_indices.items():
            mask_key = self.extract_convex_hull(landmark[indices])
            if self.dilate:
                # mask_key = mask_key[:, :, None]
                # mask_key = repeat(mask_key, 'h w -> h w k', k=3)
                # print(mask_key.shape, type(mask_key))
                mask_key = mask_key.astype(np.uint8)
                mask_key = cv2.dilate(mask_key, self.dilate_kernel, iterations=1)
            mask_organ.append(mask_key)
        return np.stack(mask_organ)
        
        
    def extract_convex_hulls(self, landmark):
        mask_dict = {}
        mask_organ = []
        for key, indices in self.landmark_indices.items():
            mask_key = self.extract_convex_hull(landmark[indices])
            if self.dilate:
                # mask_key = mask_key[:, :, None]
                # mask_key = repeat(mask_key, 'h w -> h w k', k=3)
                # print(mask_key.shape, type(mask_key))
                mask_key = mask_key.astype(np.uint8)
                mask_key = cv2.dilate(mask_key, self.dilate_kernel, iterations=1)
            mask_organ.append(mask_key)
        mask_organ = np.stack(mask_organ) # (4, 256, 256)
        mask_dict['mask_organ'] = mask_organ
        mask_dict['mask'] = self.extract_convex_hull(landmark)
        return mask_dict
    
    def extract_convex_hull(self, landmark):
        landmark = landmark * self.size
        hull = ConvexHull(landmark)
        image = np.zeros((self.size, self.size))
        points = [landmark[hull.vertices, :1], landmark[hull.vertices, 1:]]
        points = np.concatenate(points, axis=-1).astype('int32')
        mask = cv2.fillPoly(image, pts=[points], color=(255,255,255))
        mask = mask > 0
        return mask

def visualize(batch):
    n = len(batch['image'])
    os.makedirs('ldm/data/debug', exist_ok=True)
    for i in range(n):
        print(i)
        print(batch['src'][i], batch['target'][i])
        image = (batch['image'][i] + 1) / 2 * 255
        image = rearrange(image, 'h w c -> c h w')
        image = image.numpy().transpose(1, 2, 0).astype('uint8').copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'ldm/data/debug/{i}_tgt.png', image)

        image_src = (batch['image_src'][i] + 1) / 2 * 255
        image_src = rearrange(image_src, 'h w c -> c h w')
        image_src = image_src.numpy().transpose(1, 2, 0).astype('uint8').copy()
        image_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'ldm/data/debug/{i}_src.png', image_src)

        lmk = (batch['landmark'][i] * image.shape[0]).numpy().astype('int32')
        for k in range(68):
            image = cv2.circle(image, (lmk[k, 0], lmk[k, 1]), 3, (255, 0, 255), thickness=-1)
        cv2.imwrite(f'ldm/data/debug/{i}_lmk.png', image)

        mask = (batch['mask'][i].numpy() * 255).astype('uint8') #[:, :, None]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'ldm/data/debug/{i}_mask.png', mask)
        
        mask_organ = (batch['mask_organ'][i][0].numpy() * 255).astype('uint8') #[:, :, None]
        mask_organ = cv2.cvtColor(mask_organ, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'ldm/data/debug/{i}_mask_organ.png', mask_organ)
        
        mask_organ_src = (batch['mask_organ_src'][i][1].numpy() * 255).astype('uint8') #[:, :, None]
        mask_organ_src = cv2.cvtColor(mask_organ_src, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'ldm/data/debug/{i}_mask_organ_src.png', mask_organ_src)
        
if __name__ == '__main__':
    dataset = Portrait('data/portrait')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for batch in dataloader:
        visualize(batch)
        break