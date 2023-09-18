# final count: source:  7  target:  1054
from PIL import Image
from .align_trans import get_reference_facial_points, warp_and_crop_face
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import os
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")

def compute_area(item):
    return -np.prod(item['box'][-2:])

keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "data/portrait/align112x112", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    crop_size = args.crop_size
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    # unsequential 
    results_all = json.load(open('data/portrait/mtcnn/mtcnn_256.json'))
    default_tfm = np.array([[ 2.90431126e-01, 1.89934467e-03, -1.88962605e+01],
                            [-1.90354592e-03, 2.90477119e-01, -1.70081139e+01]])

    H1 = W1 = 256
    H2 = W2 = 112
    A = np.array([[2 / (W1 - 1), 0, -1], [0, 2 / (H1 - 1), -1], [0, 0, 1]])
    B = np.linalg.inv(np.array([[2 / (W2 - 1), 0, -1], [0, 2 / (H2 - 1), -1], [0, 0, 1]]))
    C = np.array([[0, 0, 1]])

    def tfm2theta(tfm):
        ttt = np.concatenate([tfm, C], axis=0)
        ttt = np.linalg.inv(ttt)
        theta = A @ ttt @ B
        return theta[:2]

    all_tfms = 0
    
    use_torch = False
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    save_img = True
    error_path = 'data/portrait/error_img.json'
    if os.path.exists(error_path):
        error_img = json.load(open(error_path, 'r'))
    else:
        error_img = {'source': [], 'target': []}
    affine_theta_all = {}
    for type in ['source', 'target']:
        cnt = 0
        affine_theta_all[type] = {}
        img_list = os.listdir(os.path.join('data/portrait/align', type))
        for img, value in tqdm(results_all[type].items()):
            value = sorted(value, key=compute_area)
            if len(value) == 0:
                error_img[type].append(img)
                continue
            value = value[0]
            facial5points = [value['keypoints'][key] for key in keys]
            tfm = warp_and_crop_face(None, facial5points, reference, crop_size=(crop_size, crop_size), return_tfm=True)

            all_tfms += tfm
            cnt += 1

            theta = tfm2theta(tfm).tolist()
            affine_theta_all[type][img] = theta
            
            if save_img:
                image = Image.open(os.path.join('data/portrait/align', type, img))
                if not use_torch:
                    face_img = cv2.warpAffine(np.array(image), tfm, (crop_size, crop_size))
                    img_warped = Image.fromarray(face_img)
                else:
                    image = to_tensor(image)[None].float()
                    image = torch.nn.functional.interpolate(image, size=(256, 256))
                    theta = torch.tensor(tfm2theta(tfm)[None]).float()
                    grid = torch.nn.functional.affine_grid(theta, size=(1, 3, crop_size, crop_size))
                    image = torch.nn.functional.grid_sample(image, grid)
                    img_warped = to_pil(image[0])

                os.makedirs(os.path.join(args.dest_root, type), exist_ok=True)
                img_warped.save(os.path.join(args.dest_root, type, img))
        print('type: {}, cnt: {}'.format(type, cnt))

    json.dump(affine_theta_all, open('data/portrait/affine_theta.json', 'w'), indent=4)
    json.dump(error_img, open(error_path, 'w'), indent=4)