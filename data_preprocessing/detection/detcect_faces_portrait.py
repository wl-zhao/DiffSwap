from mtcnn import MTCNN
import cv2
import os
import json
from tqdm import tqdm
import sys


if __name__ == '__main__':
    gpu_num = 1
    gpu = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    mtcnn_path = 'data/portrait/mtcnn'
    data_path = 'data/portrait/align'
    detector = MTCNN()
    results_all = {}
    for type in ['source', 'target']:
        results_all[type] = {}
        img_list = os.listdir(os.path.join(data_path, type))
        img_list.sort()
        count = 0
        for img_idx, img in enumerate(tqdm(img_list)):
            if img_idx % gpu_num != int(gpu):
                continue

            count += 1
            image = cv2.cvtColor(cv2.imread(os.path.join(data_path, type, img)), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            results_all[type][img] = result

    os.makedirs(mtcnn_path, exist_ok=True)
    print(f'gpu {gpu} process {count} images')
    json.dump(results_all, open(os.path.join(mtcnn_path, f'mtcnn_{gpu}.json'), 'w'), indent=4)