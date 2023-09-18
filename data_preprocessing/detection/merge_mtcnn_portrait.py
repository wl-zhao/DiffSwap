import json


if __name__ == '__main__':
    gpu_num = 1
    mtcnns = {}
    for type in ['source', 'target']:
        mtcnns[type] = {}
        for i in range(gpu_num):
            tmp = json.load(open('data/portrait/mtcnn/mtcnn_{}.json'.format(i), 'r'))
            mtcnns[type].update(tmp[type])
    
        for i, j in mtcnns[type].items():
            print(type, i)
    json.dump(mtcnns, open('data/portrait/mtcnn/mtcnn_256.json', 'w'), indent=4)