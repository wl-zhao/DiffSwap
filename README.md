# DiffSwap


Created by [Wenliang Zhao](https://wl-zhao.github.io/), [Yongming Rao](https://raoyongming.github.io/), Weikang Shi, [Zuyan Liu](https://scholar.google.com/citations?user=7npgHqAAAAAJ&hl=en), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)†

This repository contains PyTorch implementation for paper "DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion"

[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.pdf)

## Installation
Please first install the environment following [stable-diffusion](https://github.com/CompVis/stable-diffusion), and then run `pip install -r requirements.txt`.

Please download the checkpoints from [[here]](https://cloud.tsinghua.edu.cn/d/962ccd4b243442a3a144/?p=%2Fcheckpoints%2FDiffSwap&mode=list), and put them under the  `checkpoints/` folder. 
The resulting file structure should be:

```
├── checkpoints
│   ├── diffswap.pth
│   ├── glint360k_r100.pth
│   └── shape_predictor_68_face_landmarks.dat
```

## Inference
We provide a sample code to perform face swapping given the portrait source and target images. Please put the source images and target images in `data/portrait_jpg` and run
```
python pipeline.py
```
the swapped results are saved in `data/portrait/swap_res_ori`.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhao2023diffswap,
  title={DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion},
  author={Zhao, Wenliang and Rao, Yongming and Shi, Weikang and Liu, Zuyan and Zhou, Jie and Lu, Jiwen},
  journal={CVPR},
  year={2023}
}
```
