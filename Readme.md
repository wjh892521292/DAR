<div align="center">
<h1>Scalable Autoregressive Monocular Depth Estimation</h1>


**CVPR 2025**  
<a href='https://depth-ar.github.io/' style="margin-right: 20px;"><img src='https://img.shields.io/badge/Project Page-DAR-darkgreen' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2411.11361" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-arXiv-maroon' alt='arXiv page'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scalable_Autoregressive_Monocular_Depth_Estimation_CVPR_2025_paper.pdf" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-CvF-blue' alt='IEEE Xplore Paper'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wang_Scalable_Autoregressive_Monocular_CVPR_2025_supplemental.pdf" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Supplementary-CvF-blue' alt='IEEE Xplore Paper'></a>

</div>

## News 
- [Jun 2025] Code released.
- [Feb 2025] DAR accepted in CVPR'2025.

## Installation


``` bash
git clone https://github.com/wjh892521292/DAR/
cd DAR
conda env create -f requirements.yml
conda activate dar
```


- (Optional) install and compile ```xformers``` for faster attention computation. 

```
pip install xformers==0.0.21
```


## Prepare Dataset 
Prepare **NYU Depth V2** and **KITTI** datasets from [BTS](https://github.com/cleinc/bts). After downloading the datasets, change the data paths in the respective bash files to point to your dataset location where you have downloaded the datasets. 

- For example, when training on **NYU Depth V2** dataset, the training bash file should contain:
  
```
    --data_path <path_to_nyu_dataset>\
```
Note the dataset structure inside the path you have given in the bash files should look like this:  
**NYU Depth V2**
``` bash
nyu
├── nyu_depth_v2
│   ├── official_splits
│   └── sync
```
**KITTI**: 
``` bash
kitti
├── KITTI
│   ├── 2011_09_26
│   ├── 2011_09_28
│   ├── 2011_09_29
│   ├── 2011_09_30
│   └── 2011_10_03
└── kitti_gt
    ├── 2011_09_26
    ├── 2011_09_28
    ├── 2011_09_29
    ├── 2011_09_30
    └── 2011_10_03
```

## Pretrained Models

Download the v1-5 checkpoint of [stable-diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt) and put it in the `<repo root>/checkpoints` directory. Please create an empty directory if you find that such a path does not exist. 

## Training 
We recommend using 8 NVIDIA A100 GPUs to train the model with a total batch size of 32. Inside the `train_{kitti,nyu}.sh` set the `NPROC_PER_NODE` variable and `--batch_size` argument to the desired values as per your system resources. For our method we set them as `NPROC_PER_NODE=8` and `--batch_size=4` (resulting in a total batch size of 32). Run the following instuction to train the DAR model:

1. **Train on NYUv2 dataset**:  
`bash train_nyu.sh`  

1. **Train on KITTI dataset**:  
`bash train_kitti.sh`




