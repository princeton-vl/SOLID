# Label-Free Synthetic Pretraining of Object Detectors

Code for reproducing the results in the following paper:

[Label-Free Synthetic Pretraining of Object Detectors]()  
Hei Law, Jia Deng  
arXiv, 2022

## Getting Started
Install [Anaconda](https://anaconda.org), create an environment, install packages and activate the environment.
```bash
conda create --name solid python=3.9
pip install -r conda_requirements.txt
conda activate solid
```
In the following sections, we assume the conda environment is activated. This code is only tested on Linux.

## Dataset
This section describes steps to generate the synthetic data. Or you can skip to the [Pre-training and Fine-tuning](#pre-training-and-fine-tuning) section and download the dataset used in our paper.

Codes related to this section can be found in the `render` directory. We assume everything in this section is run under `render`.

### Installing Python packages and Blender
1. Download [Blender](https://mirrors.ocf.berkeley.edu/blender/release/Blender2.93/blender-2.93.9-linux-x64.tar.xz), untar it and rename the directory.
```bash
curl -O https://mirrors.ocf.berkeley.edu/blender/release/Blender2.93/blender-2.93.9-linux-x64.tar.xz 
tar -xvf blender-2.93.9-linux-x64.tar.xz
mv blender-2.93.9-linux-x64 blender
```
We only tested our code on Blender 2.93.

2. Install pip and Python packages in Blender.
```bash
./blender/2.93/python/bin/python3.9 -m ensurepip --upgrade
./blender/2.93/python/bin/python3.9 -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
./blender/2.93/python/bin/python3.9 -m pip install -r blender_requirements.txt
```
Blender comes with its own Python binary. Packages installed under the conda environment cannot be used in Blender and vice versa.

### Downloading and pre-processing 3D models
We construct our dataset with 3D models from [ShapeNet](https://shapenet.org) and [SceneNet](https://robotvault.bitbucket.io).

#### SceneNet
Scenes from SceneNet are used as backgrounds in our datasets. Because the scenes come with 3D models, we remove 3D models in the scenes except ceiling, floor and wall before using them as backgrounds. The cleaned up version of SceneNet can be downloaded from [here](https://drive.google.com/file/d/1HaN-NF4l-IBrwWDHAajC_EYWxnOXpwqt/view?usp=sharing).

1. Download the file to the `data` directory and untar it
```bash
cd data
tar -xvf scenenet.tar.gz
```

#### ShapeNet
3D models from ShapeNet are used as foregound objects in our datasets. The ShapeNet models need to be pre-processed so that they are rendered properly in Blender.

1. Apply an account [here](https://shapenet.org) to download ShapeNet.

2. Download the models to the `data` directory and unzip it.

3. Install Node.js via [nvm](https://github.com/nvm-sh/nvm), and install a tool which converts the ShapeNet models from OBJ format to GLTF format
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm use 16
npm install -g obj2gltf
```

4. Pre-process the ShapeNet models
```bash
cd data
bash obj2gltf.sh
```
`obj2gltf.sh` assumes the ShapeNet models are in `ShapeNetCore.v2` and saves the output to `shapenet`. This process may take a while to finish. Due to copyright issues, we cannot provide the pre-processed models.

### Rendering Images
You will need a GPU to render images. We only tested the code on an RTX 2080 Ti.

1. Update paths in `scripts/setup_paths.sh` if you install Blender or the datasets somewhere else.

2. Create a [Zarr](https://zarr.readthedocs.io/en/stable/) dataset to store images and annotations.
```bash
python create_zarr.py datasets/SOLID.zarr --num_images 1000000 --num_classes 52447 --num_shots 8
```
This creates an empty Zarr dataset for storing 1 million target images generated with 52447 3D models and each 3D model has 8 query images. Zarr stores data as a large array. It automatically divides a large array into smaller chunks where each chunk is saved as a single file. In our case, each chunk consists of 256 images. So, there will be 3907 chunks for 1 million images. Our rendering scripts, which will discussed below, process one chunk at a time. Because the scripts only write to a single file, you can run multiple rendering jobs simultaneously as long as each job processes a different chunk.

3. Render target images.
```bash
bash ./scripts/target_images.sh data/shapenet.json datasets/SOLID.zarr <chunk id>
```
`<chunk id>` starts from zero.

4. Render query images.
```bash
bash ./scripts/query_images.sh data/shapenet.json datasets/SOLID.zarr <chunk id>
```

## Pre-training and Fine-tuning
Codes related to pre-training and fine-tuning can be found in the `detection` directory. We assume everything in this section is run under `detection`.

### Dataset
If you render your own dataset, you can skip this step. Otherwise, you can download our dataset from [here](https://drive.google.com/drive/folders/1lLv0k9gtu-Un-rVCoBeoDpkxLI0s_AXB?usp=sharing). Our dataset is large so we divide them into smaller files. Download the files to the `datasets` directory. Concatenate the files and untar the dataset.
```bash
cd datasets
cat SOLID.zarr.tar.{00..23} | tar -xvf -
```
If you are getting an error saying too many users have viewed or downloaded the files, you can select all files, right click, select "Make a Copy" to copy them to your Google Drive and download the files from your Google Drive.

### Installing PyTorch and Detectron2
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

### Pre-training
Pre-training requires 8 A6000 GPUs. We early stop the pre-training after 750000 iterations.
```bash
bash pretrain.sh \
    configs/Pretrain/mask_rcnn_R_50_FPN.py \
    dataloader.train.train_zarr='./datasets/SOLID.zarr' \
    train.output_dir='./output/Pretrain/mask_rcnn'
```
We provide a pre-trained model which can be downloaded [here](https://drive.google.com/file/d/1Sf6hLH-6EwxVE5MuiUbeYPcx0E-isgom/view?usp=sharing).

### Fine-tuning
Follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html#expected-dataset-structure-for-coco-instance-keypoint-detection) to download and set up the COCO dataset. Fine-tuning requires 4 RTX 2080 Ti GPUs.
```bash
bash finetune.sh \
    configs/Finetune/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.WEIGHTS './output/Pretrain/mask_rcnn/model_0749999.pth' \
    OUTPUT_DIR './output/Finetune/mask_rcnn_1x'
```
