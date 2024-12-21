## Installation
**a. Create a conda virtual environment and activate**
```shell
conda create -n LOcc python=3.8 -y
conda activate LOcc
```
**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
or 
```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
**c. Install mmcv-full.**

```shell
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
**d. Install mmdet & mmsegmentation.**

```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```
**e. Install mmdetection3d.**

```shell
git clone git@github.com:open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
# pip install -v -e .
python setup.py install
cd ../
```

If you met this error: cannot import name 'packaging' from 'pkg_resources', please fix the bug using the following script, and then installation can be fullfilled.
```shell
pip install setuptools==69.5.1
```
**f. Install mmdet3d_plugin**

```shell
cd mmdet3d_plugin
pip install -v -e .
# python setup.py install
```
**g. Install other packages**

```shell
pip install -r docs/requirements.txt
```

## Preparation

The dataset folder should be organized as

```
data
├── occ3d/
|   ├── can_bus/
|   ├── gts/
|   ├── maps/
|   ├── samples/
|   ├── sweeps/
|   ├── qwen_texts_step1/
|   ├── qwen_texts_step2/
|   ├── qwen_texts/
|   ├── san_qwen_scene/
|   ├── san_gts_qwen_scene/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
|   ├── text_embedding/
```

## Training

Please download the pretrained [ckpts](https://github.com/pkqbajng/LOcc/releases/download/v1.0/ckpts.zip) and put them under the folder ckpts.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --config_path configs/bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256.py --log_folder bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256 --seed 7240 --log_every_n_steps 100
```

## Test

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --eval --config_path configs/bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256.py --log_folder bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256-eval --ckpt_path logs/bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256/tensorboard/version_0/checkpoints/best.ckpt
```

For training and test, we provide [bevdet4d-ovo-r50-san-qwen-704x256](configs/bevdet4d/bevdet4d-ovo-r50-san-qwen-704x256.py) as example.