## Installation
Following the official [SAN](https://github.com/MendelXu/SAN)

**a. Create a conda virtual environment and activate**
```shell
conda create -n san python=3.8 -y
conda activate san
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
or
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
**c. Install other packages**
```shell
pip install -r requirements.txt
```

**d. Install detectron2**
```shell
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
git checkout v0.6
cd ../
pip install detectron2
```

**e. Prepare checkpoints**
Download [checkpoints](https://huggingface.co/Mendel192/san/resolve/main/san_vit_large_14.pth) and put it under ckpts folder.

## Usage
Please use the following script to generate segmentation results
```shell
# scene
python main.py --data_root data/occ3d --output_root data/occ3d/san_qwen_scene --vocab_root data/occ3d/qwen_texts
# single-frame vocabulary
python main.py --data_root data/occ3d --output_root data/occ3d/san_qwen_frame --vocab_root data/occ3d/qwen_texts
# save feat
python main_feat.py --data_root data/occ3d --output_root data/occ3d/san_feats --vocab_root data/occ3d/qwen_texts
```
Then the dataset folder should be organized as follows:
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
|   ├── san_qwen_frame/
|   ├── san_feats/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```