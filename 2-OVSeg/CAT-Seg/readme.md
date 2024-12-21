## Installation
Following the official [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg)

**a. Create a conda virtual environment and activate**
```shell
conda create -n cat_seg python=3.8 -y
conda activate cat_seg
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
or
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
**c. Install other packages**
```shell
pip install -r requirements.txt
```

**d. Install detectron2**
```shell
git clone https://github.com/facebookresearch/detectron2.git
pip install detectron2
```

**e. Prepare checkpoints**
Please download the [sam_vit_h.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [model_final.pth](https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_large.pth), and put them under the folder ckpts. Please rename the names to sam_vit_h.pth and model_final.pth in advance.

## Usage
Please use the following script to generate segmentation results.
```shell
python main.py --data_root data/occ3d --output_root data/occ3d/cat_seg_qwen_scene --vocab_root data/occ3d/qwen_texts
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
|   ├── cat_seg_qwen_scene/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```