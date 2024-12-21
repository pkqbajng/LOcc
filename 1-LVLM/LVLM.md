## Installation
Following the official [Qwen-VL](https://github.com/QwenLM/Qwen-VL)

**a. Create a conda virtual environment and activate**
```shell
conda create -n llm python=3.10 -y
conda activate llm
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

or

```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**c. Install other packages**
```shell
pip install -r requirements.txt
```

## Inference
Please use the following scripts for vocabulary extraction with LVLM. 

[Step1](./qwen_vlm_step1.py) uses the Qwen-VL-Chat to extract vocabulary from single images. 
```shell
python qwen_vlm_step1.py --data_root data/occ3d --output_root data/occ3d/qwen_texts_step1
```

[Step2](./qwen_vlm_step2.py) organizes the extracted vocabulary from step1 as a set of words. 
```shell
python qwen_vlm_step2.py --data_root data/nuscenes --source_root qwen_texts_step1 --output_root data/occ3d/qwen_texts_step2
```
[Step3](./qwen_vlm_step3.py) first unifies the forms of the words from each frame and then merges them into a unified set, referred to as scene vocabulary.

```shell
python qwen_vlm_step3.py --data_root data/nuscenes --source_root qwen_texts_step2 --output_root data/occ3d/qwen_texts
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
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```