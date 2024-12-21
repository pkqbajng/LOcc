## Step-by-step installation instructions
**a. Create a conda virtual environment and activate**

```shell
conda create -n ODISE python=3.9 -y
conda activate ODISE
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

**c. Install libcusolver-dev**

```shell
conda install -c "nvidia/label/cuda-11.3" libcusolver-dev
```

**d. Install maskformer**
download the detectron2, panopticapi, lvis manully and rename the files
```shell
cd third_party/Mask2Former
```

replace the original install_requires in the setup.py with the following scripts
```python
install_requires=[
        f"detectron2 @ file://localhost/{os.getcwd()}/v0.6.zip",
        "scipy>=1.7.3",
        "boto3>=1.21.25",
        "hydra-core==1.1.1",
        # there is BC breaking in omegaconf 2.2.1
        # see: https://github.com/omry/omegaconf/issues/939
        "omegaconf==2.1.1",
        f"panopticapi @ file://localhost/{os.getcwd()}/panopticapi.zip",
        f"lvis @ file://localhost/{os.getcwd()}/lvis.zip",
    ],
```

pip install -v -e .

**e. Install odise**
```shell
cd ../../
pip install -v -e .
```

**f. Prepare checkpoints**
```shell
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model --resume-download CompVis/stable-diffusion-v-1-3-original
huggingface-cli download --repo-type model --resume-download openai/clip-vit-large-patch14
```

**g. fix bugs**
```shell
pip install -r requirments.txt
```

## Usage
Please use the following script to generate segmentation results.
```shell
python main.py --data_root data/occ3d --output_root data/occ3d/odise_qwen_scene --vocab_root data/occ3d/qwen_texts
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
|   ├── odise_qwen_scene/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```