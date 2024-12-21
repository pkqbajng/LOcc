## Installation
Please refer to [BEVDet](../5-OVO/BEVDet/BEVDet.md) to prepare environment for ground truth generation. To achieve nearest label assignment, please install the chamfer_dist:
```shell
cd chamfer_dist
python setup.py install
cd ../
```

## Usage
Before generating ground truth, the dataset folder should be organized as follows:
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
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```
where san_qwen_scene could be san_qwen_frame, san_feats, cat_seg_qwen_scene, odise_qwen_scene or other open-vocabulary segmentation results. Then use the following scripts to generate pseudo-labeled ground truth.
```shell
# SAN scene
python PseudoOccGeneration.py --split train --data_root data/occ3d --seg_root data/occ3d/san_qwen_scene --save_path data/occ3d/san_gts_qwen_scene
# SAN frame
python PseudoOccGeneration.py --split train --data_root data/occ3d --seg_root data/occ3d/san_qwen_frame --save_path data/occ3d/san_gts_qwen_frame
# SAN nearest
python PseudoOccGeneration-Nearest.py --split train --data_root data/occ3d --seg_root data/occ3d/san_qwen_scene --save_path data/occ3d/san_gts_qwen_nearest
# SAN feat
python PseudoOccGeneration-Feat.py --split train --data_root data/occ3d --seg_root data/occ3d/san_feats --save_path data/occ3d/san_gts_feats
# SAN voxel-based model-view projection
python PseudoOccGeneration-VoxelProjection --split train --data_root data/occ3d --seg_root data/occ3d/san_qwen_scene --save_path data/occ3d/san_gts_projection
# CAT-Seg scene
python PseudoOccGeneration.py --split train --data_root data/occ3d --seg_root data/occ3d/cat_seg_qwen_scene --save_path data/occ3d/cat_seg_gts_qwen_scene
# ODISE scene
python PseudoOccGeneration.py --split train --data_root data/occ3d --seg_root data/occ3d/odise_qwen_scene --save_path data/occ3d/odise_gts_qwen_scene
```

The dataset folder with pseudo-labeled ground truth should be organized as:
```shell
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
```