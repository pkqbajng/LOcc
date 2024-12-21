## nuScenes-Occ3d
Please download nuScenes full dataset v1.0, CAN bus expansion, and nuScenes-lidarseg from the [official website](https://www.nuscenes.org/download). For the occupancy labels, please download them from the [Occ3d website](https://tsinghua-mars-lab.github.io/Occ3D/). The dataset folder should be organized as follows:
```
data
├── occ3d/
|   ├── can_bus/
|   ├── gts/
|   ├── maps/
|   ├── samples/
|   ├── sweeps/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
```
Then please use the following scripts to prepare the pkl files.
```shell
python data_tools/create_data_bevdet.py
```
Then the finally dataset folder will be organized as
```
data
├── occ3d/
|   ├── can_bus/
|   ├── gts/
|   ├── maps/
|   ├── samples/
|   ├── sweeps/
|   ├── v1.0-trainval/
|   ├── v1.0-test/
|   ├── lidarseg/
|   |   ├── v1.0-trainval/
|   |   ├── v1.0-test/
|   |   ├── v1.0-mini/
|   ├── bevdetv2-nuscenes_infos_train.pkl
|   ├── bevdetv2-nuscenes_infos_val.pkl
```