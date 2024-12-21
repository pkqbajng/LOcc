## Installation
Please refer to [BEVDet](../5-OVO/BEVDet/BEVDet.md) to prepare environment for training autoencoder and install open_clip:
```shell
pip install open_clip_torch
```

## Usage
First, please count all the words in the entire dataset and generate text embeddings. 
```shell
python count_words.py --data_root data/occ3d --ovo_root data/occ3d/san_gts_qwen_scene --embedding_file data/occ3d/text_embedding/overall_embedding.json
```
Then the autoencoder can be trained using the follow script:
```shell
python train.py --text_embedding_file data/occ3d/text_embedding/overall_embedding.json --log_dir qwen --num_epochs 300 --encoder_dims 256 256 128 128 --decoder_dims 128 256 256 512
```
Similarlly, please generate the query text embedding:
```shell
python generate_query_embedding --embedding_file data/occ3d/text_embedding/query.json
```
The text embedding for query words and scene vocabulary can be obtained by
```shell
# query embedding
python generate_embedding.py --data_root data/occ3d --query --query_embedding_file data/occ3d/text_embedding/query.json
# gt embedding
python generate_embedding.py --data_root data/occ3d --ovo_root data/occ3d/san_gts_qwen_scene
```
To map the query embedding or gt embedding to low-dimensional space, please use
```shell
# query embedding
python map_embedding.py --data_root data/occ3d --query --query_embedding_file data/occ3d/text_embedding/query.json --low_dimension_query_embedding_file data/occ3d/text_embedding/query_128.json
# gt embedding
python map_embedding.py --data_root data/occ3d --ovo_root data/occ3d/san_gts_qwen_scene
```