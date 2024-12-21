CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py \
--config_path configs/bevformer-ovo-r101-704x256-san-qwen-512.py \
--log_folder bevformer-ovo-r101-704x256-san-qwen-512 \
--seed 7240 --log_every_n_steps 100