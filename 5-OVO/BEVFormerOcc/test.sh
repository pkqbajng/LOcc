CUDA_VISIBLE_DEVICES=0 python main.py --eval \
--config_path configs/bevformer-ovo-r101-704x256-san-qwen.py \
--log_folder bevformer-ovo-r101-704x256-san-qwen-eval \
--ckpt_path logs/bevformer-ovo-r101-704x256-san-qwen/tensorboard/version_0/checkpoints/best.ckpt \
--visualize ./bevformer_ovo_visualize 