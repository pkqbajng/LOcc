CUDA_VISIBLE_DEVICES=0 nohup python main.py --eval \
--config_path configs/bevstereo/bevstereo-ovo-r50-san-qwen-704x256.py \
--log_folder bevstereo/bevstereo-ovo-r50-san-qwen-704x256-eval \
--ckpt_path logs/bevstereo/bevstereo-ovo-r50-san-qwen-704x256/tensorboard/version_0/checkpoints/best.ckpt \
--visualize ./bevstereo_visualize