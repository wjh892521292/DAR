NPROC_PER_NODE=8 #8
NO_CPU_CORES=64

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$((NO_CPU_CORES / NPROC_PER_NODE))
export TOKENIZERS_PARALLELISM=false

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
/ossfs/workspace/hy58/caijin/envs/ecodepth/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:1234 \
    --nproc_per_node=8 \
    train.py \
    --batch_size 4 \
    --dataset nyudepthv2 \
    --data_path /ossfs/workspace/ts/wangjinhong/data/nyu \
    --weight_decay 0.1 \
    --num_filters 32 32 32 \
    --deconv_kernels 2 2 2 \
    --flip_test \
    --layer_decay 0.9 \
    --drop_path_rate 0.3 \
    ${@:2} \
    --pro_bar True \
    --save_depths_gray \
    --save_depths_color \
    --vit_model google/vit-base-patch16-224 \
    --learning_rate_schedule one_cycle \
    --gradient_accumulation False \
    --epochs 100 \
    --log_dir /ossfs/workspace/hy65/caijin/result/DAR_rebuttal/models/nyu \
    --log_in_wandb False \
    --eigen_crop_in_dataloader_itself_for_nyu False \
    --min_depth_eval 1e-3 \
    --max_depth_eval 10 \
    --no_of_classes 100 \
    --variance_focus 0.85 \
    --cutflip=False \
    --save_last_model \
    --encoder VIT-14 \
    --exp_name DAR_nyu_eco_best_reproduct2_2b \
