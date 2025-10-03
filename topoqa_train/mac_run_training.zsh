export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONHASHSEED=222
export PL_SEED_WORKERS=1
export TORCH_USE_DETERMINISTIC_ALGORITHMS=1
# export OMP_NUM_THREADS=8

python k_mac_train_topoqa.py \
    --graph_dir graph_data \
    --train_label_file train.csv \
    --val_label_file val.csv \
    --attention_head 8 \
    --pooling_type "mean" \
    --batch_size 16 \
    --learning_rate 0.005 \
    --num_epochs 200 \
    --accumulate_grad_batches 32 \
    --seed 222 \
    --save_dir "./experiments"
