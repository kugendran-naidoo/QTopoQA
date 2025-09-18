export PYTORCH_ENABLE_MPS_FALLBACK=1

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
