train_data_size=16
val_data_size=128

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir '/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/agent-verl/data/verl-agent/' \