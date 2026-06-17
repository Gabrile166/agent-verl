#!/bin/bash

new_file="/mnt/shared-storage-user/ailab-hs/zhaojun/tangjixin/agent-verl/examples/ppo_trainer/run_alfworld.sh"

chmod +x $new_file

set -ex
REPLICAS=1
name=simtask-10141801
rjob submit -e DISTRIBUTED_JOB=true \
    --image=registry.h.pjlab.org.cn/ailab/xpuyu:torch-2.6.0-45d96d5f-0607 \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 96  --memory 480000 --charged-group hs_gpu \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -- bash -ecx $new_file