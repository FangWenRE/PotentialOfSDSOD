#!/bin/bash
model=$1
target=$2
cuda=$3
data_path=$4

CUDA_VISIBLE_DEVICES=$cuda python3 -u train.py $model --data_path=$data_path --weight="" --gpus=$cuda --save_tar=$target
wait
CUDA_VISIBLE_DEVICES=$cuda python3 -u test.py $model --gpus=$cuda --weight="weight/$target/best.pth"  --save_tar=$target --save

# nohup bash train_test.sh maxsum res 2 dataset/DUTS-TR > ./logs/training/res.log 2>&1 &
# ps -ef | grep train.py | grep -v grep | awk '{print $2}' | xargs kill -9
