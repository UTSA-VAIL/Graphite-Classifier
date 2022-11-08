#!/bin/bash

#python3 main.py \
python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_dir=/data/graphite/processed/Tiles/ \
--exp_dir=experiments/Graphite_vgg19_supervised_100s_24e_dist \
--num_classes=3 \
--image-size=384 \
--seed=100 \
--eval=True \
--distributed=True