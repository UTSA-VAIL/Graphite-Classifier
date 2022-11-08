#!/bin/bash

#python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
python3 main.py \
--data_dir=/data/graphite/processed/Tiles/ \
--exp_dir=./experiments/Graphite \
--epochs=24 \
--model=vgg19 \
--batch_size=16 \
--num_classes=3 \
--image-size=384 \
--seed=100 \
#--distributed=True
#--eval=True \
#--enable_cw=True \