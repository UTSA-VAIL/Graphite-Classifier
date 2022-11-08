from pathlib import Path
from sklearn import multiclass
import torch
from torchgeometry.losses import DiceLoss
from torch.utils import data
from torchmetrics import JaccardIndex
from segmentation_models_pytorch import losses

import datahandler
from model import get_model
from trainer import train_multiclass_model, train_semi_multiclass_model_2
from helper import per_class_mIoU, get_multi_class_weights
from analysis import test_analysis
from loss import DiceCCELoss
import argparse
import numpy as np
import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP




def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size",default=4, type=int)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--model_type", default='resnet18', type=str)
    parser.add_argument("--mode", dest='mode', default='supervised', type=str)
    parser.add_argument("--image-size", dest='size', default=512, type=int)
    parser.add_argument("--seed", dest='seed', default=100, type=int)
    parser.add_argument("--enable_cw", dest='enable_cw', default=False, type=bool)
    parser.add_argument("--distributed", dest='distributed', default=False, type=bool)
    parser.add_argument("--eval", dest='eval', default=False, type=bool)
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.set_defaults(binary=True)
    args = parser.parse_args()
    return args




def main(args):
    distributed = args.distributed
    local_rank = 0
    if distributed:
        init_distributed()
        local_rank = int(os.environ['LOCAL_RANK'])
    data_dir = Path(args.data_dir)

    jaccard = JaccardIndex(num_classes=args.num_classes, average='weighted').cuda()
    metrics = {'miou': jaccard,
                #might add more later
                }
    # Run a test if in eval mode
    if args.eval:
        exp_dir = args.exp_dir
        if local_rank == 0:
            model = torch.load(f"./{exp_dir}/weights.pt")
            _, test_images = datahandler.get_multiclass_dataloader(data_dir, batch_size=args.batch_size, num_classes=args.num_classes, seed=args.seed, distributed=distributed)
            test_analysis(model, jaccard, data_dir, exp_dir, test_images, args.size)
        return

    path_list = [args.exp_dir, args.model_type, args.mode, str(args.seed)+'s', str(args.epochs)+'e']
    if args.enable_cw:
        path_list.append("cw")
    if args.distributed:
        path_list.append("dist")
    exp_dir = Path('_'.join(path_list))

    if local_rank == 0 and not exp_dir.exists():
        exp_dir.mkdir()


    model = get_model(args.model_type, outputchannels=args.num_classes)
    model = model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    dataloaders, test_images = datahandler.get_multiclass_dataloader(data_dir, batch_size=args.batch_size, num_classes=args.num_classes, seed=args.seed, distributed=distributed)

    if args.enable_cw:
        print("Class weights enabled.")
        class_weights = get_multi_class_weights(dataloaders['Train'], args.num_classes)
    else:
        class_weights = None
        

    lr = 1e-4
    weight_decay=0
    criterion = DiceCCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(dataloaders['Train']), eta_min=1e-10)
    
    if args.mode == 'semi':
        if local_rank == 0:
            print('Semi-Supervised Training Enabled')
            print(f"Training model...")
        model = train_semi_multiclass_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    scheduler,
                    exp_dir=exp_dir,
                    metrics=metrics,
                    num_epochs=args.epochs,
                    local_rank=local_rank,
                    class_weights = class_weights,
                    distributed=distributed)
    else:
        if local_rank == 0:
            print(f"Training model...")
        model = train_multiclass_model(model,
                        criterion,
                        dataloaders,
                        optimizer,
                        scheduler,
                        exp_dir=exp_dir,
                        metrics=metrics,
                        num_epochs=args.epochs,
                        local_rank=local_rank,
                        class_weights=class_weights,
                        distributed=distributed)


    if local_rank == 0:
        torch.save(model, f'{exp_dir}/weights.pt')
        test_analysis(model, jaccard, data_dir, exp_dir, test_images, args.size)
    return



if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    args = get_args()
    main(args)