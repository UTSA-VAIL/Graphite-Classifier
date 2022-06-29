from pathlib import Path
from sklearn import multiclass
import torch
from torchgeometry.losses import DiceLoss
from torch.utils import data
from torchmetrics import JaccardIndex

import datahandler
from model import get_model
from trainer import train_model, train_multiclass_model
from helper import per_class_mIoU, get_binary_class_weights, get_multi_class_weights
from analysis import test_analysis_multi, test_analysis_multi_all, test_analysis_multi_all_nonbinary
from loss import DiceBCELoss#, DiceLoss, ExpDiceLoss
import argparse


threshold=0.2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size",default=4, type=int)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--model_type", default='resnet152', type=str)
    parser.add_argument("--no-binary", dest='binary', action='store_false')
    parser.set_defaults(binary=True)
    args = parser.parse_args()
    return args




def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        exp_dir.mkdir()

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.binary:
        jaccard = JaccardIndex(num_classes=2, threshold=threshold, average='weighted').to(device)
    else:
        jaccard = JaccardIndex(num_classes=args.num_classes, threshold=threshold, average='weighted').to(device)
    jaccard_test_all = JaccardIndex(num_classes=args.num_classes, average='weighted')
    metrics = {'miou': jaccard,
                #might add more later
                }

    if args.binary is True:
        models = {}
        mious = []
        dataloaders, test_images = datahandler.get_binary_dataloader(data_dir, args.num_classes, batch_size=args.batch_size, seed=100)

        for c in range(1, args.num_classes):
            #weights = get_binary_class_weights(iter(dataloaders[c]['Train'])).to(device)
            print(f"Training model: '{c}'...")
            model = get_model(args.model_type, outputchannels=1)
            model.to(device)

            #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
            criterion = DiceBCELoss()
            #criterion = DiceLoss(weight=weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

            model = train_model(model,
                            criterion,
                            dataloaders[c],
                            optimizer,
                            scheduler,
                            bpath=exp_dir,
                            metrics=metrics,
                            num_epochs=args.epochs,
                            device=device,
                            class_num=c,
                            num_classes = args.num_classes)

            if args.save_model:
                torch.save(model, f'{exp_dir}/{c}/weights_{c}.pt')

            miou = test_analysis_multi(model, jaccard, data_dir, exp_dir, test_images, c)
            mious.append(miou)
            models[c] = model

        test_analysis_multi_all(models, jaccard_test_all, data_dir, exp_dir, test_images, threshold)

    else:
        mious = []
        dataloaders, test_images = datahandler.get_multiclass_dataloader(data_dir, batch_size=args.batch_size, seed=50, num_classes=args.num_classes)

        #weights = get_multi_class_weights(iter(dataloaders['Train']), args.num_classes)
        print(f"Training model...")
        model = get_model(args.model_type, outputchannels=args.num_classes)
        model = model.to(device)

        #criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = DiceBCELoss(weight=weights)
        #criterion = DiceBCELoss()
        #criterion = DiceLoss()
        #criterion = ExpDiceLoss()
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = [DiceLoss(), torch.nn.CrossEntropyLoss()]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        model = train_multiclass_model(model,
                        criterion,
                        dataloaders,
                        optimizer,
                        scheduler,
                        bpath=exp_dir,
                        metrics=metrics,
                        num_epochs=args.epochs,
                        device=device,
                        num_classes = args.num_classes)

        if args.save_model:
            torch.save(model, f'{exp_dir}/weights.pt')

        #miou = test_analysis_multi(model, jaccard, data_dir, exp_dir, test_images, c)
        #mious.append(miou)
        #models[c] = model

        test_analysis_multi_all_nonbinary(model, jaccard_test_all, data_dir, exp_dir, test_images, threshold)



if __name__ == "__main__":
    main()