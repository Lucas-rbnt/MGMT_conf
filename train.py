# Standard libraries
import argparse
import random
import os
from typing import List, Union, Optional

# Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Local dependencies
from mgmt_conf.logger import logger
from mgmt_conf.utils import prepare_dataframes, use_pretrained_weights, training_loop
from mgmt_conf.datasets import MultimodalMGMTNiftiDataset, UnimodalMGMTNiftiDataset, UnimodalMGMTDicomDataset, MultimodalMGMTDicomDataset
from mgmt_conf.models import ResNet10Wrapper, MultimodalModel
from mgmt_conf.confidence import confidence_branch


method_options = ["regular", "confidence_branch"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="Whether to use regular training or confidence branch.", help="Training method.", choices=method_options)
    parser.add_argument("--data_path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--data_type", help="Data format, can be either dicom or nifti.", default="nifti", type=str)
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--path_to_save", type=str, required=True, help="Where to save torch models.")
    parser.add_argument("--modalities", nargs='+', type=Union[str, List[str]], default=["FLAIR", "T1wCE"], help="Modality or modalities to use. For example, to use FLAIR only, do in your cli: --modalities FLAIR. To use both FLAIR and T1wCE, do: --modalities FLAIR T1wCE.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--fusion_type", default="early", type=str, help="Whether to use early or intermediate fusion.")
    parser.add_argument("--entity", default=None, type=Optional[str], help="If provided, entity name for wandb logger.")
    parser.add_argument("--pretrained", default="True", type=str, help="Whether or not to use pretrained weights.")
    parser.add_argument("--project_name", default="MGMT Methylation", type=str, help="If provided, project name for wandb logger.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size used for training.")
    parser.add_argument("--tumor_centered", type=str, default="False", help="Whether or not to use segmentation mask to isolate the tumor.")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for training and data splitting.")
    parser.add_argument("--n_cpus", type=int, default=20, help="Number of cpus available for data processing.")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of gpus available for model training.")
    parser.add_argument("--lmbda", type=float, default=0.1, help="Lambda value for confidence loss.")
    parser.add_argument("--budget", type=float, default=0.3, help="Budget regularisation.")
    args = parser.parse_args()
    logger.info("Using {} files for this training!", args.data_type)

    ## Ensure reproducibility
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.pretrained == "True":
        args.pretrained = True
    else:
        args.pretrained = False

    if args.tumor_centered == "True":
        args.tumor_centered = True
        image_size = (96, 96)
        depth = 32
    else:
        args.tumor_centered = False
        image_size = (180, 180)
        depth = 64


    dataframe_train, dataframe_val = prepare_dataframes(os.path.join(args.data_path, "train_labels.csv"), args.random_state)
    if args.pretrained == "True":
        args.p
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Training with {} method", args.method)
    if not os.path.isdir(os.path.join(args.path_to_save, args.data_type, args.method)):
        os.makedirs(os.path.join(args.path_to_save, args.data_type, args.method))
    use_confidence_branch = True if args.method == "confidence_branch" else False
    if len(args.modalities) > 1:
        logger.info("Using multimodal data: {} with fusion type: {}", args.modalities, args.fusion_type)
        if args.data_type == "nifti":
            dataset_train = MultimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_train, args.modalities, args.fusion_type, tumor_centered=args.tumor_centered, image_size=image_size, depth=depth, augment=True)
            dataset_val = MultimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_val, args.modalities, args.fusion_type, tumor_centered=args.tumor_centered, image_size=image_size, depth=depth, augment=False)
        else:
            dataset_train = MultimodalMGMTDicomDataset(args.data_path, dataframe_train, args.modalities, image_size=image_size, depth=depth, split="train", augment=True)
            dataset_val = MultimodalMGMTDicomDataset(args.data_path, dataframe_val, args.modalities, image_size=image_size, depth=depth, split="train", augment=False)
        if args.fusion_type == "intermediate":
            logger.info("Creating model for intermediate fusion")
            base_model = ResNet10Wrapper(n_input_channels=1, n_classes=2, confidence_branch=False, embracenet=True)
            if args.pretrained:
                base_model = use_pretrained_weights(base_model, device=device)
            model = MultimodalModel(base_model, args.modalities, device, confidence_branch=use_confidence_branch)
            is_intermediate = True
        elif args.fusion_type == "early":
            if args.data_type == "dicom":
                raise ValueError("early fusion is not compatible with dicom-based dataset since MRI scans are not registered properly")
            logger.info("Creating model for early fusion")
            model = ResNet10Wrapper(n_input_channels=len(args.modalities), n_classes=2, confidence_branch=use_confidence_branch, embracenet=False)
            is_intermediate = False
        else:
            raise ValueError(f"fusion type {args.fusion_type} given is not supported, please use either 'late' or 'intermediate' instead")
            
    else:
        logger.info("Using unimodal data: {}", args.modalities)
        if args.data_type == "nifti":
            dataset_train = UnimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_train, args.modalities[0], tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=True)
            dataset_val = UnimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_val, args.modalities[0], tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=False)
        else:
            dataset_train = UnimodalMGMTDicomDataset(args.data_path, dataframe_train, args.modalities[0], image_size=(180, 180), depth=64, split="train", augment=True)
            dataset_val = UnimodalMGMTDicomDataset(args.data_path, dataframe_val, args.modalities[0], image_size=(180, 180), depth=64, split="train", augment=False)
        model = ResNet10Wrapper(n_input_channels=1, n_classes=2, confidence_branch=use_confidence_branch, embracenet=False)
        if args.pretrained:
            model = use_pretrained_weights(model, device)
        is_intermediate = False


    #logger.info("Computing the sampling policy...")
    #if os.path.exists("sampling_weights.npy"):
    #    weights = np.load("sampling_weights.npy")
    #else:
    #    class_weights = [1.1, 1.0]
    #    weights = [class_weights[w.item()] for _, w in dataset_train]
    #    np.save("sampling_weights.npy", np.array(weights))
            
    #sampler = WeightedRandomSampler(weights, len(dataset_train))
    #logger.info("Done!")

    train_dataloader = DataLoader(dataset_train, num_workers=args.n_cpus, shuffle=True, batch_size=args.batch_size, pin_memory=True, persistent_workers=True, drop_last=True)
    val_dataloader = DataLoader(dataset_val, num_workers=args.n_cpus, batch_size=args.batch_size, shuffle=False, pin_memory=True, persistent_workers=True, drop_last=False)

    if args.n_gpus > 1:
        logger.info("Using {} gpus to train the model", args.n_gpus)
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpus)))

    model = model.to(device)
    criterion = nn.NLLLoss() if use_confidence_branch else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    if use_confidence_branch:
        confidence_branch.train(model, args.epochs, train_dataloader, val_dataloader, criterion, 2, optimizer, scheduler, device, os.path.join(args.path_to_save, args.data_type, "confidence_branch", f"{args.fusion_type}_{'-'.join(args.modalities)}_tumor_centered_{args.tumor_centered}.pth"), lmbda=args.lmbda, budget=args.budget, is_intermediate=is_intermediate, project_name=args.project_name, entity=args.entity, name=os.path.join("Confidence_Branch", '-'.join(args.modalities), f"resnet_intermediate_{is_intermediate}_{args.fusion_type}"))
    else:
        _, _ = training_loop(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, epochs=args.epochs, path_to_save=os.path.join(args.path_to_save, args.data_type, "regular", f"{args.fusion_type}_{'-'.join(args.modalities)}_tumor_centered_{args.tumor_centered}.pth"), entity=args.entity, prefix= os.path.join("RegularTraining", '-'.join(args.modalities), f"resnet_intermediate_{is_intermediate}_{args.fusion_type}"), project_name=args.project_name, is_intermediate=is_intermediate)
            
        logger.info("End training on {} MRI scans", args.modalities)