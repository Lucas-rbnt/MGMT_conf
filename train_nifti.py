# Standard libraries
import argparse
from email.mime import image
import random
from typing import List
import os

# Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# Local dependencies
from mgmt_conf.logger import logger
from mgmt_conf.utils import prepare_dataframes, use_pretrained_weights, training_loop
from mgmt_conf.datasets import MultimodalMGMTNiftiDataset, UnimodalMGMTNiftiDataset, UnimodalMGMTDicomDataset, MultimodalMGMTDicomDataset
from mgmt_conf.models import ResNet10Wrapper, MultimodalModel
from mgmt_conf.confidence import confidence_branch


method_options = ["regular", "confidence_branch"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="osef", help="the training method")#, choices=method_options)
    parser.add_argument("--data_path", type=str, required=True, help="dataset path.")
    parser.add_argument("--data_type", default="nifti", type=str)
    parser.add_argument("--epochs", type=int, default=60, help="number of used epochs")
    parser.add_argument("--path_to_save", type=str, required=True, help="where to save pth models.")
    parser.add_argument("--modalities", type=List[str], default=("FLAIR", "T1wCE"), help="modality or modalities to used.")
    parser.add_argument("--lr", type=float, default=1e-5, help="training's learning rate.")
    parser.add_argument("--fusion_type", default="early", type=str, help="Whether to use early or intermediate fusion")
    parser.add_argument("--entity", default="luktenib", type=str, help="entity name for weights and biases")
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--project_name", default="MGMT Methylation", type=str, help="wandb project")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size used for training")
    parser.add_argument("--tumor_centered", type=str, default=False, help="Whether or not to use segmentation mask to isolate the tumor")
    parser.add_argument("--random_state", type=int, default=42, help="the default seed for training and data splitting")
    parser.add_argument("--n_cpus", type=int, default=40, help="Number of cpus available for data processing")
    parser.add_argument("--n_gpus", type=int, default=4, help="Number of gpus available for model training")
    parser.add_argument("--lmbda", type=float, default=0.1, help="Lambda value for confidence loss")
    parser.add_argument("--budget", type=float, default=0.3, help="budget regularisation")
    args = parser.parse_args()
    logger.info("Using {} files for this training!", args.data_type)
    logger.info("Preparing training for {} method", args.method)


    dataframe_train, dataframe_val = prepare_dataframes(os.path.join(args.data_path, "train_labels.csv"), args.random_state)

    ## Ensure reproducibility
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    for method in ["regular", "confidence_branch"]:
        if not os.path.isdir(os.path.join(args.path_to_save, args.data_type, method)):
            os.makedirs(os.path.join(args.path_to_save, args.data_type, method))
        use_confidence_branch = True if method == "confidence_branch" else False
        for modalities, fusion_type in [(("FLAIR",), 'early'), (("T1wCE",), 'early'), (("FLAIR", "T1wCE"), 'early'),(("FLAIR", "T1wCE"), 'intermediate'), (("FLAIR", "T1wCE", "T1w", "T2w"), "early")]:
            if len(modalities) > 1:
                logger.info("Using multimodal data: {} with fusion type: {}", modalities, fusion_type)
                if args.data_type == "nifti":
                    dataset_train = MultimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_train, modalities, fusion_type, tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=True)
                    dataset_val = MultimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_val, modalities, fusion_type, tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=False)
                else:
                    dataset_train = MultimodalMGMTDicomDataset(args.data_path, dataframe_train, modalities, image_size=(180, 180), depth=64, split="train", augment=True)
                    dataset_val = MultimodalMGMTDicomDataset(args.data_path, dataframe_val, modalities, image_size=(180, 180), depth=64, split="train", augment=False)
                if fusion_type == "intermediate":
                    logger.info("Creating model for intermediate fusion")
                    base_model = ResNet10Wrapper(n_input_channels=1, n_classes=2, confidence_branch=False, embracenet=True)
                    if args.pretrained:
                        base_model = use_pretrained_weights(base_model, device=device)
                    model = MultimodalModel(base_model, modalities, device, confidence_branch=use_confidence_branch)
                    epochs = 60
                    is_intermediate = True
                elif fusion_type == "early":
                    if args.data_type == "dicom":
                        raise ValueError("early fusion is not compatible with dicom-based dataset since MRI scans are not registered properly")
                    logger.info("Creating model for early fusion")
                    model = ResNet10Wrapper(n_input_channels=len(modalities), n_classes=2, confidence_branch=use_confidence_branch, embracenet=False)
                    epochs = 30
                    is_intermediate = False
                else:
                    raise ValueError(f"fusion type {fusion_type} given is not supported, please use either 'late' or 'intermediate' instead")
            
            else:
                logger.info("Using unimodal data: {}", modalities)
                if args.data_type == "nifti":
                    dataset_train = UnimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_train, modalities[0], tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=True)
                    dataset_val = UnimodalMGMTNiftiDataset(os.path.join(args.data_path, 'archive'), dataframe_val, modalities[0], tumor_centered=args.tumor_centered, image_size=(180, 180), depth=64, augment=False)
                else:
                    dataset_train = UnimodalMGMTDicomDataset(args.data_path, dataframe_train, modalities[0], image_size=(180, 180), depth=64, split="train", augment=True)
                    dataset_val = UnimodalMGMTDicomDataset(args.data_path, dataframe_val, modalities[0], image_size=(180, 180), depth=64, split="train", augment=False)
                model = ResNet10Wrapper(n_input_channels=1, n_classes=2, confidence_branch=use_confidence_branch, embracenet=False)
                if args.pretrained:
                    model = use_pretrained_weights(model, device)
                is_intermediate = False
                epochs = 30


            logger.info("Computing the sampling policy...")
            if os.path.exists("sampling_weights.npy"):
                weights = np.load("sampling_weights.npy")
            else:
                class_weights = [1.1, 1.0]
                weights = [class_weights[w.item()] for _, w in dataset_train]
                np.save("sampling_weights.npy", np.array(weights))
            
            sampler = WeightedRandomSampler(weights, len(dataset_train))
            logger.info("Done!")

            train_dataloader = DataLoader(dataset_train, num_workers=args.n_cpus, shuffle=True, batch_size=args.batch_size, pin_memory=True, persistent_workers=True, drop_last=True)
            val_dataloader = DataLoader(dataset_val, num_workers=args.n_cpus, batch_size=args.batch_size, shuffle=False, pin_memory=True, persistent_workers=True, drop_last=True)

            if args.n_gpus > 1:
                logger.info("Using {} gpus for training the model", args.n_gpus)
                model = nn.DataParallel(model, device_ids=list(range(args.n_gpus)))

            model = model.to(device)
            criterion = nn.NLLLoss() if use_confidence_branch else nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
            if use_confidence_branch:
                confidence_branch.train(model, epochs, train_dataloader, val_dataloader, criterion, 2, optimizer, scheduler, device, os.path.join(args.path_to_save, "confidence_branch", f"{fusion_type}_{'-'.join(modalities)}_tumor_centered_{args.tumor_centered}.pth"), lmbda=args.lmbda, budget=args.budget, is_intermediate=is_intermediate, project_name=args.project_name, entity=args.entity, name=os.path.join("Confidence_Branch", '-'.join(modalities), f"resnet_intermediate_{is_intermediate}_{fusion_type}"))
            else:
                _, _ = training_loop(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, epochs=epochs, path_to_save=os.path.join(args.path_to_save, "regular", f"{fusion_type}_{'-'.join(modalities)}_tumor_centered_{args.tumor_centered}.pth"), entity=args.entity, prefix= os.path.join("RegularTraining", '-'.join(modalities), f"resnet_intermediate_{is_intermediate}_{fusion_type}"), project_name=args.project_name, is_intermediate=is_intermediate)
            
            logger.info("End training on {} MRI scans", modalities)