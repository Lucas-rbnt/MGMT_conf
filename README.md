# Once and for All! MRI-Based Deep Learning Tools for MGMT Promoter Methylation Detection: a Thorough Evaluation
=========================================================================

This repository contains the source code associated to the proposed paper.

To use this repository, it is highly recommended to create a dedicated python 3.9 environment.

For instance, in you may run in your terminal:

```
$ conda install python=3.9 mgmt
```

Then, to install all requirements:

```
$ cd MGMT_conf/
$ pip install -r requirements.txt
```

### Training loop
=================

All that concerns the training of the models is usable via the script `train.py`.

A training loop can be launched as follows:

```
$ python train.py --method regular --data_path PATH/TO/THE/DATASET --data_type nifti --epochs 30 --modalities FLAIR T1wCE --path_to_save folder_with_pth/
```

In method you can either choose a regular training loop or a training loop with a confidence branch as in the paper presented by Terrance DeVries, Graham W. Taylor [Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/abs/1802.04865). 

Concerning the dataset, the paper uses MRIs from the [BraTS Challenge](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification).
You can either use [DICOM](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data) or [NIfTI](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) files.

The repository assumes that data are extracted directly from Kaggle and NIfTI files are extracted from the `archive.zip` at the root of the project conformly as instructions given in this [notebook](https://www.kaggle.com/code/dschettler8845/load-task-1-dataset-comparison-w-task-2-dataset).


Your data folder should have the following structure

```
- rsna-miccai-brain-tumor-radiogenomic-classification/
    - archive/ 
        - NIfTI files....
    - train/
        - DICOM files...
    - test/
        - DICOM files...
    sample_submission.csv
    train_labels.csv
```

### Confidence & OOD metrics
============================

Everything related to Confidence & OOD metrics can be found in the Jupyter notebook `Confidence.ipynb`. Results presented in the paper and found by the authors during experiments are currently displayed but feel free to modify the editable part to put your own trained model, data path and check your results....

### Logging
==========

All the useful and necessary information for the trainings and the realization of the paper have been logged directly on [Weights and Biases](https://wandb.ai/site). Therefore, it is also possible to use wandb as a logger with this repository. To do so, you have to add your entity name in the training command line.

```
$ python train.py .... --entity MyCoolWandbNickname
```

Otherwise, entity is set as `None` by default. Training progression will be displayed directly in your console.