# Standard libraries
from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy as np

# Third-party libraries
import torch

# Local dependencies
from .base import _BaseMGMTDicomDataset, _BaseMGMTNiftiDataset

__all__ = [
    "UnimodalMGMTNiftiDataset",
    "MultimodalMGMTNiftiDataset",
    "UnimodalMGMTDicomDataset",
    "MultimodalMGMTDicomDataset",
]


class UnimodalMGMTNiftiDataset(_BaseMGMTNiftiDataset):
    """
    Class for unimodal nifti-based dataset.

    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data, this is to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
        tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modality: str,
        tumor_centered: bool,
        image_size: Tuple[int, int],
        depth: int,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
            tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(UnimodalMGMTNiftiDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            tumor_centered=tumor_centered,
            image_size=image_size,
            depth=depth,
            augment=augment,
        )
        self.modality = modality

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Reads and returns the i-th data point in the dataset.
        """
        patient_id = super()._get_patient_id(idx)
        x = super()._prepare_nifti_volume(patient_id, self.modality)
        x = super()._preprocess(x)

        target = super()._get_target(idx)

        return torch.tensor(x).float(), torch.tensor(target)


class MultimodalMGMTNiftiDataset(_BaseMGMTNiftiDataset):
    """
    Class for multimodal nifti-based dataset.

    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data, this is to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
        fusion (str) : How modalities will be fused. 2 different types according to the work done: `intermediate` or `early` fusion.
        tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modalities: Tuple[str, ...],
        fusion: str,
        tumor_centered: bool,
        image_size: Tuple[int, int],
        depth: int,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
            fusion (str) : How modalities will be fused. 2 different types according to the work done: `intermediate` or `early` fusion.
            tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(MultimodalMGMTNiftiDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            tumor_centered=tumor_centered,
            image_size=image_size,
            depth=depth,
            augment=augment,
        )
        self.modalities = modalities
        self.fusion = fusion

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.tensor, torch.tensor], Tuple[Dict[str, torch.tensor], torch.tensor]
    ]:
        """
        Reads and returns the i-th data point in the dataset.
        """
        patient_id = super()._get_patient_id(idx)
        target = super()._get_target(idx)

        item = OrderedDict()
        if self.fusion == "early":
            seed = np.random.randint(0, 10000) + idx
        for modality in self.modalities:
            if self.fusion == "early":
                np.random.seed(seed)
            x = super()._prepare_nifti_volume(patient_id, modality)
            x = super()._preprocess(x)
            item[modality] = torch.tensor(x).float()

        if self.fusion == "early":
            return torch.cat(tuple(item.values()), 0), torch.tensor(target)

        else:
            return item, torch.tensor(target)


class UnimodalMGMTDicomDataset(_BaseMGMTDicomDataset):
    """
    Class for unimodal DICOM-based dataset.

    Attributes:
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modality: str,
        image_size: Tuple[int, int],
        depth: int,
        split: str,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(UnimodalMGMTDicomDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            image_size=image_size,
            depth=depth,
            split=split,
            augment=augment,
        )
        self.modality = modality

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Reads and returns the i-th data point in the dataset.
        """
        x = super()._prepare_dicom_volume(idx, self.modality)
        x = super()._preprocess(x)

        target = super()._get_target(idx)

        return torch.tensor(x).float(), torch.tensor(target)


class MultimodalMGMTDicomDataset(_BaseMGMTDicomDataset):
    """
    Class for multimodal DICOM-based dataset.

    Attributes:
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w") for instance.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modalities: Tuple[str, ...],
        image_size: Tuple[int, int],
        depth: int,
        split: str,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w") for instance.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(MultimodalMGMTDicomDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            image_size=image_size,
            depth=depth,
            split=split,
            augment=augment,
        )
        self.modalities = modalities

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
        """
        Reads and returns the i-th data point in the dataset.
        """
        item = OrderedDict()
        target = super()._get_target(idx)
        for modality in self.modalities:
            x = super()._prepare_dicom_volume(idx, modality)
            x = super()._preprocess(x)
            item[modality] = torch.tensor(x).float()

        else:
            return item, torch.tensor(target)
