# Standard libraries
import os
from collections import OrderedDict
from typing import Dict, Tuple, Union

# Third-party libraries
import torch

# Local dependencies
from .base import _BaseMGMTPrivateDataset

__all__ = [
    "UnimodalMGMTPrivateDataset",
    "MultimodalMGMTPrivateDataset",
    "UnimodalTumorCenteredMGMTPrivateDataset",
    "MultimodalTumorCenteredMGMTPrivateDataset",
]


class UnimodalMGMTPrivateDataset(_BaseMGMTPrivateDataset):
    """
    Class for unimodal private dataset.


    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data and kaggle and private datasets, this dictionnary helps to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modality: str,
        image_size: Tuple[int, int],
        depth: int,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
        """
        super(UnimodalMGMTPrivateDataset, self).__init__(
            base_path=base_path, dataframe=dataframe, image_size=image_size, depth=depth
        )
        self.modality = modality

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Reads and returns the i-th data point in the dataset.
        """
        patient_id = super()._get_patient_id(idx)

        x = super()._prepare_private_volume(patient_id, self.modality)
        x = super()._preprocess(x)

        target = super()._get_target(idx)

        return torch.tensor(x).float(), torch.tensor(target)


class MultimodalMGMTPrivateDataset(_BaseMGMTPrivateDataset):
    """
    Class for multimodal private dataset.


    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data and kaggle and private datasets, this dictionnary helps to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
        fusion (str) : How modalities will be fused. 2 different types according to the work done: `intermediate` or `early` fusion.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        modalities: Tuple[str, ...],
        fusion: str,
        image_size: Tuple[int, int],
        depth: int,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
            fusion (str) : How modalities will be fused. 2 different types according to the work done: `intermediate` or `early` fusion.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
        """
        super(MultimodalMGMTPrivateDataset, self).__init__(
            base_path=base_path, dataframe=dataframe, image_size=image_size, depth=depth
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

        for modality in self.modalities:
            x = super()._prepare_private_volume(patient_id, modality)
            x = super()._preprocess(x)
            item[modality] = torch.tensor(x).float()

        if self.fusion == "early":
            return torch.cat(tuple(item.values()), 0), torch.tensor(target)

        else:
            return item, torch.tensor(target)


class UnimodalTumorCenteredMGMTPrivateDataset(UnimodalMGMTPrivateDataset):
    """
    Class for unimodal, tumor-centered dataset.

    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data and kaggle and private datasets, this dictionnary helps to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
    """

    def __init__(self, base_path, dataframe, modality):
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modality (str) : the MRI modality to use. Can be among "FLAIR", "T1wCE"....
        """
        super(UnimodalTumorCenteredMGMTPrivateDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            modality=modality,
            image_size=None,
            depth=None,
        )

    def __getitem__(self, idx):
        """
        Reads and returns the i-th data point in the dataset.
        """
        patient_id = super()._get_patient_id(idx)
        target = super()._get_target(idx)
        return torch.load(
            os.path.join(self.base_path, patient_id, f"{self.modality.lower()}.pt")
        ), torch.tensor(target)


class MultimodalTumorCenteredMGMTPrivateDataset(MultimodalMGMTPrivateDataset):
    """
    Class for multimodal, tumor-centered dataset.


    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data and kaggle and private datasets, this dictionnary helps to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
        fusion (str) : How modalities will be fused. 2 different types according to the work done: `intermediate` or `early` fusion.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
    """

    def __init__(self, base_path, dataframe, modalities):
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            modalities (Tuple[str, ...]) : the MRI modalities to use. Can be among "FLAIR", "T1wCE".... Format is ("T1w", "T2w").
        """
        super(MultimodalTumorCenteredMGMTPrivateDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            modality=modalities,
            image_size=None,
            depth=None,
        )

    def __getitem__(self, idx: int):
        """
        Reads and returns the i-th data point in the dataset.
        """
        patient_id = super()._get_patient_id(idx)
        target = super()._get_target(idx)
        x = list()
        for modality in self.modalities:
            x.append(
                torch.load(
                    os.path.join(self.base_path, patient_id, f"{modality.lower()}.pt")
                )
            )

        return torch.stack(x), torch.tensor(target)
