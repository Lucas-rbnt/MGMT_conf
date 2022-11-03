# Standard libraries
import os
from abc import abstractmethod
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

# Third-party libraries
from torch.utils.data import Dataset

# Local dependencies
from ..preprocessing.utils import (
    clahe,
    load_complete_mri,
    normalize_intensity,
    random_flip,
    random_noise,
    random_rotate,
)


class _BaseMGMTDataset(Dataset):
    """
    Base class for every dataset that is used.


    Attributes:
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        image_size: Tuple[int, int],
        depth: int,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        self.base_path = base_path
        self.dataframe = dataframe
        self.image_size = image_size
        self.depth = depth
        self.augment = augment

    def _get_patient_id(self, idx: int) -> str:
        """
        Private method that returns the patient id associated to an index.
        """
        return str(self.dataframe["BraTS21ID"].iloc[idx]).zfill(5)

    def _get_target(self, idx: int) -> int:
        """
        Private method that returns the target associated to an index.
        """
        return self.dataframe["MGMT_value"].iloc[idx]

    @staticmethod
    def _crop_on_nonzero_voxels(x: np.ndarray) -> np.ndarray:
        """
        Static method cropping the volume around non-zero voxels only.

        Args:
            x (np.ndarray) : the initial volume.

        Returns:
            np.ndarray : the cropped volume.
        """
        argwhere = np.argwhere(x)
        min_z, min_x, min_y = (
            np.min(argwhere[:, 0]),
            np.min(argwhere[:, 1]),
            np.min(argwhere[:, 2]),
        )
        max_z, max_x, max_y = (
            np.max(argwhere[:, 0]),
            np.max(argwhere[:, 1]),
            np.max(argwhere[:, 2]),
        )

        x = x[min_z:max_z, min_x:max_x, min_y:max_y]

        return x

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Returns a preprocessed version of an input.

        Args:
            x (np.ndarray) : the initial volume.

        Returns:
            np.ndarray : the volume after being resized, augmented, normalized and formatted.
        """
        [height, width, depth] = x.shape
        scale = [
            self.image_size[0] * 1.0 / height,
            self.image_size[1] * 1.0 / width,
            self.depth * 1 / depth,
        ]

        x = ndimage.zoom(x, scale, order=3)
        x = clahe(x)
        x = normalize_intensity(x)
        if self.augment:
            x = random_noise(x)

        x = np.expand_dims(x, 0)

        return x

    @abstractmethod
    def __getitem__(self, idx: int) -> NotImplementedError:
        """
        Reads and returns the i-th data point in the dataset.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns dataset's length.
        """
        return len(self.dataframe)


class _BaseMGMTNiftiDataset(_BaseMGMTDataset):
    """
    Base class for every Nifti-based dataset.


    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data, this is to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    conversion_dict = {"FLAIR": "flair", "T1wCE": "t1ce", "T1w": "t1", "T2w": "t2"}

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        tumor_centered: bool,
        image_size: Tuple[int],
        depth: int,
        augment: bool,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            tumor_centered (bool) : In Nifti files, segmentation masks are available. Whether or not to use only a volume around the tumor.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(_BaseMGMTNiftiDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            image_size=image_size,
            depth=depth,
            augment=augment,
        )
        self.tumor_centered = tumor_centered

    def _prepare_mask(self, patient_id: str) -> None:
        """
        Private method loading segmentation mask edges.

        Args:
            patient_id (str) : the segmentation mask of this specific patient id is loaded.
        """
        mask = nib.load(
            f"{self.base_path}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_seg.nii.gz"
        ).get_fdata()
        argwhere = np.argwhere(mask)
        min_x, min_y, min_z = (
            np.min(argwhere[:, 0]),
            np.min(argwhere[:, 1]),
            np.min(argwhere[:, 2]),
        )
        max_x, max_y, max_z = (
            np.max(argwhere[:, 0]),
            np.max(argwhere[:, 1]),
            np.max(argwhere[:, 2]),
        )
        self.mask = (min_x, min_y, min_z, max_x, max_y, max_z)

    def _prepare_nifti_volume(self, patient_id: str, modality: str) -> np.ndarray:
        """
        Private method loading the volume and performing tumor extraction or cropping it around non-zero voxels.

        Args:
            patient_id (str) : preprocessing on this specific patient id volume is done.
            modality (str) : the used modality.

        Returns:
            np.ndarray : the volume of interest.
        """
        x = nib.load(
            f"{self.base_path}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{self.conversion_dict[modality]}.nii.gz"
        ).get_fdata()

        if self.tumor_centered:
            self._prepare_mask(patient_id)
            x = x[
                self.mask[0] : self.mask[3],
                self.mask[1] : self.mask[4],
                self.mask[2] : self.mask[5],
            ]

        else:
            x = super()._crop_on_nonzero_voxels(x)

        if self.augment:
            x = random_rotate(x)
            p = np.random.rand(1)[0]
            if p > 0.5:
                x = random_flip(x)

        return x


class _BaseMGMTDicomDataset(_BaseMGMTDataset):
    """
    Base class for DICOM-based dataset.

    Attributes:
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
        augment (bool) : whether or not to use data augmentation.
    """

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        image_size: Tuple[int],
        depth: int,
        split: str,
        augment: bool,
    ) -> None:
        """
        Class constructor

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
            augment (bool) : whether or not to use data augmentation.
        """
        super(_BaseMGMTDicomDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            image_size=image_size,
            depth=depth,
            augment=augment,
        )
        self.split = split

    def _prepare_dicom_volume(self, idx: int, modality: str) -> np.ndarray:
        """
        Private method loading the volume and cropping it around non-zero voxels.

        Args:
            idx (int) : index of the dataset.
            modality (str) : the used modality.

        Returns:
            np.ndarray : the volume of interest.
        """
        patient_id, path_to_split = self._get_patient_id_and_path_to_split(idx)
        x = load_complete_mri(
            path_to_split,
            patient_id,
            modality,
            target_size=None,
            voi_lut=False,
            rotate=0,
            scale=False,
        )
        x = super()._crop_on_nonzero_voxels(x)

        if self.augment:
            x = random_rotate(x)
            p = np.random.rand(1)[0]
            if p > 0.5:
                x = random_flip(x)
        return x

    def _get_patient_id_and_path_to_split(self, idx: int) -> Tuple[str, str]:
        """
        Private method to return patient id and the path to split data. This method is implemented to deal with the data representation of dicom-datasets

        - folder
            -train
                - patient_id
                    - FLAIR
                    - T2w
                    - T1w
                    - T1wCE
                - patient_id
                ...
            - test
            ...
        """
        return str(self.dataframe["BraTS21ID"].iloc[idx]).zfill(5), os.path.join(
            self.base_path, self.split
        )


class _BaseMGMTPrivateDataset(_BaseMGMTDataset):
    """
    Base class for every private dataset.


    Attributes:
        conversion_dict (Dict[str, str]) : Since the format was not consistent between dicom and nifti data and kaggle and private datasets, this dictionnary helps to keep some continuity in the instantiation of these classes.
        base_path (str) : the path to the folder containing the data.
        dataframe (pd.DataFrame) : the dataframe with patient id and target.
        image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
        depth (int) : the depth to which volumes are resized.
    """

    conversion_dict = {
        "T1 gado": "T1wCE",
        "T1 Gado": "T1wCE",
        "T1 Gado post op": "T1wCE",
        "T1 Gado postop": "T1wCE",
        "T1 3D gado": "T1wCE",
        "3D T1 Gado": "T1wCE",
        "Ax 3D T1 Gado": "T1wCE",
        "3D T1 gado": "T1wCE",
        "Ax T1 gado": "T1wCE",
        "T1 3D Gado": "T1wCE",
        "3D t1 gado": "T1wCE",
        "T2 flair": "FLAIR",
        "Ax T2 flair": "FLAIR",
        "Ax T2 Flair": "FLAIR",
        "Ax Flair": "FLAIR",
        "Flair Ax": "FLAIR",
        "T2 Flair Ax": "FLAIR",
        "Flair Long": "FLAIR",
        "Flair long": "FLAIR",
        "3D flair": "FLAIR",
        "3D Flair": "FLAIR",
        "Flair": "FLAIR",
        "T2 Flair": "FLAIR",
        "Flair post op": "FLAIR",
        "Flair postop": "FLAIR",
    }

    def __init__(
        self,
        base_path: str,
        dataframe: "pd.DataFrame",  # noqa
        image_size: Tuple[int, int],
        depth,
    ) -> None:
        """
        Class constructor.

        Args:
            base_path (str) : the path to the folder containing the data.
            dataframe (pd.DataFrame) : the dataframe with patient id and target.
            image_size (Tuple[int, int]) : the image resolution to which volumes are resized.
            depth (int) : the depth to which volumes are resized.
        """
        super(_BaseMGMTPrivateDataset, self).__init__(
            base_path=base_path,
            dataframe=dataframe,
            image_size=image_size,
            depth=depth,
            augment=False,
        )

    def _get_patient_id(self, idx: int) -> str:
        """
        Private method to get patient id according to the index of the dataframe associated to the test set.
        """
        return self.dataframe.iloc[idx].NNNPP

    def _get_target(self, idx) -> float:
        """
        Private method to get methylation percentage according to the index of the dataframe associated to the test set.
        """
        return self.dataframe.iloc[idx]["% Méthylé"]

    def _prepare_private_volume(self, patient_id: str, modality: str) -> np.ndarray:
        """
        Private method loading the volume and cropping it around non-zero voxels.

        Args:
            patient_id (str) : preprocessing on this specific patient id volume is done.
            modality (str) : the used modality.

        Returns:
            np.ndarray : the volume of interest.
        """

        if modality == "FLAIR":
            n = "FL"
        else:
            n = "T1CE"

        x = nib.load(
            f"{self.base_path}/{patient_id}/{n}_to_SRI_brain.nii.gz"
        ).get_fdata()

        x = super()._crop_on_nonzero_voxels(x)
        return x
