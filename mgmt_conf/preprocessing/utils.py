# Standard libraries
import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mclahe as mc
import numpy as np

# Third-party libraries
import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut
from scipy import ndimage

__all__ = [
    "load_single_file",
    "load_partial_mri",
    "load_complete_mri",
    "load_volumes",
    "normalize_intensity",
    "clahe",
    "random_noise",
    "random_flip",
    "random_rotate",
    "display_scans_cut",
    "create_animation",
]


def load_single_file(
    path: str,
    target_size: Optional[Tuple[int]] = None,
    voi_lut: bool = True,
    rotate: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the specified file in the dicom format.
    Inspired from https://github.com/FirasBaba/rsna-resnet10/blob/main/working/utils.py

    Args:
        path (str) : the dicom file to read from.
        target_size (Optional[Tuple[int]]) : if not `None`, the format at which we want the image to be resized. Default to `None`.
        voi_lut (bool) : whether or not to apply VOI LUT on the given image. Default to `True`.
        rotate (int) : integer between 0 and 1 and defining the corresponding rotation. Default to 0.
        scale (bool) : whether or not to put the 2D scans in the range (0, 255). Default to `True`.
    Returns:
        np.darray : the corresponding array.
    """
    img = pydicom.read_file(path)
    if voi_lut:
        img = apply_voi_lut(img.pixel_array, img)
    else:
        img = img.pixel_array
    if scale:
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        img = (img * 255).astype(np.uint8)
    if rotate > 0:
        rotations = [
            0,
            cv2.ROTATE_180,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]
        img = cv2.rotate(img, rotations[rotate])
    if target_size:
        img = cv2.resize(img, target_size)
    return img


def load_complete_mri(
    base_path: str,
    patient_id: int,
    scan: str,
    target_size: Optional[Tuple[int]] = None,
    voi_lut: bool = True,
    rotate: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the whole desired scan.

    Args:
        base_path (str) : the path to the folder containing every scan for every patient.
        patiend_id (int) : the patient files to read from.
        scan (str) : the desired scan wanted for this particular patient.
        target_size (Optional[Tuple[int]]) : if not `None`, the format at which we want the image to be resized. Default to `None`.
        voi_lut (bool) : whether or not to apply VOI LUT on the given image. Default to `True`.
        rotate (int) : integer between 0 and 2 and defining the corresponding rotation. Default to 0.
        scale (bool) : whether or not to put the 2D scans in the range (0, 255). Default to `True`.

    Returns:
        np.ndarray : the corresponding scan.
    """
    return load_partial_mri(
        base_path,
        patient_id,
        scan,
        target_size,
        voi_lut,
        rotate,
        startpoint=0,
        scale=scale,
    )


def load_partial_mri(
    base_path: str,
    patient_id: int,
    scan: str,
    target_size: Optional[Tuple[int]] = None,
    voi_lut=True,
    rotate: int = 0,
    startpoint: int = 0,
    endpoint: Optional[int] = None,
    scale: bool = True,
) -> np.ndarray:
    """
    Returns an array containing a part of the desired scan.

    Args:
        base_path (str) : the path to the folder containing every scan for every patient.
        patiend_id (int) : the patient files to read from.
        scan (str) : the desired scan wanted for this particular patient.
        target_size (Optional[Tuple[int]]) : if not `None`, the format at which we want the image to be resized. Default to `None`.
        voi_lut (bool) : whether or not to apply VOI LUT on the given image. Default to `True`.
        rotate (int) : integer between 0 and 2 and defining the corresponding rotation. Default to 0.
        startpoint (int) : integer defining the starting point (on the time axis) at which to begin the loading process of a given MRI. Default to 0.
        endpoint (Optional[int]) : integer defining the ending point (on the time axis) at which to begin the loading process of a given MRI. If `None`, the loading process will load until the very last file. Default to `None`.
        scale (bool) : whether or not to put the 2D scans in the range (0, 255). Default to `True`.

    Returns:
        np.ndarray : the corresponding scan.

    """
    full_path: str = os.path.join(base_path, str(patient_id).zfill(5), scan)
    dicom_filepaths: List[str] = sorted(
        glob.glob(os.path.join(full_path, "*.dcm")),
        key=lambda x: int(x.split("-")[-1][:-4]),
    )
    volume = load_single_file(
        dicom_filepaths[startpoint],
        target_size=target_size,
        voi_lut=voi_lut,
        rotate=rotate,
        scale=scale,
    )
    volume = volume[:, :, np.newaxis]
    if not endpoint:
        endpoint = len(dicom_filepaths)
    for i in range(startpoint + 1, endpoint):
        img = load_single_file(
            dicom_filepaths[i],
            target_size=target_size,
            voi_lut=voi_lut,
            rotate=rotate,
            scale=scale,
        )
        img = img[:, :, np.newaxis]
        volume = np.concatenate((volume, img), 2)

    return volume


def load_volumes(
    base_path: str,
    patient_id: int,
    scans: Tuple[str] = ("FLAIR", "T1w", "T1wCE", "T2w"),
    scale: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Returns a dictionary where every MRI array can be accessed through the MRI type.

    Args:
        base_path (str) : the path to the folder containing every scan for every patient.
        patiend_id (int) : the patient files to read from.
        scans (Tuple[str]) : the desired scans wanted for this particular patient. Default to `("FLAIR", "T1w", "T1wCE", "T2w")`.
        scale (bool) : whether or not to put the 2D scans in the range (0, 255). Default to `True`.
    Returns:
        Dict[str, np.ndarray] : a dictionnary where each element of scans is a key and the corresponding value is the array of the given MRI.


    """
    volumes = dict()
    for scan in scans:
        volume = load_complete_mri(base_path, patient_id, scan=scan, scale=scale)
        volumes[scan] = volume

    return volumes


def clahe(x: np.ndarray) -> np.ndarray:
    """
    Returns the input after applying Multidimensional Contrast Limited Adaptive Histogram Equalization https://github.com/VincentStimper/mclahe

    Args:
        x (np.ndarray) : the input that will be preprocessed.

    Returns:
        np.ndarray : the given input with mclahe applied.

    """

    x = mc.mclahe(
        x,
        kernel_size=[32, 32, 8],
        n_bins=128,
        clip_limit=0.05,
        adaptive_hist_range=False,
    )
    return x.astype(np.float32)


def normalize_intensity(x: np.ndarray) -> np.ndarray:
    """
    Returns a normalized version (on non-zero voxels only) of an input.

    Args:
        x (np.ndarray) : the given input.

    Returns:
        np.ndarray : the normalized input.
    """
    mask = np.where(x != 0.0)
    mean, std = x[mask].mean(), x[mask].std()
    return (x - mean) / (std + 1e-9)


def random_rotate(x: np.ndarray, low: int = -10, high: int = 10) -> np.ndarray:
    """
    Returns a rotated version of the volume.

    Args:
        x (np.ndarray) : the input.
        low (int) : represents the angle lower bound that can be applied.
        high (int) : the angle upper bound that can be applied.

    Returns:
        np.ndarray : the rotated array.
    """
    angle = np.random.randint(low=low, high=high + 1)
    return ndimage.rotate(x, angle, axes=(0, 1))


def random_flip(x: np.ndarray, axis: Tuple[int] = (0, 1)) -> np.ndarray:
    """
    Returns a flipped version of the array

    Args:
        x (np.ndarray) : the input.
        axis (Tuple[int]) : axes that will be permuted.

    Returns:
        np.ndarray : the flipped array.
    """
    axe = np.random.randint(0, 1)
    x = np.asarray(x).swapaxes(axis[axe], 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis[axe])
    return x


def random_noise(x: np.ndarray, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """
    Returns a perturbed version of the input. Pertubation here is gaussian noise.

    Args:
        x (np.ndarray) : the input.
        mean (float) : the noise's mean.
        std (float) : the noise's standard deviation.

    Returns:
        np.ndarray : the perturbed array.
    """
    return x + np.random.normal(mean, std, x.shape)


def display_scans_cut(
    base_path: str,
    patient_id: int,
    methylation: bool,
    cutoff: float = 0.5,
    scans: Tuple[str] = ("FLAIR", "T1w", "T1wCE", "T2w"),
    target_size: Optional[Tuple[int]] = None,
    voi_lut: bool = True,
) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(20, 15))
    for i, scan in enumerate(scans):
        full_path = os.path.join(base_path, str(patient_id).zfill(5), scan)
        dicom_filepaths = sorted(
            glob.glob(os.path.join(full_path, "*.dcm")),
            key=lambda x: int(x.split("-")[-1][:-4]),
        )
        imag = load_single_file(
            dicom_filepaths[int(cutoff * len(dicom_filepaths))],
            target_size=target_size,
            voi_lut=voi_lut,
        )
        ax[i].imshow(imag, cmap="gray")
        ax[i].set_title(scan)
        ax[i].axis("off")

    plt.suptitle(f"MGMT promoter methylation: {methylation}", fontsize=16, y=0.7)
    plt.show()


def create_animation(
    base_path: str,
    patient_id: int,
    scan: str,
    methylation: bool,
    target_size: Optional[Tuple[int]] = None,
    voi_lut: bool = True,
) -> animation.ArtistAnimation:
    """
    Full credit for this function to ayushn https://www.kaggle.com/code/ayushn2000/eda-primary
    """
    fig = plt.figure(figsize=(15, 12))
    full_scan = load_complete_mri(
        base_path, patient_id, scan, target_size=target_size, voi_lut=voi_lut
    )
    ims = []
    for snapshot in range(0, full_scan.shape[0]):
        im = plt.imshow(full_scan[snapshot, :, :], animated=True, cmap="gray")
        plt.axis("off")
        ims.append([im])

    plt.title(
        f"Patient ID : {str(patient_id).zfill(5)} - MRI : {scan} - Methylation :"
        f" {bool(methylation)}"
    )
    ani = animation.ArtistAnimation(
        fig, ims, interval=100, blit=False, repeat_delay=1000
    )
    plt.close()
    return ani
