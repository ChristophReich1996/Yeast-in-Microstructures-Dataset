from typing import Callable, List, Optional, Tuple

import os

import kornia.augmentation
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .utils import (
    to_one_hot,
    bounding_box_x0y0x1y1_to_xcycwh,
    absolute_bounding_box_to_relative,
    normalize,
    normalize_0_1,
)


class YIMDataset(Dataset):
    """This class implements the Yeast in Microstructures dataset."""

    def __init__(
        self,
        path: str,
        augmentations: Optional[kornia.augmentation.AugmentationSequential] = None,
        normalize: bool = True,
        normalization_function: Callable[[Tensor], Tensor] = normalize,
        return_absolute_bounding_box: bool = False,
    ) -> None:
        """Constructor method.

        Args:
            path (str): Path to dataset.
            augmentations (Optional[kornia.augmentation.AugmentationSequential]): Augmentations. Default None.
            normalize (bool): If true images are normalized by the given normalization function. Default True.
            normalization_function (Callable[[Tensor], Tensor]): Normalization function. Default 0 mean & 1 std norm.
            return_absolute_bounding_box (bool): If true BBs returned absolut format (else relative). Default False.
        """
        # Call super constructor
        super(YIMDataset, self).__init__()
        # Save parameters
        self.transforms: Optional[kornia.augmentation.AugmentationSequential] = augmentations
        self.normalize = normalize
        self.normalization_function = normalization_function
        self.return_absolute_bounding_box = return_absolute_bounding_box
        # Check augmentations
        self._check_transforms()
        # Get paths of input images
        self.inputs: List[str] = self._get_files(os.path.join(path, "inputs"))
        # Get paths of instances
        self.instances: List[str] = self._get_files(os.path.join(path, "instances"))
        # Get paths of class labels
        self.class_labels: List[str] = self._get_files(os.path.join(path, "classes"))
        # Get paths of bounding boxes
        self.bounding_boxes: List[str] = self._get_files(os.path.join(path, "bounding_boxes"))

    def _check_transforms(self) -> None:
        """Checks if transformations are valid.

        Raises:
            RuntimeError if transformations are not correctly configured.
        """
        # If no transformation is given we have a valid case
        if self.transforms is None:
            return
        # Check if augmentations include all keys
        if (
            (self.transforms.data_keys[0].value == 0)
            and (self.transforms.data_keys[1].value == 3)
            and (self.transforms.data_keys[2].value == 1)
        ):
            return
        raise RuntimeError("Transforms must entail the data keys: [''input'', ''bbox_xyxy'', ''mask''].")

    def _get_files(self, path: str) -> List[str]:
        """Gets all files in a given path.

        Args:
            path (str): Path to search in.

        Returns:
            files (List[str]): List of all files in path.
        """
        files: List[str] = []
        for file in sorted(os.listdir(path)):
            if (not file.startswith(".")) and (os.path.isfile(os.path.join(path, file))):
                files.append(os.path.join(path, file))
        return files

    def __len__(self) -> int:
        """Method returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Method returns an instance of the dataset.

        Notes:
            class_labels is a one-hot vector.
            The semantic class of traps is 0 the semantic class of cells is 1.

        Args:
            item (int): Index of the dataset instance

        Returns:
            image (Tensor): Image if the shape [1, H, W].
            instances (Tensor): Instance maps of the shape [N, H, W].
            bounding_boxes (Tensor): Bounding boxes of the shape [N, 4].
            class_labels (Tensor): Class labels of the shape [N, C].
        """
        # Load data (image, instance maps, bounding boxes, and class labels)
        image: Tensor = torch.load(self.inputs[item], map_location="cpu").unsqueeze(dim=0)
        bounding_boxes: Tensor = torch.load(self.bounding_boxes[item], map_location="cpu")
        instances: Tensor = torch.load(self.instances[item], map_location="cpu")
        class_labels: Tensor = torch.load(self.class_labels[item], map_location="cpu")
        # Ensure image is in range [0, 1]
        image = normalize_0_1(image)
        # Apply transformations
        if self.transforms:
            tensors = self.transforms(image[None], bounding_boxes[None], instances[None])
            image, bounding_boxes, instances = tensors[0][0], tensors[1][0], tensors[2][0]
        # Encode class labels as one-hot
        class_labels = to_one_hot(class_labels, num_classes=2)
        # Normalize image if utilized
        if self.normalize:
            image = self.normalization_function(image)
        # Convert absolute bounding box to relative bounding box of utilized
        if not self.return_absolute_bounding_box:
            bounding_boxes = absolute_bounding_box_to_relative(
                bounding_boxes=bounding_boxes, height=image.shape[1], width=image.shape[2]
            )
        return image, instances, bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes), class_labels


def collate_function_yim_dataset(
    batch: List[Tuple[Tensor]],
) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
    """Custom collate function for YIM dataset.

    Args:
        batch (Tuple[Iterable[Tensor], Iterable[Tensor], Iterable[Tensor], Iterable[Tensor]]):
            Batch of input data, instances maps, bounding boxes and class labels

    Returns:
        images (Tensor): Batched images of the shape [B, 1, H, W]
        instances (List[Tensor]): List of instance maps as tensors with shape [B, H, W] each.
        bounding_boxes (List[Tensor]): Bounding boxes as a list of tensors with shape [N, 4] each.
        class_labels (List[Tensor]): Class labels as a list of tensors with shape [N, C].
    """
    return (
        torch.stack([input_samples[0] for input_samples in batch], dim=0),
        [input_samples[1] for input_samples in batch],
        [input_samples[2] for input_samples in batch],
        [input_samples[3] for input_samples in batch],
    )
