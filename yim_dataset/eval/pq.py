from typing import List, Tuple

import torch
import torchmetrics
from torch import Tensor


class PanopticQuality(torchmetrics.detection.PanopticQuality):
    """This class implements the Panoptic Quality for the YIM dataset format."""

    def __init__(
        self,
        things: Tuple[int, ...] = (1, 2),
        stuffs: Tuple[int, ...] = (0,),
    ) -> None:
        """Constructor method.

        Notes:
            Since we are considering the background as a semantic class the trap and cell semantic class shift
            to 1 and 2, respectively.

        Args:
            things (Tuple[int, ...]): Index of things classes. Default (1, 2) both traps and cells.
            stuffs (Tuple[int, ...]): Index of stuff classes. Default (0) only the background.
        """
        # Call super constructor
        super(PanopticQuality, self).__init__(things=things, stuffs=stuffs)

    def update(
        self,
        instances_pred: List[Tensor],
        classes_pred: List[Tensor],
        instances_target: List[Tensor],
        classes_target: List[Tensor],
    ) -> None:
        """Updates the state of the metrix with a new sample.

        Notes:
            instances_pred must be a binary map without overlapping instances.
            classes_pred must entail the semantic class not the logit vector.
            If no instance was detected the respective list entry should be None!
            We assume that at least a single object is present in the label!
            We also assume that the spatial dimensions between the label and the prediction are matching.

        Args:
            instances_pred (List[Tensor]): List of instance masks each of shape [N, H, W].
            classes_pred (List[Tensor]): List of semantic classes each of the shape [N].
            instances_target (List[Tensor]): List of instance mask labels each of shape [N, H, W].
            classes_target (List[Tensor]): List of semantic class labels each of the shape [N].
        """
        # Make semantic label
        semantic_target: Tensor = torch.zeros(
            len(classes_target), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index, classes in enumerate(classes_target):
            # We add one to the semantic class index since we are computing the PQ also over the background
            semantic_target[index] = ((classes + 1).view(-1, 1, 1) * instances_target[index]).sum(dim=0)
        # Make instance label
        instance_map_target: Tensor = torch.zeros(
            len(classes_target), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index in range(len(classes_target)):
            instance_map_target[index] = (
                torch.arange(start=1, end=instances_target[index].shape[0] + 1, step=1).view(-1, 1, 1)
                * instances_target[index]
            ).sum(dim=0)
        # Make semantic prediction
        semantic_pred: Tensor = torch.zeros(
            len(classes_pred), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index, classes in enumerate(classes_pred):
            if classes is not None:
                # We add one to the semantic class index since we are computing the PQ also over the background
                semantic_pred[index] = ((classes + 1).view(-1, 1, 1) * instances_pred[index]).sum(dim=0)
        # Make instance label
        instance_map_pred: Tensor = torch.zeros(
            len(classes_pred), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index in range(len(classes_pred)):
            if classes_pred[index] is not None:
                instance_map_pred[index] = (
                    torch.arange(start=1, end=instances_pred[index].shape[0] + 1, step=1).view(-1, 1, 1)
                    * instances_pred[index]
                ).sum(dim=0)
        # Make prediction and target tensors of shape [B, H, W, 2 (semantic class, instance id)]
        preds: Tensor = torch.stack((semantic_pred, instance_map_pred), dim=-1)
        target: Tensor = torch.stack((semantic_target, instance_map_target), dim=-1)
        # Call super method
        super(PanopticQuality, self).update(preds=preds, target=target)

    def compute(self) -> Tensor:
        """Method computes the PQ by just calling the super method.

        Returns:
            pq (Tensor): Panoptic Quality metric.
        """
        pq: Tensor = super(PanopticQuality, self).compute()
        return pq
